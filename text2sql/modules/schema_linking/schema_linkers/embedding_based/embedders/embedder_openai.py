import asyncio
import json

import numpy as np
import openai
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_core.language_models.llms import create_base_retry_decorator
from upath import UPath

from text2sql.commons.io_utils import read_yaml, save_embeddings
from text2sql.commons.logging_utils import get_logger
from text2sql.llms.domain.models import OpenAIEmbeddingModelEndpoint
from text2sql.modules.schema_linking.configuration import EmbeddingsJobConfiguration
from text2sql.modules.schema_linking.domain.enums import AvailableEmbeddingModels
from text2sql.modules.schema_linking.domain.model import Embedding, EmbeddingJobExample
from text2sql.modules.schema_linking.schema_linkers.embedding_based.embedders.embbeder_base import EmbedderBase

OPENAI_MODELS_NAMING_MAPPING = {
    AvailableEmbeddingModels.TEXT_EMBEDDING_3_LARGE: "text-embedding-3-large",
    AvailableEmbeddingModels.TEXT_EMBEDDING_ADA_002: "text-embedding-ada-002",
}

logger = get_logger(__name__)


class EmbedderOpenAI(EmbedderBase):
    def __init__(self, embedding_job_config: EmbeddingsJobConfiguration) -> None:
        super().__init__(embedding_job_config=embedding_job_config)
        self._model_config = self._config.embedding_model_configuration
        self._endpoint_config = self._parse_endpoint_config()

        self._embedding_model = self._setup_embedding_model()

        self._setup_retry_mechanism_for_embed_single_method()
        self._semaphore = asyncio.Semaphore(self._endpoint_config.max_concurrency)

    def _setup_retry_mechanism_for_embed_single_method(self) -> None:
        self._embed_single = create_base_retry_decorator(
            error_types=[
                openai.error.RateLimitError,
                openai.error.APIError,
                openai.error.Timeout,
                openai.error.APIConnectionError,
                openai.error.ServiceUnavailableError,
            ],
            max_retries=self._endpoint_config.max_retries,
        )(self._embed_single)

    def _parse_endpoint_config(self) -> OpenAIEmbeddingModelEndpoint:
        mapped_model_name = OPENAI_MODELS_NAMING_MAPPING[self._model_config.model_name]
        llm_config = read_yaml(UPath(self._model_config.path_to_endpoint_config)).pop(mapped_model_name)
        return OpenAIEmbeddingModelEndpoint.parse_obj(llm_config)

    def _setup_embedding_model(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            deployment=self._endpoint_config.deployment,
            model=self._endpoint_config.model,
            openai_api_base=self._endpoint_config.openai_api_base,
            openai_api_type=self._endpoint_config.openai_api_type,
            openai_api_key=self._endpoint_config.openai_api_key,
            request_timeout=self._endpoint_config.request_timeout,
        )

    async def _embed_single(self, example: EmbeddingJobExample) -> Embedding:
        async with self._semaphore:
            embedding = await self._embedding_model.aembed_documents(
                [example.input_to_embed], chunk_size=self._model_config.batch_size
            )
            return Embedding(example_id=example.example_id, embedding=np.array(embedding[0], dtype=np.float32))

    async def _embed_async(self, document_or_query_representations: list[EmbeddingJobExample]) -> list[Embedding]:
        tasks = [self._embed_single(example) for example in document_or_query_representations]
        return await asyncio.gather(*tasks)

    def embed(
        self, document_or_query_representations: list[EmbeddingJobExample], is_query: bool = False
    ) -> list[Embedding]:
        loop = asyncio.get_event_loop()

        target_staging_dir = self._model_config.staging_output_dir.joinpath("queries" if is_query else "documents")
        sorted_document_or_query_representations = sorted(document_or_query_representations, key=lambda x: x.example_id)
        for chunk_index in range(0, len(sorted_document_or_query_representations), self._model_config.chunk_size):
            chunk = sorted_document_or_query_representations[chunk_index : chunk_index + self._model_config.chunk_size]

            if self._is_chunk_already_dumped(target_staging_dir, chunk_index):
                logger.info(f"Chunk {chunk_index} already dumped, skipping...")
                continue

            embeddings = loop.run_until_complete(self._embed_async(chunk))
            save_embeddings(embeddings, target_staging_dir.joinpath(f"chunk_{chunk_index}.json"))

        return self._gather_all_embeddings(target_staging_dir)

    @staticmethod
    def _is_chunk_already_dumped(target_staging_dir: UPath, chunk_index: int) -> bool:
        return target_staging_dir.joinpath(f"chunk_{chunk_index}.json").exists()

    @staticmethod
    def _gather_all_embeddings(target_staging_dir: UPath) -> list[Embedding]:
        all_embeddings = []

        for chunk_file in target_staging_dir.iterdir():
            if chunk_file.name.startswith("chunk_"):
                with chunk_file.open("r") as f:
                    chunk_embeddings = json.load(f)
                    all_embeddings.extend([Embedding(**embedding_dict) for embedding_dict in chunk_embeddings])

        return all_embeddings
