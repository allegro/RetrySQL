from dataclasses import dataclass
from functools import partial

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from text2sql.commons.logging_utils import get_logger
from text2sql.modules.schema_linking.configuration import EmbeddingsJobConfiguration
from text2sql.modules.schema_linking.domain.enums import AvailableEmbeddingModels
from text2sql.modules.schema_linking.domain.model import Embedding, EmbeddingJobExample
from text2sql.modules.schema_linking.schema_linkers.embedding_based.embedders.embbeder_base import EmbedderBase

logger = get_logger(__name__)


HF_TRANSFORMERS_MODELS_NAMING_MAPPING = {
    AvailableEmbeddingModels.SNOWFLAKE_ARCTIC_EMBED_M: "Snowflake/snowflake-arctic-embed-m",
    AvailableEmbeddingModels.SNOWFLAKE_ARCTIC_EMBED_M_LONG: "Snowflake/snowflake-arctic-embed-m-long",
    AvailableEmbeddingModels.SNOWFLAKE_ARCTIC_EMBED_L: "Snowflake/snowflake-arctic-embed-l",
}


@dataclass
class Batch:
    example_ids: list[int]
    input_tokens: torch.Tensor


class EmbedderHuggingFaceTransformers(EmbedderBase):
    def __init__(self, embedding_job_config: EmbeddingsJobConfiguration) -> None:
        super().__init__(embedding_job_config=embedding_job_config)
        self._embedding_config = self._config.embedding_model_configuration

        self._embedding_model, self._tokenizer = self._setup_model_and_tokenizer()
        self._embedding_model.eval()

        # For optimal retrieval quality, use the CLS token to embed each text portion and use the query prefix below (just on the query)
        # Source: https://github.com/Snowflake-Labs/arctic-embed?tab=readme-ov-file#using-huggingface-transformers
        self._query_prefix = "Represent this sentence for searching relevant passages: "

        self._max_supported_sequence_length = 2048
        self._normalization_exponent_value = 2
        self._normalization_dimension = 1

    def _setup_model_and_tokenizer(self) -> tuple[AutoModel, AutoTokenizer]:
        mapped_model_name = HF_TRANSFORMERS_MODELS_NAMING_MAPPING[self._embedding_config.model_name]
        if (
            self._embedding_config.model_name == AvailableEmbeddingModels.SNOWFLAKE_ARCTIC_EMBED_M_LONG
            and self._embedding_config.max_sequence_length > self._max_supported_sequence_length
        ):
            # If you use the long context model with more than 2048 tokens, ensure that you initialize the model with RoPE
            # source: https://github.com/Snowflake-Labs/arctic-embed?tab=readme-ov-file#usage-note-long-context-embedding-with-m-long
            model = AutoModel.from_pretrained(
                mapped_model_name, trust_remote_code=True, safe_serialization=True, rotary_scaling_factor=2
            )
        else:
            model = AutoModel.from_pretrained(mapped_model_name)

        tokenizer = AutoTokenizer.from_pretrained(mapped_model_name)
        model.eval()

        return model, tokenizer

    def _build_batch(self, batch: list[EmbeddingJobExample], is_query: bool) -> Batch:
        ids = [example.example_id for example in batch]
        texts = [
            f"{self._query_prefix}{example.input_to_embed}" if is_query else example.input_to_embed for example in batch
        ]
        tokenized_texts = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self._embedding_config.max_sequence_length,
        )
        return Batch(example_ids=ids, input_tokens=tokenized_texts)

    def _get_dataloader(
        self, document_or_query_representations: list[EmbeddingJobExample], is_query: bool
    ) -> DataLoader:
        return DataLoader(
            document_or_query_representations,
            batch_size=self._embedding_config.batch_size,
            collate_fn=partial(self._build_batch, is_query=is_query),
            num_workers=self._embedding_config.num_workers,
        )

    def _get_embeddings(self, batch: Batch) -> npt.NDArray[np.float32]:
        batch_embeddings = self._embedding_model(**batch.input_tokens)
        cls_token_embeddings = batch_embeddings[0][:, 0]
        normalized_embeddings = (
            torch.nn.functional.normalize(
                cls_token_embeddings, p=self._normalization_exponent_value, dim=self._normalization_dimension
            )
            .cpu()
            .numpy()
        )
        return normalized_embeddings

    def embed(
        self, document_or_query_representations: list[EmbeddingJobExample], is_query: bool = False
    ) -> list[Embedding]:
        embeddings = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(self._get_dataloader(document_or_query_representations, is_query=is_query))
            ):
                logger.info("Embedding batch idx: %d", batch_idx)
                batch_embeddings = self._get_embeddings(batch)
                for embedding, example_id in zip(batch_embeddings, batch.example_ids):
                    embeddings.append(Embedding(example_id=example_id, embedding=embedding))

        return embeddings
