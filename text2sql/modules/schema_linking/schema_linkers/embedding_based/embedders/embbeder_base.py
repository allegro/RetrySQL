from abc import ABC

import numpy as np

from text2sql.modules.schema_linking.configuration import EmbeddingsJobConfiguration
from text2sql.modules.schema_linking.domain.model import Embedding, EmbeddingJobExample


class EmbedderBase(ABC):
    def __init__(self, embedding_job_config: EmbeddingsJobConfiguration) -> None:
        self._config = embedding_job_config

    def embed(
        self, document_or_query_representations: list[EmbeddingJobExample], is_query: bool = False
    ) -> list[Embedding]:
        pass


class DummyEmbedder(EmbedderBase):
    def __init__(self, embedding_job_config: EmbeddingsJobConfiguration) -> None:
        super().__init__(embedding_job_config=embedding_job_config)
        self._embedding_size = embedding_job_config.embedding_model_configuration.embedding_size

    def embed(
        self,
        document_or_query_representations: list[EmbeddingJobExample],
        is_query: bool = False,
    ) -> list[Embedding]:
        return [
            Embedding(example_id=example.example_id, embedding=np.random.rand(self._embedding_size))
            for example in document_or_query_representations
        ]
