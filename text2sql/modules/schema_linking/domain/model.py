from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel

from text2sql.datasets.domain.model import DatabaseSchema, SchemaLink, SchemaLinkingDocumentRawData


class SchemaLinkerHeuristicBasedAlgorithmInputExample(BaseModel):
    question_id: int
    question: str
    database_schema: DatabaseSchema
    external_knowledge: str | None = None


class SchemaLinkingOutput(BaseModel):
    question_id: int
    question: str
    schema_links: list[SchemaLink] | None = None


class EmbeddingJobExample(BaseModel):
    example_id: int
    input_to_embed: str | None = None


@dataclass
class Embedding:
    example_id: int
    embedding: npt.NDArray[np.float32]


@dataclass
class QueryAndDocumentProcessorOutput:
    processed_queries: list[EmbeddingJobExample]
    processed_documents: list[EmbeddingJobExample]
    raw_documents: list[SchemaLinkingDocumentRawData]


@dataclass
class QueryAndDocumentEmbeddings:
    query_embeddings: list[Embedding]
    document_embeddings: list[Embedding]
