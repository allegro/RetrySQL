import numpy as np
import pytest

from text2sql.datasets.domain.model import SchemaLink, SchemaLinkingDocumentRawData
from text2sql.modules.schema_linking.domain.model import Embedding, EmbeddingJobExample, SchemaLinkingOutput
from text2sql.modules.schema_linking.repository.index_builder import IndexBuilder
from text2sql.modules.schema_linking.schema_linkers.embedding_based.schema_linker_nn_search import (
    SchemaLinkerNearestNeighborSearch,
)


class TestEmbeddingBasedSchemaLinking:
    @pytest.mark.parametrize(
        "question, max_neighbour_count, expected_linked_columns",
        [
            ("Who is the cook with the longest name?", 1, ["name"]),
            ("Who is the highest cook in the kitchen?", 2, ["name", "height"]),
        ],
        ids=["Single column match", "Multiple columns in one table"],
    )
    def test_linking_max_less_than_number_of_documents(
        self, question: str, max_neighbour_count: int, expected_linked_columns: list[str]
    ) -> None:
        # GIVEN
        all_raw_documents = [
            SchemaLinkingDocumentRawData(
                document_id=123, table_name="cooks", col_name="name", is_primary_key=False, is_foreign_key=False
            ),
            SchemaLinkingDocumentRawData(
                document_id=122, table_name="cooks", col_name="height", is_primary_key=False, is_foreign_key=False
            ),
            SchemaLinkingDocumentRawData(
                document_id=321, table_name="kitchens", col_name="address", is_primary_key=False, is_foreign_key=False
            ),
        ]

        document_search_index = (
            IndexBuilder(embedding_size=3)
            .add_embeddings(
                [
                    Embedding(example_id=123, embedding=np.array([1, 2, 3])),
                    Embedding(example_id=122, embedding=np.array([1, 2, 2])),
                    Embedding(example_id=321, embedding=np.array([3, 2, 1])),
                ]
            )
            .build()
        )

        raw_queries = [EmbeddingJobExample(example_id=1, input_to_embed=question)]
        embedded_queries = [Embedding(example_id=1, embedding=np.array([1, 2, 3]))]

        schema_linker = SchemaLinkerNearestNeighborSearch(use_external_knowledge=False)

        # WHEN
        schema_linking_outputs = schema_linker.forward(
            embedded_queries=embedded_queries,
            raw_queries=raw_queries,
            all_documents=all_raw_documents,
            search_index=document_search_index,
            max_neighbour_count=max_neighbour_count,
        )

        # THEN
        assert schema_linking_outputs == [
            SchemaLinkingOutput(
                question_id=1,
                question=question,
                schema_links=[SchemaLink(table_name="cooks", columns=expected_linked_columns)],
            )
        ]

    def test_linking_max_more_than_number_of_documents(
        self,
    ) -> None:
        # GIVEN
        all_raw_documents = [
            SchemaLinkingDocumentRawData(
                document_id=123, table_name="cooks", col_name="name", is_primary_key=False, is_foreign_key=False
            ),
            SchemaLinkingDocumentRawData(
                document_id=122, table_name="cooks", col_name="height", is_primary_key=False, is_foreign_key=False
            ),
            SchemaLinkingDocumentRawData(
                document_id=321, table_name="kitchens", col_name="address", is_primary_key=False, is_foreign_key=False
            ),
        ]

        document_search_index = (
            IndexBuilder(embedding_size=3)
            .add_embeddings(
                [
                    Embedding(example_id=123, embedding=np.array([1, 2, 3])),
                    Embedding(example_id=122, embedding=np.array([1, 2, 2])),
                    Embedding(example_id=321, embedding=np.array([3, 2, 1])),
                ]
            )
            .build()
        )

        raw_queries = [EmbeddingJobExample(example_id=1, input_to_embed="Who is the highest cook in the kitchen?")]
        embedded_queries = [Embedding(example_id=1, embedding=np.array([1, 2, 3]))]

        schema_linker = SchemaLinkerNearestNeighborSearch(use_external_knowledge=False)

        # WHEN
        schema_linking_outputs = schema_linker.forward(
            embedded_queries=embedded_queries,
            raw_queries=raw_queries,
            all_documents=all_raw_documents,
            search_index=document_search_index,
            max_neighbour_count=10,
        )

        # THEN
        assert schema_linking_outputs == [
            SchemaLinkingOutput(
                question_id=1,
                question="Who is the highest cook in the kitchen?",
                schema_links=[
                    SchemaLink(table_name="cooks", columns=["name", "height"]),
                    SchemaLink(table_name="kitchens", columns=["address"]),
                ],
            )
        ]
