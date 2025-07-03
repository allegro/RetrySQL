from faiss import Index
from pydash import chain

from text2sql.datasets.domain.model import SchemaLink, SchemaLinkingDocumentRawData
from text2sql.modules.schema_linking.domain.model import Embedding, EmbeddingJobExample, SchemaLinkingOutput
from text2sql.modules.schema_linking.repository.index_repository import IndexSearchRepository
from text2sql.modules.schema_linking.schema_linkers.schema_linker_base import SchemaLinkerBase


class SchemaLinkerNearestNeighborSearch(SchemaLinkerBase):
    def forward(
        self,
        embedded_queries: list[Embedding],
        raw_queries: list[EmbeddingJobExample],
        all_documents: list[SchemaLinkingDocumentRawData],
        search_index: Index,
        max_neighbour_count: int,
    ) -> list[SchemaLinkingOutput]:
        repository = IndexSearchRepository(search_index, max_neighbour_count)

        output_list = []

        for query in embedded_queries:
            result = repository.search_single(query.embedding)

            matched_documents = chain(all_documents).filter(lambda document: document.document_id in result).value()

            documents_grouped_by_table = chain(matched_documents).group_by(lambda document: document.table_name).value()

            query_question = (
                chain(raw_queries)
                .filter(lambda raw_query: raw_query.example_id == query.example_id)
                .head()
                .value()
                .input_to_embed
            )

            output_list.append(
                SchemaLinkingOutput(
                    question_id=query.example_id,
                    question=query_question,
                    schema_links=[
                        SchemaLink(table_name=table_name, columns=[document.col_name for document in documents])
                        for table_name, documents in documents_grouped_by_table.items()
                    ],
                )
            )

        return output_list
