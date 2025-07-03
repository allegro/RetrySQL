import numpy as np

from text2sql.modules.schema_linking.domain.model import Embedding
from text2sql.modules.schema_linking.repository.index_builder import IndexBuilder
from text2sql.modules.schema_linking.repository.index_repository import IndexSearchRepository


class TestSearchIndex:
    def test_search_single(self):
        # GIVEN
        embeddings = [
            Embedding(example_id=123, embedding=np.array([1, 2, 3])),
            Embedding(example_id=321, embedding=np.array([3, 2, 1])),
        ]

        index_builder = IndexBuilder(embedding_size=3)
        search_index = index_builder.add_embeddings(embeddings).build()

        index_repository = IndexSearchRepository(search_index=search_index, max_neighbour_count=1)

        # WHEN
        found_document_ids = index_repository.search_single(embedding_vector=np.array([1, 2, 3]))

        # THEN
        assert found_document_ids == [123]

    def test_search_batch(self):
        # GIVEN
        embeddings = [
            Embedding(example_id=123, embedding=np.array([1, 2, 3])),
            Embedding(example_id=321, embedding=np.array([3, 2, 1])),
            Embedding(example_id=234, embedding=np.array([2, 3, 4])),
        ]

        index_builder = IndexBuilder(embedding_size=3)
        search_index = index_builder.add_embeddings(embeddings).build()

        index_repository = IndexSearchRepository(search_index=search_index, max_neighbour_count=1)

        # WHEN
        found_document_ids = index_repository.search_batch(embedding_vectors=[np.array([1, 2, 3]), np.array([2, 3, 4])])

        # THEN
        assert found_document_ids == [[123], [234]]
