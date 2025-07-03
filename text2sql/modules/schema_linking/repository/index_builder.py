from __future__ import annotations

import numpy as np
from faiss import Index, IndexFlatL2, IndexIDMap

from text2sql.modules.schema_linking.domain.model import Embedding


class IndexBuilder:
    def __init__(self, embedding_size: int) -> None:
        underlying_index = IndexFlatL2(embedding_size)
        self._search_index = IndexIDMap(underlying_index)

    def add_embeddings(self, embeddings: list[Embedding]) -> IndexBuilder:
        for embedding in embeddings:
            self._search_index.add_with_ids(np.array([embedding.embedding]), np.array([embedding.example_id]))

        return self

    def build(self) -> Index:
        return self._search_index
