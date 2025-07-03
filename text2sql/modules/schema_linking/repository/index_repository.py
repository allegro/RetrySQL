import numpy as np
import numpy.typing as npt
from faiss import Index


class IndexSearchRepository:
    def __init__(self, search_index: Index, max_neighbour_count: int) -> None:
        self._search_index = search_index
        self._max_neighbour_count = max_neighbour_count

    def search_single(self, embedding_vector: npt.NDArray[np.float32]) -> list[int]:
        _, found_ids = self._search_index.search(x=np.array([embedding_vector]), k=self._max_neighbour_count)

        return found_ids.flatten()

    def search_batch(self, embedding_vectors: list[npt.NDArray[np.float32]]) -> list[list[int]]:
        return [self.search_single(embedding_vector) for embedding_vector in embedding_vectors]
