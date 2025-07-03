from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from text2sql.datasets.domain.model import PretrainDataSample
from text2sql.llm_training.domain.enums import ModelExecutionModes

T = TypeVar('T')


class BaseDataset(Dataset, ABC, Generic[T]):
    def __init__(
            self,
            mode: ModelExecutionModes,
            tokenizer: PreTrainedTokenizer,
            input_data: list[PretrainDataSample]
    ) -> None:
        super().__init__()

        self._tokenizer = tokenizer
        self._mode = mode
        self.tokenized_data = self._tokenize_dataset(input_data)

    @abstractmethod
    def _tokenize_dataset(
            self,
            input_data: list[PretrainDataSample]
    ) -> list[T]:
        pass
