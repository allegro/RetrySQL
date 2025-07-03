import abc

from pydantic import BaseModel


class BaseDataset(abc.ABC):

    @abc.abstractmethod
    def load_dev(self) -> list[BaseModel]: ...

    @abc.abstractmethod
    def load_train(self) -> list[BaseModel]: ...

    @abc.abstractmethod
    def load_test(self) -> list[BaseModel]: ...
