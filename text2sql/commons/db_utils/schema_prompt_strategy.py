from abc import ABC, abstractmethod

from text2sql.commons.db_utils.domain.model import DatabaseInfo
from text2sql.commons.logging_utils import get_logger
from text2sql.datasets.domain.model import SchemaLink

logger = get_logger(__name__)


class SchemaPromptStrategy(ABC):

    @abstractmethod
    def create_schema_prompt(
        self,
        db_info: DatabaseInfo,
        schema_links: list[SchemaLink],
        shuffle_cols: bool = False,
    ) -> str:
        raise NotImplementedError()
