import random
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from pydash import chain
from sqlglot.expressions import Column, Table, replace_tables, Expression, column

from text2sql.datasets.domain.model import CorruptedSample, DatabaseSchema, DatabaseTablesAndColumns, \
    Text2SqlDataSample

T = TypeVar('T', bound=Text2SqlDataSample)


class AbstractErrorGenerator(ABC, Generic[T]):
    @abstractmethod
    def corrupt(
            self,
            data_sample: T,
            probability: float,
            bird_metadata: list[DatabaseSchema]
    ) -> CorruptedSample:
        ...


class SqlBasedErrorGenerator(AbstractErrorGenerator[Text2SqlDataSample], ABC):
    @staticmethod
    def _extract_all_tables_and_columns_for_given_database(
            database_name: str,
            bird_metadata: list[DatabaseSchema]
    ) -> DatabaseTablesAndColumns:
        full_db_schema = (
            chain(bird_metadata)
            .filter(lambda database_schema: database_schema.database_name == database_name)
            .head()
            .value()
        )

        all_tables = list(full_db_schema.tables.keys())
        all_columns = (
            chain(full_db_schema.tables.values())
            .flatten()
            .map(lambda column: column.column_name)
            .uniq()
            .value()
        )

        return DatabaseTablesAndColumns(
            tables=all_tables,
            columns=all_columns
        )

    def _traverse_tree_and_corrupt(
            self,
            parsed_expression: Expression,
            probability: float,
            all_tables_and_columns: DatabaseTablesAndColumns,
    ) -> Expression:
        corrupted_expression = parsed_expression.copy()

        for expression in corrupted_expression.dfs():
            if self._is_table(expression) and random.random() < probability:
                corrupted_table_name = self._draw_random_element_with_exception(
                    list_to_draw_from=all_tables_and_columns.tables,
                    element_to_skip=expression.name
                )

                new_expression = replace_tables(
                    expression=expression.parent,
                    mapping={
                        f"`{expression.name}`": f"{self._corrupt_name(expression.name, corrupted_table_name)}"
                    },
                    dialect="sqlite"
                )
                new_expression.comments = None
                expression.parent.replace(new_expression)

            if self._is_column(expression) and random.random() < probability:
                corrupted_column_name = self._draw_random_element_with_exception(
                    list_to_draw_from=all_tables_and_columns.columns,
                    element_to_skip=expression.name
                )

                new_expression = expression.parent.replace(
                    column(f"{self._corrupt_name(expression.name, corrupted_column_name)}"))
                new_expression.comments = None
                expression.parent.replace(new_expression)

        return corrupted_expression

    def _corrupt_name(self, original_name: str, corrupted_name: str) -> str:
        raise NotImplementedError()

    @staticmethod
    def _is_table(expression: Expression) -> bool:
        return expression.is_leaf() and expression.parent.find(Table) is not None and expression.parent.find(
            Table).name == expression.name

    @staticmethod
    def _is_column(expression: Expression) -> bool:
        return expression.is_leaf() and expression.parent.find(Column) is not None and expression.parent.find(
            Column).name == expression.name

    @staticmethod
    def _draw_random_element_with_exception(list_to_draw_from: list[str], element_to_skip: str) -> str:
        return (
            chain(list_to_draw_from)
            .filter(lambda element: element != element_to_skip)
            .sample()
            .value()
        )
