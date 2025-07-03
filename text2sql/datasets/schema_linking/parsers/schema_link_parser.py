from pydash import chain
from sqlglot.expressions import Column

from text2sql.datasets.domain.model import (
    DatabaseSchema,
    SchemaColumn,
    SchemaLink,
    SchemaLinkingDataSample,
    TableColumnPair,
    TableWithAlias,
    Text2SqlDataSample,
)


class SchemaLinkParser:
    EMPTY_NAME = ""

    def run(
        self,
        bird_ground_truth_data: list[Text2SqlDataSample],
        bird_metadata: list[DatabaseSchema],
    ) -> list[SchemaLinkingDataSample]:
        results = []

        for ground_truth_example in bird_ground_truth_data:
            metadata_for_query_database = (
                chain(bird_metadata)
                .filter(lambda database_metadata: database_metadata.database_name == ground_truth_example.database_name)
                .head()
                .value()
            )

            schema_links = []

            for table_name, columns in metadata_for_query_database.tables.items():
                column_links = (
                    chain(columns)
                    .filter(lambda column: self._is_column_in_query(
                        table_name=table_name,
                        column_name=column.column_name,
                        query=ground_truth_example.sql_query
                    ))
                    .map(lambda column: column.column_name)
                    .value()
                )

                if len(column_links) == 0:
                    continue

                schema_links.append(
                    SchemaLink(
                        table_name=table_name,
                        columns=list(sorted(column_links))
                    )
                )

            results.append(
                SchemaLinkingDataSample(
                    question_id=ground_truth_example.question_id,
                    question=ground_truth_example.question,
                    schema_links=schema_links
                )
            )

        return results
    
    @staticmethod
    def _is_column_in_query(table_name: str, column_name: str, query: str) -> bool:
        return (
            column_name.lower() in query.lower() 
            and table_name.lower() in query.lower()
        )

    @staticmethod
    def _find_table_name(candidate_name: str, query_tables: list[TableWithAlias]) -> str:
        found_name = chain(query_tables).filter(lambda query_table: query_table.name == candidate_name).value()

        if len(found_name) != 0:
            return candidate_name

        found_alias = chain(query_tables).filter(lambda query_table: query_table.alias == candidate_name).value()

        if len(found_alias) != 0:
            return found_alias[0].name

        return ""

    def _identify_columns_with_known_table(self, table_candidates: list[TableColumnPair]) -> list[TableColumnPair]:
        return (
            chain(table_candidates)
            .filter(lambda table_column_pair: table_column_pair.table_name != self.EMPTY_NAME)
            .value()
        )

    def _identify_columns_without_table(
        self,
        query_columns: list[Column],
        table_candidates: list[TableColumnPair],
    ) -> list[str]:
        column_names_without_table = (
            chain(query_columns)
            .filter(lambda column: column.table == self.EMPTY_NAME)
            .map(lambda column: column.name.lower())
            .uniq()
            .value()
        )

        empty_candidate_columns = (
            chain(table_candidates)
            .filter(lambda table_column_pair: table_column_pair.table_name == self.EMPTY_NAME)
            .map(lambda x: x.expanded_column.column_name.lower())
            .value()
        )

        return column_names_without_table + empty_candidate_columns

    @staticmethod
    def _find_required_columns(
        query_tables: list[TableWithAlias],
        database_metadata: DatabaseSchema,
        column_names_without_table: list[str],
        columns_with_known_table: list[TableColumnPair],
    ) -> dict[str, set[str]]:
        all_required_columns = {}
        query_table_names = [query_table.name for query_table in query_tables]

        for table_name, columns in database_metadata.tables.items():
            if table_name not in query_table_names:
                continue

            required_columns = set([column.column_name for column in columns]).intersection(column_names_without_table)

            if len(required_columns) == 0:
                continue

            all_required_columns[table_name] = required_columns

        for column_with_known_table in columns_with_known_table:
            if column_with_known_table.table_name not in all_required_columns:
                all_required_columns[column_with_known_table.table_name] = set()
            all_required_columns[column_with_known_table.table_name].add(
                column_with_known_table.expanded_column.column_name
            )

        return all_required_columns
