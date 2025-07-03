import csv
import os
import sqlite3

from pydash import chain
from upath import UPath

from text2sql.commons.io_utils import save_objects_to_jsonl_file
from text2sql.commons.logging_utils import get_logger
from text2sql.datasets.domain.model import (
    ColumnDescriptor,
    DatabaseColumn,
    SchemaLinkingDocumentRawData,
    SchemaLinkingQueryRawData,
)
from text2sql.datasets.schema_linking.parsers.bird_dataset_parsing_utils import (
    read_bird_database_schemas,
    read_bird_ground_truth,
)

logger = get_logger(__name__)


class SchemaLinkingDataDumperForRetrieval:

    def __init__(
        self,
        bird_ground_truth_path: UPath,
        bird_metadata_path: UPath,
        database_root_path: UPath,
        output_dir_path: UPath,
    ) -> None:
        self._database_root_path = database_root_path
        self._output_dir_path = output_dir_path

        self._bird_ground_truth = read_bird_ground_truth(bird_ground_truth_path)
        self._bird_database_schemas = read_bird_database_schemas(bird_metadata_path)

    def _load_queries(self) -> list[SchemaLinkingQueryRawData]:
        logger.info("Dumping schema linking raw data for queries")
        return (
            chain(self._bird_ground_truth)
            .map(
                lambda example: SchemaLinkingQueryRawData(
                    question_id=example.question_id,
                    question=example.question,
                    external_knowledge=example.knowledge,
                )
            )
            .value()
        )

    @staticmethod
    def _get_column_describer(table_description_csv_file_path: UPath) -> dict[str, ColumnDescriptor]:
        column_describer = {}
        with table_description_csv_file_path.open("r", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                normalized_row = {k: v.strip().lower() for k, v in row.items()}
                column_descriptor = ColumnDescriptor.parse_obj(normalized_row)
                column_describer[column_descriptor.original_column_name] = column_descriptor
        return column_describer

    def _get_flattened_database_columns(self) -> list[DatabaseColumn]:
        return [
            DatabaseColumn(database_name=database.database_name, table_name=table_name, column=column)
            for database in self._bird_database_schemas
            for table_name, columns in database.tables.items()
            for column in columns
        ]

    def _load_documents(self) -> list[SchemaLinkingDocumentRawData]:
        logger.info("Dumping schema linking raw data for documents")

        flattened_database_columns = self._get_flattened_database_columns()

        documents = []
        for document_id, database_column in enumerate(flattened_database_columns):
            column_describer = self._get_column_describer(
                table_description_csv_file_path=self._database_root_path.joinpath(
                    database_column.database_name, "database_description", f"{database_column.table_name}.csv"
                ),
            )
            values = self._retrieve_column_values(
                database_column.database_name, database_column.table_name, database_column.column.column_name
            )
            documents.append(
                SchemaLinkingDocumentRawData(
                    document_id=document_id,
                    table_name=database_column.table_name,
                    col_name=database_column.column.column_name,
                    is_primary_key=database_column.column.is_primary_key,
                    is_foreign_key=database_column.column.is_foreign_key,
                    col_description=column_describer[database_column.column.column_name].column_description,
                    col_data_format=column_describer[database_column.column.column_name].data_format,
                    col_values=values,
                )
            )
        return documents

    def _retrieve_column_values(self, database_name: str, table_name: str, column_name: str) -> list[str]:
        path = os.path.join(self._database_root_path, database_name, f"{database_name}.sqlite")
        db_connector = sqlite3.connect(path)
        cursor = db_connector.cursor()
        cursor.execute(f'SELECT "{column_name}" FROM "{table_name}"')
        values = cursor.fetchall()
        return list(set([value[0] for value in values]))

    def dump(self) -> None:
        queries = self._load_queries()
        logger.info(f"Retrieved {len(queries)} queries")
        save_objects_to_jsonl_file([query.dict() for query in queries], self._output_dir_path.joinpath("queries.jsonl"))

        documents = self._load_documents()
        logger.info(f"Retrieved {len(documents)} documents")
        save_objects_to_jsonl_file([doc.dict() for doc in documents], self._output_dir_path.joinpath("documents.jsonl"))
