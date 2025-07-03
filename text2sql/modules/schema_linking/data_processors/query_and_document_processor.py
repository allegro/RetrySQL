from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import regex as re

from text2sql.commons.io_utils import read_jsonl
from text2sql.commons.logging_utils import get_logger
from text2sql.datasets.domain.enums import BirdSupportedColumnDataFormats
from text2sql.datasets.domain.model import SchemaLinkingDocumentRawData, SchemaLinkingQueryRawData
from text2sql.modules.schema_linking.configuration import EmbeddingsJobConfiguration
from text2sql.modules.schema_linking.data_processors.base_processor import BaseSchemaLinkingDataProcessor
from text2sql.modules.schema_linking.domain.model import EmbeddingJobExample, QueryAndDocumentProcessorOutput

logger = get_logger(__name__)

STRING_SEPARATOR = " "
DEFAULT_NUMBER_OF_MOST_COMMON_VALUES = 3
NUMERICAL_DATA_FORMAT_ALIASES = [
    BirdSupportedColumnDataFormats.INTEGER,
    BirdSupportedColumnDataFormats.REAL,
    BirdSupportedColumnDataFormats.DATETIME,
    BirdSupportedColumnDataFormats.DATE,
]
# source: https://stackoverflow.com/questions/136505/searching-for-uuids-in-text-with-regex
UUID_REGEX_EXPRESSION = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


class QueryAndDocumentProcessor(BaseSchemaLinkingDataProcessor):
    def __init__(self, config: EmbeddingsJobConfiguration) -> None:
        self._config = config

    @staticmethod
    def _normalize_text(text: str) -> str:
        return text.strip().lower()

    def get_query_representation(self, raw_query_data: SchemaLinkingQueryRawData) -> str:
        query_representation = self._normalize_text(raw_query_data.question)
        if self._config.query_configuration.include_external_knowledge:
            query_representation += f"{STRING_SEPARATOR}{self._normalize_text(raw_query_data.external_knowledge)}"
        return query_representation

    def _process_queries(self) -> list[EmbeddingJobExample]:

        raw_queries = read_jsonl(self._config.path_to_raw_queries, SchemaLinkingQueryRawData)

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(
                    lambda raw_query_data: EmbeddingJobExample(
                        example_id=raw_query_data.question_id,
                        input_to_embed=self.get_query_representation(raw_query_data),
                    ),
                    raw_queries,
                )
            )
        logger.info(f"Number of queries before processing {len(raw_queries)} - After: {len(results)}")
        return results

    def _serialize_column_values_to_str(self, column_values: list[Any]) -> str:
        string_values = STRING_SEPARATOR.join(map(str, column_values))
        return self._normalize_text(string_values)

    def _column_should_be_excluded(self, raw_document_data: SchemaLinkingDocumentRawData) -> bool:

        def _does_column_contain_uuid_like_values(values: list[Any]) -> bool:
            values_representative_string = str(values[0])
            return re.search(UUID_REGEX_EXPRESSION, values_representative_string)

        doc_configuration = self._config.document_configuration

        if raw_document_data.is_primary_key or raw_document_data.is_foreign_key:

            if doc_configuration.exclude_all_primary_and_foreign_keys:
                return True
            if (
                doc_configuration.exclude_only_numerical_primary_and_foreign_keys
                and BirdSupportedColumnDataFormats(raw_document_data.col_data_format) in NUMERICAL_DATA_FORMAT_ALIASES
            ):
                return True
            if (
                doc_configuration.exclude_primary_and_foreign_keys_with_uuid_values
                and _does_column_contain_uuid_like_values(raw_document_data.col_values)
            ):
                return True

        if doc_configuration.exclude_all_columns_with_id_in_the_name and "id" in raw_document_data.col_name.lower():
            return True

        return False

    def _get_representative_values(self, raw_document_data: SchemaLinkingDocumentRawData) -> str:

        def _get_most_frequent_categorical_values(list_of_categorical_values: list[str]) -> list[str]:
            number_of_most_common_values = min(DEFAULT_NUMBER_OF_MOST_COMMON_VALUES, len(list_of_categorical_values))
            most_common_values = Counter(list_of_categorical_values).most_common(number_of_most_common_values)
            return self._serialize_column_values_to_str([item for item, count in most_common_values])

        def _is_valid_data_format(data_format: str) -> bool:
            return data_format in BirdSupportedColumnDataFormats.get_values()

        def _get_filtered_column_values(values: list[Any]) -> list[Any]:
            return list(filter(lambda v: v is not None, values))

        representative_values = ""
        if not _is_valid_data_format(raw_document_data.col_data_format):
            return representative_values

        column_values = _get_filtered_column_values(raw_document_data.col_values)
        if not column_values:
            return representative_values

        mapped_col_data_format = BirdSupportedColumnDataFormats(raw_document_data.col_data_format)
        if mapped_col_data_format in NUMERICAL_DATA_FORMAT_ALIASES:
            return f"{min(column_values)}{STRING_SEPARATOR}{max(column_values)}"
        if mapped_col_data_format == BirdSupportedColumnDataFormats.TEXT:
            return _get_most_frequent_categorical_values(column_values)

        return representative_values

    def get_document_representation(self, raw_document_data: SchemaLinkingDocumentRawData) -> str | None:

        if self._column_should_be_excluded(raw_document_data):
            return None

        document_representation = (
            f"{self._normalize_text(raw_document_data.table_name)}"
            f"{STRING_SEPARATOR}"
            f"{self._normalize_text(raw_document_data.col_name)}"
        )
        if self._config.document_configuration.include_column_description:
            document_representation += f"{STRING_SEPARATOR}{self._normalize_text(raw_document_data.col_description)}"
        if self._config.document_configuration.include_column_data_format:
            document_representation += f"{STRING_SEPARATOR}{self._normalize_text(raw_document_data.col_data_format)}"
        if self._config.document_configuration.include_column_values:
            if self._config.document_configuration.include_only_values_representative:
                document_representation += f"{STRING_SEPARATOR}{self._get_representative_values(raw_document_data)}"
            else:
                document_representation += (
                    f"{STRING_SEPARATOR}{self._serialize_column_values_to_str(raw_document_data.col_values)}"
                )

        return document_representation

    def _process_documents(self) -> tuple[list[EmbeddingJobExample], list[SchemaLinkingDocumentRawData]]:
        raw_documents = read_jsonl(self._config.path_to_raw_documents, SchemaLinkingDocumentRawData)

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(
                    lambda raw_document_data: EmbeddingJobExample(
                        example_id=raw_document_data.document_id,
                        input_to_embed=self.get_document_representation(raw_document_data),
                    ),
                    raw_documents,
                )
            )

        processed_documents = [result for result in results if result.input_to_embed is not None]
        logger.info(f"Number of documents before processing {len(raw_documents)} - After: {len(processed_documents)}")
        return processed_documents, raw_documents

    def process(self) -> QueryAndDocumentProcessorOutput:
        with ThreadPoolExecutor() as executor:
            processed_queries = executor.submit(self._process_queries).result()
            processed_documents, raw_documents = executor.submit(self._process_documents).result()

        return QueryAndDocumentProcessorOutput(
            processed_queries=processed_queries, processed_documents=processed_documents, raw_documents=raw_documents
        )
