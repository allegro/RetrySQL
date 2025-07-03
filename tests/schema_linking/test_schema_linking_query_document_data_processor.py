import pytest

from tests.helpers.builders import (
    DocumentConfigurationBuilder,
    EmbeddingsJobConfigurationBuilder,
    QueryConfigurationBuilder,
)
from text2sql.datasets.domain.model import SchemaLinkingDocumentRawData, SchemaLinkingQueryRawData
from text2sql.modules.schema_linking.data_processors.query_and_document_processor import QueryAndDocumentProcessor


class TestQueryAndDocumentProcessor:

    @pytest.mark.parametrize(
        "include_external_knowledge, expected_representation",
        [
            (True, "what is the capital of france? paris is the capital of france."),
            (False, "what is the capital of france?"),
        ],
    )
    def test_get_query_representation(self, include_external_knowledge, expected_representation):
        # GIVEN
        raw_query_sample = SchemaLinkingQueryRawData(
            question_id=1,
            question="What is the capital of France?",
            external_knowledge="Paris is the capital of France.",
        )

        # AND
        query_config = (
            QueryConfigurationBuilder.default().with_include_external_knowledge(include_external_knowledge).build()
        )
        config = EmbeddingsJobConfigurationBuilder.default().with_query_configuration(query_config).build()
        processor = QueryAndDocumentProcessor(config)

        # WHEN
        result = processor.get_query_representation(raw_query_sample)

        assert result == expected_representation

    @pytest.mark.parametrize(
        "include_column_description, include_column_data_format, include_column_values, "
        "exclude_all_primary_and_foreign_keys, exclude_only_numerical_primary_and_foreign_keys, "
        "exclude_primary_and_foreign_keys_with_uuid_values, exclude_all_columns_with_id_in_the_name, "
        "input_raw_document, expected_representation",
        [
            (
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                SchemaLinkingDocumentRawData(
                    document_id=1,
                    table_name="countries",
                    col_name="capital",
                    is_primary_key=False,
                    is_foreign_key=True,
                    col_description="Name of the capital city",
                    col_data_format="text",
                    col_values=["Paris", "Berlin"],
                ),
                "countries capital name of the capital city text paris berlin",
            ),
            (
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                SchemaLinkingDocumentRawData(
                    document_id=4,
                    table_name="countries",
                    col_name="capital",
                    is_primary_key=False,
                    is_foreign_key=True,
                    col_description="Name of the capital city",
                    col_data_format="text",
                    col_values=["Paris", "Berlin"],
                ),
                "countries capital",
            ),
            (
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                SchemaLinkingDocumentRawData(
                    document_id=5,
                    table_name="countries",
                    col_name="capital",
                    is_primary_key=True,
                    is_foreign_key=False,
                    col_description="Name of the capital city",
                    col_data_format="text",
                    col_values=["Paris"],
                ),
                None,
            ),
            (
                True,
                True,
                True,
                False,
                True,
                False,
                False,
                SchemaLinkingDocumentRawData(
                    document_id=6,
                    table_name="countries",
                    col_name="capital",
                    is_primary_key=False,
                    is_foreign_key=True,
                    col_description="Name of the capital city",
                    col_data_format="integer",
                    col_values=[1, 2, 3],
                ),
                None,
            ),
            (
                True,
                True,
                True,
                False,
                False,
                True,
                False,
                SchemaLinkingDocumentRawData(
                    document_id=7,
                    table_name="countries",
                    col_name="capital",
                    is_primary_key=True,
                    is_foreign_key=False,
                    col_description="Name of the capital city",
                    col_data_format="text",
                    col_values=["123e4567-e89b-12d3-a456-426614174000", "456e4567-1748-uu78-a456-426614174000"],
                ),
                None,
            ),
            (
                True,
                True,
                True,
                False,
                False,
                False,
                True,
                SchemaLinkingDocumentRawData(
                    document_id=8,
                    table_name="countries",
                    col_name="capital_id",
                    is_primary_key=False,
                    is_foreign_key=False,
                    col_description="Name of the capital city",
                    col_data_format="text",
                    col_values=["Paris"],
                ),
                None,
            ),
        ],
    )
    def test_get_document_representation(
        self,
        include_column_description,
        include_column_data_format,
        include_column_values,
        exclude_all_primary_and_foreign_keys,
        exclude_only_numerical_primary_and_foreign_keys,
        exclude_primary_and_foreign_keys_with_uuid_values,
        exclude_all_columns_with_id_in_the_name,
        input_raw_document,
        expected_representation,
    ) -> None:

        # GIVEN
        doc_config = (
            DocumentConfigurationBuilder.default()
            .with_column_description(include_column_description)
            .with_column_data_format(include_column_data_format)
            .with_column_values(include_column_values)
            .with_exclude_all_primary_and_foreign_keys(exclude_all_primary_and_foreign_keys)
            .with_exclude_only_numerical_primary_and_foreign_keys(exclude_only_numerical_primary_and_foreign_keys)
            .with_exclude_primary_and_foreign_keys_with_uuid_values(exclude_primary_and_foreign_keys_with_uuid_values)
            .with_exclude_columns_with_id_in_name(exclude_all_columns_with_id_in_the_name)
            .build()
        )
        config = EmbeddingsJobConfigurationBuilder.default().with_document_configuration(doc_config).build()
        processor = QueryAndDocumentProcessor(config)

        # WHEN
        result = processor.get_document_representation(input_raw_document)

        # THEN
        assert result == expected_representation

    @pytest.mark.parametrize(
        "input_raw_document, expected_representation",
        [
            (
                SchemaLinkingDocumentRawData(
                    document_id=1,
                    table_name="countries",
                    col_name="capital",
                    is_primary_key=False,
                    is_foreign_key=True,
                    col_description="Name of the capital city",
                    col_data_format="text",
                    col_values=["Berlin", "Berlin", "Berlin", "Paris", "Rome", "Paris", "Berlin"],
                ),
                "countries capital name of the capital city text berlin paris rome",
            ),
            (
                SchemaLinkingDocumentRawData(
                    document_id=1,
                    table_name="countries",
                    col_name="capital",
                    is_primary_key=False,
                    is_foreign_key=True,
                    col_description="Name of the capital city",
                    col_data_format="integer",
                    col_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                ),
                "countries capital name of the capital city integer 1 10",
            ),
            (
                SchemaLinkingDocumentRawData(
                    document_id=1,
                    table_name="countries",
                    col_name="capital",
                    is_primary_key=False,
                    is_foreign_key=True,
                    col_description="Name of the capital city",
                    col_data_format="real",
                    col_values=[0.01, 100, 1000, 4.05, 100939.76],
                ),
                "countries capital name of the capital city real 0.01 100939.76",
            ),
            (
                SchemaLinkingDocumentRawData(
                    document_id=1,
                    table_name="countries",
                    col_name="capital",
                    is_primary_key=False,
                    is_foreign_key=True,
                    col_description="Name of the capital city",
                    col_data_format="date",
                    col_values=["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"],
                ),
                "countries capital name of the capital city date 2021-01-01 2021-01-05",
            ),
            (
                SchemaLinkingDocumentRawData(
                    document_id=1,
                    table_name="countries",
                    col_name="capital",
                    is_primary_key=False,
                    is_foreign_key=True,
                    col_description="Name of the capital city",
                    col_data_format="datetime",
                    col_values=[
                        "2018-06-12 09:55:22",
                        "2018-06-12 09:55:23",
                        "2018-06-12 09:55:24",
                        "2018-06-12 09:55:25",
                        "2018-06-12 09:55:26",
                    ],
                ),
                "countries capital name of the capital city datetime 2018-06-12 09:55:22 2018-06-12 09:55:26",
            ),
            (
                SchemaLinkingDocumentRawData(
                    document_id=1,
                    table_name="countries",
                    col_name="capital",
                    is_primary_key=False,
                    is_foreign_key=True,
                    col_description="Name of the capital city",
                    col_data_format="unknown",
                    col_values=[None, "something", 124],
                ),
                "countries capital name of the capital city unknown ",
            ),
        ],
    )
    def test_values_representative_for_document_representation(
        self,
        input_raw_document,
        expected_representation,
    ) -> None:

        # GIVEN
        doc_config = DocumentConfigurationBuilder.default().with_only_column_values_representatives(True).build()
        config = EmbeddingsJobConfigurationBuilder.default().with_document_configuration(doc_config).build()
        processor = QueryAndDocumentProcessor(config)

        # WHEN
        result = processor.get_document_representation(input_raw_document)

        # THEN
        assert result == expected_representation
