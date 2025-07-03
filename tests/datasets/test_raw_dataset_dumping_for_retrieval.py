import json
import shutil

import pytest

from text2sql.commons.io_utils import read_jsonl
from text2sql.datasets.domain.model import SchemaColumn, SchemaLinkingDocumentRawData, SchemaLinkingQueryRawData
from text2sql.datasets.schema_linking.parsers.bird_dataset_parsing_utils import read_bird_database_schemas
from text2sql.datasets.schema_linking.parsers.bird_parser_for_retrieval import SchemaLinkingDataDumperForRetrieval
from text2sql.settings import TESTS_DIR

BIRD_DEV_DATASET_EXTRACT_ROOT_DIR = TESTS_DIR.joinpath("resources/bird_dev_dataset_extract")
TEST_OUTPUT_DIR = TESTS_DIR.joinpath("resources/output_dir")


class TestRawDatasetDumpingForRetrieval:

    def teardown_method(self):
        if TEST_OUTPUT_DIR.exists():
            shutil.rmtree(TEST_OUTPUT_DIR.path)

    @pytest.fixture
    def sample_bird_metadata_file(self):
        sample_bird_metadata = [
            {
                "db_id": "california_schools",
                "table_names_original": ["frpm", "satscores", "schools"],
                "column_names_original": [
                    [-1, "*"],
                    [0, "CDSCode"],
                    [1, "cds"],
                    [2, "NCESDist"],
                    [2, "NCESSchool"],
                ],
                "primary_keys": [1, 3],
                "foreign_keys": [[1, 3]],
            }
        ]

        if not TEST_OUTPUT_DIR.exists():
            TEST_OUTPUT_DIR.mkdir()

        metadata_path = TEST_OUTPUT_DIR.joinpath("sample_metadata.json")
        with metadata_path.open("w") as json_file:
            json.dump(sample_bird_metadata, json_file, indent=4)
        return metadata_path

    def test_end_to_end(self) -> None:
        # GIVEN
        bird_ground_truth_path = BIRD_DEV_DATASET_EXTRACT_ROOT_DIR.joinpath("dev.json")
        bird_metadata_path = BIRD_DEV_DATASET_EXTRACT_ROOT_DIR.joinpath("dev_tables.json")
        database_root_path = BIRD_DEV_DATASET_EXTRACT_ROOT_DIR.joinpath("dev_databases")
        output_dir_path = TEST_OUTPUT_DIR

        # AND
        data_dumper = SchemaLinkingDataDumperForRetrieval(
            bird_ground_truth_path=bird_ground_truth_path,
            bird_metadata_path=bird_metadata_path,
            database_root_path=database_root_path,
            output_dir_path=output_dir_path,
        )

        # WHEN
        data_dumper.dump()
        dumped_queries = read_jsonl(output_dir_path.joinpath("queries.jsonl"), class_schema=SchemaLinkingQueryRawData)
        dumped_documents = read_jsonl(
            output_dir_path.joinpath("documents.jsonl"), class_schema=SchemaLinkingDocumentRawData
        )

        # THEN

        assert len(dumped_queries) == 3
        assert len(dumped_documents) == 89
        assert [isinstance(dumped_query, SchemaLinkingQueryRawData) for dumped_query in dumped_queries]
        assert [isinstance(dumped_document, SchemaLinkingDocumentRawData) for dumped_document in dumped_documents]

    def test_read_bird_metadata_database_name(self, sample_bird_metadata_file):
        # WHEN
        bird_metadata = read_bird_database_schemas(sample_bird_metadata_file)
        # THEN
        assert bird_metadata[0].database_name == "california_schools"

    def test_read_bird_metadata_tables_reading(self, sample_bird_metadata_file):
        # WHEN
        bird_metadata = read_bird_database_schemas(sample_bird_metadata_file)

        # THEN
        assert len(bird_metadata[0].tables) == 3
        assert "frpm" in bird_metadata[0].tables
        assert "satscores" in bird_metadata[0].tables
        assert "schools" in bird_metadata[0].tables

    def test_whether_bird_metadata_reading_does_lower_case(self, sample_bird_metadata_file):
        # GIVEN
        bird_metadata = read_bird_database_schemas(sample_bird_metadata_file)

        # WHEN
        first_table_columns = bird_metadata[0].tables["frpm"]
        second_table_columns = bird_metadata[0].tables["satscores"]
        third_table_columns = bird_metadata[0].tables["schools"]

        # AND
        first_table_column_names = [col.column_name for col in first_table_columns]
        second_table_columns = [col.column_name for col in second_table_columns]
        third_table_columns = [col.column_name for col in third_table_columns]

        # THEN
        assert "cdscode" in first_table_column_names
        assert "cds" in second_table_columns
        assert "ncesdist" in third_table_columns
        assert "ncesschool" in third_table_columns

    def test_primary_and_foreign_key_identification(self, sample_bird_metadata_file):
        # GIVEN
        bird_metadata = read_bird_database_schemas(sample_bird_metadata_file)

        # WHEN
        first_table_columns = bird_metadata[0].tables["frpm"]
        second_table_columns = bird_metadata[0].tables["satscores"]
        third_table_columns = bird_metadata[0].tables["schools"]

        # THEN
        assert first_table_columns == [SchemaColumn(column_name="cdscode", is_primary_key=True, is_foreign_key=True)]
        assert second_table_columns == [SchemaColumn(column_name="cds", is_primary_key=False, is_foreign_key=False)]
        assert third_table_columns == [
            SchemaColumn(column_name="ncesdist", is_primary_key=True, is_foreign_key=True),
            SchemaColumn(column_name="ncesschool", is_primary_key=False, is_foreign_key=False),
        ]
