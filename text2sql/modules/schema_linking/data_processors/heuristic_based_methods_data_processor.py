from text2sql.datasets.bird_datasets import BirdDataset
from text2sql.datasets.domain.model import DatabaseSchema, Text2SqlDataSample
from text2sql.datasets.schema_linking.parsers.bird_dataset_parsing_utils import read_bird_database_schemas
from text2sql.modules.schema_linking.configuration import HeuristicBasedAlgorithmConfiguration
from text2sql.modules.schema_linking.data_processors.base_processor import BaseSchemaLinkingDataProcessor
from text2sql.modules.schema_linking.domain.model import SchemaLinkerHeuristicBasedAlgorithmInputExample


class HeuristicBasedMethodsDataProcessor(BaseSchemaLinkingDataProcessor):
    def __init__(self, config: HeuristicBasedAlgorithmConfiguration) -> None:
        self._config = config

        self._bird_dataset = BirdDataset()
        self._bird_database_schemas = read_bird_database_schemas(config.bird_metadata_path)

    def _find_matching_metadata(self, bird_example: Text2SqlDataSample) -> DatabaseSchema | None:
        matching_metadata = None
        for db_meta in self._bird_database_schemas:
            if db_meta.database_name == bird_example.database_name:
                matching_metadata = DatabaseSchema(database_name=db_meta.database_name, tables=db_meta.tables)
                break
        return matching_metadata

    def process(self) -> list[SchemaLinkerHeuristicBasedAlgorithmInputExample]:
        input_for_schema_linking = []
        for bird_example in self._bird_dataset.load_dev():
            matching_metadata = self._find_matching_metadata(bird_example)
            if matching_metadata:
                input_for_schema_linking.append(
                    SchemaLinkerHeuristicBasedAlgorithmInputExample(
                        question_id=bird_example.question_id,
                        question=bird_example.question,
                        external_knowledge=bird_example.knowledge,
                        database_schema=matching_metadata,
                    )
                )
        return input_for_schema_linking
