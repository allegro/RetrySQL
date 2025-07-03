from text2sql.commons.db_utils.create_table_schema_prompt import CreateTableSchemaPrompt
from text2sql.commons.db_utils.database_info import get_database_info
from text2sql.commons.io_utils import translate_gcs_dir_to_local
from text2sql.datasets.bird_datasets import BirdDataset
from text2sql.datasets.domain.model import Text2SqlDataSample
from text2sql.modules.schema_linking.configuration import LlmBasedAlgorithmConfiguration
from text2sql.modules.schema_linking.data_processors.base_processor import BaseSchemaLinkingDataProcessor


class LlmBasedMethodDataProcessor(BaseSchemaLinkingDataProcessor):
    def __init__(self, config: LlmBasedAlgorithmConfiguration) -> None:
        self._config = config

        self._schema_prompt_creator = CreateTableSchemaPrompt()
        self._shuffle_schema_cols = False
        self._num_example_rows = 0

    def process(self) -> list[Text2SqlDataSample]:
        translated_db_root_path = translate_gcs_dir_to_local(self._config.db_root_dirpath)

        bird_dev_data = BirdDataset().load_dev(
            samples_path=self._config.path_to_bird_dev,
            db_root_dirpath=translated_db_root_path,
        )

        db_name_path_map = {sample.database_name: sample.database_path for sample in bird_dev_data}
        database_infos = {
            db_name: get_database_info(db_path=db_root_path, num_example_rows=self._num_example_rows)
            for db_name, db_root_path in db_name_path_map.items()
        }

        for sample in bird_dev_data:
            sample.database_schema = self._schema_prompt_creator.create_schema_prompt(
                db_info=database_infos[sample.database_name],
                schema_links=[],
                shuffle_cols=self._shuffle_schema_cols,
            )

        return bird_dev_data
