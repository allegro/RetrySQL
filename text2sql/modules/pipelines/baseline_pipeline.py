from copy import deepcopy

from allms.models import AbstractModel

from text2sql.commons.db_utils.domain.model import DatabaseInfo
from text2sql.commons.db_utils.database_info import get_database_info
from text2sql.commons.db_utils.schema_prompt_strategy import SchemaPromptStrategy
from text2sql.datasets.domain.model import BirdDevDataSample
from text2sql.modules.pipelines.domain import PipelineResults
from text2sql.modules.generation.cloud_llm_generator import CloudLlmGenerator
from text2sql.modules.generation.cloud_llm_sql_generator import CloudLlmSqlGenerator
from text2sql.modules.llm_input.domain.model import GeneratedSqlOutput
from text2sql.modules.llm_input.llm_input_creator import LlmInputCreator
from text2sql.modules.schema_linking.domain.model import SchemaLinkingOutput
from text2sql.modules.llm_input.prompt_templates.sql_generation_prompt_templates import BaselineSqlPromptTemplate
from text2sql.commons.db_utils.schema_links_postprocessing import update_db_schema_with_predicted_schema_links


class BaselinePipeline:

    def __init__(
        self,
        llm: AbstractModel,
        schema_prompt_creator: SchemaPromptStrategy,
        use_cot: bool,
        use_knowledge: bool,
        pre_computed_schema_links: list[SchemaLinkingOutput],
        schema_linking_postprocess: bool = True,
        num_example_rows: int = 0,
        shuffle_schema_cols: bool = False,
        use_pydantic_output: bool = False,
    ) -> None:
        self._schema_prompt_creator = schema_prompt_creator
        self._num_example_rows = num_example_rows
        self._shuffle_schema_cols = shuffle_schema_cols

        self._pre_computed_schema_links = pre_computed_schema_links
        self._schema_linking_postprocess = schema_linking_postprocess

        llm_input_creator = LlmInputCreator(
            prompt_template=BaselineSqlPromptTemplate(),
            use_cot=use_cot,
            use_knowledge=use_knowledge,
        )

        if use_pydantic_output:
            self.generator = CloudLlmSqlGenerator(
                llm=llm,
                llm_input_creator=llm_input_creator,
                output_model=GeneratedSqlOutput,
                system_prompt=None,
                extract_sql_from_error=True,
            )
        else:
            self.generator = CloudLlmGenerator(
                llm=llm,
                llm_input_creator=llm_input_creator,
                output_model=None,
                system_prompt=None,
            )

    @property
    def schema_creator(self) -> SchemaPromptStrategy:
        return self._schema_prompt_creator

    @schema_creator.setter
    def schema_creator(self, strategy: SchemaPromptStrategy) -> None:
        self._schema_prompt_creator = strategy

    def __call__(
        self,
        data: list[BirdDevDataSample],
    ) -> PipelineResults:
        return self.forward(data=data)

    def _updated_schema_with_precomputed_schema_links(
            self,
            data: list[BirdDevDataSample],
            database_infos: dict[str, DatabaseInfo],
    ) -> list[BirdDevDataSample]:

        if not self._pre_computed_schema_links:
            return data

        assert len(data) == len(self._pre_computed_schema_links), "Data and schema link outputs must have the same length"

        data_with_updated_db_schema, _ = update_db_schema_with_predicted_schema_links(
            data=data,
            schema_linking_responses_raw=self._pre_computed_schema_links,
            schema_prompt_creator=self._schema_prompt_creator,
            database_infos=database_infos,
            schema_link_postprocessing=self._schema_linking_postprocess,
            shuffle_schema_columns=self._shuffle_schema_cols,
        )

        return data_with_updated_db_schema

    def _update_schema_in_data_samples(
        self,
        data: list[BirdDevDataSample],
        database_infos: dict[str, DatabaseInfo],
    ) -> list[BirdDevDataSample]:
        data_copy = deepcopy(data)

        for sample in data_copy:
            sample.database_schema = self._schema_prompt_creator.create_schema_prompt(
                db_info=database_infos[sample.database_name],
                schema_links=[],
                shuffle_cols=self._shuffle_schema_cols,
            )

        return data_copy

    def forward(
        self,
        data: list[BirdDevDataSample],
    ) -> PipelineResults:
        db_name_path_map = {sample.database_name: sample.database_path for sample in data}
        database_infos = {
            db_name: get_database_info(db_path=db_root_path, num_example_rows=self._num_example_rows)
            for db_name, db_root_path in db_name_path_map.items()
        }

        updated_data = self._update_schema_in_data_samples(
            data=data,
            database_infos=database_infos
        )

        data_after_linking = self._updated_schema_with_precomputed_schema_links(
            data=updated_data,
            database_infos=database_infos
        )

        predictions = self.generator(data=data_after_linking)

        return PipelineResults(sql_generation_responses=predictions)
