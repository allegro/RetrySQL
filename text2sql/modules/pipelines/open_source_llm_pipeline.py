from copy import deepcopy

from text2sql.commons.db_utils.database_info import get_database_info
from text2sql.commons.db_utils.domain.model import DatabaseInfo
from text2sql.commons.db_utils.schema_links_postprocessing import update_db_schema_with_predicted_schema_links
from text2sql.commons.db_utils.schema_prompt_strategy import SchemaPromptStrategy
from text2sql.datasets.domain.model import BirdDevDataSample
from text2sql.llms.configuration import OpenSourceLlmConfiguration
from text2sql.llms.domain.models import OpenSourceLlm
from text2sql.modules.generation.open_source_llm_sql_generator import OpenSourceLlmSqlGenerator
from text2sql.modules.llm_input.prompt_templates.sql_generation_prompt_templates import BaselineSqlPromptTemplate
from text2sql.modules.pipelines.domain import PipelineResults
from text2sql.modules.schema_linking.domain.model import SchemaLinkingOutput
from text2sql.modules.schema_linking.schema_linkers.schema_linker_base import SchemaLinkerBase


class OpenSourceLlmPipeline:

    def __init__(
        self,
        open_source_llm: OpenSourceLlm,
        open_source_llm_config: OpenSourceLlmConfiguration,
        schema_prompt_creator: SchemaPromptStrategy,
        sql_generation_prompt_template: BaselineSqlPromptTemplate,
        pre_computed_schema_links: list[SchemaLinkingOutput],
        schema_linker: SchemaLinkerBase | None = None,
        shuffle_schema_cols: bool = False,
        schema_linking_postprocess: bool = True,
        num_example_rows: int = 0
    ) -> None:
        self._schema_prompt_creator = schema_prompt_creator
        self._schema_linker = schema_linker
        self._pre_computed_schema_links = pre_computed_schema_links
        self._shuffle_schema_cols = shuffle_schema_cols
        self._schema_linking_postprocess = schema_linking_postprocess
        self._num_example_rows = num_example_rows
        self.generator = OpenSourceLlmSqlGenerator(
            llm=open_source_llm,
            llm_config=open_source_llm_config,
            sql_generation_prompt_template=sql_generation_prompt_template
        )

    @property
    def schema_creator(self) -> SchemaPromptStrategy:
        return self._schema_prompt_creator

    @schema_creator.setter
    def schema_creator(self, strategy: SchemaPromptStrategy) -> None:
        self._schema_prompt_creator = strategy

    @property
    def schema_linker(self) -> SchemaLinkerBase:
        return self._schema_linker

    @schema_linker.setter
    def schema_linker(self, schema_linker: SchemaLinkerBase) -> None:
        self._schema_linker = schema_linker

    def __call__(
        self,
        data: list[BirdDevDataSample],
    ) -> PipelineResults:
        return self.forward(data=data)

    def schema_linking_step(
            self,
            data: list[BirdDevDataSample],
            database_infos: dict[str, DatabaseInfo],
    ) -> tuple[list[BirdDevDataSample], list[SchemaLinkingOutput]]:

        if not (self._schema_linker or self._pre_computed_schema_links):
            return data, []

        schema_linking_responses_raw = (
            self._schema_linker(data=data)
            if self._schema_linker
            else self._pre_computed_schema_links
        )

        assert len(data) == len(schema_linking_responses_raw), "Data and schema link outputs must have the same length"
        
        data_with_updated_db_schema, schema_linking_responses = update_db_schema_with_predicted_schema_links(
            data=data,
            schema_linking_responses_raw=schema_linking_responses_raw,
            schema_prompt_creator=self._schema_prompt_creator,
            database_infos=database_infos,
            schema_link_postprocessing=self._schema_linking_postprocess,
            shuffle_schema_columns=self._shuffle_schema_cols,
        )

        return data_with_updated_db_schema, schema_linking_responses

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
        data_after_linking, schema_linking_responses = self.schema_linking_step(
            data=updated_data, 
            database_infos=database_infos
        )

        generation_responses = self.generator(data=data_after_linking)

        return PipelineResults(
            schema_linking_responses=schema_linking_responses,
            sql_generation_responses=generation_responses
        )
