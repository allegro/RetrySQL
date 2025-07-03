from collections import defaultdict

from allms.models import AbstractModel

from text2sql.datasets.domain.model import SchemaLink, Text2SqlDataSample
from text2sql.modules.generation.cloud_llm_generator import CloudLlmGenerator
from text2sql.modules.llm_input.domain.model import TableAndColumnSchemaLinkingOutput
from text2sql.modules.llm_input.llm_input_creator import LlmInputCreator
from text2sql.modules.llm_input.prompt_templates.schema_linking_prompt_templates import (
    TableAndColumnSchemaLinkingPromptTemplate,
)
from text2sql.modules.schema_linking.domain.model import SchemaLinkingOutput
from text2sql.modules.schema_linking.schema_linkers.schema_linker_base import SchemaLinkerBase
from text2sql.commons.db_utils.schema_links_postprocessing import postprocess_schema_links
from text2sql.commons.db_utils.database_info import get_database_info


class LlmSchemaLinkerTableAndColumn(SchemaLinkerBase):

    def __init__(
        self,
        llm: AbstractModel,
        use_cot: bool,
        use_knowledge: bool,
    ) -> None:
        super().__init__(use_external_knowledge=use_knowledge)

        llm_input_creator = LlmInputCreator(
            prompt_template=TableAndColumnSchemaLinkingPromptTemplate(),
            use_cot=use_cot,
            use_knowledge=self.use_external_knowledge,
        )

        self.generator = CloudLlmGenerator(
            llm=llm,
            llm_input_creator=llm_input_creator,
            output_model=TableAndColumnSchemaLinkingOutput,
            system_prompt=None,
        )

    def __call__(self, data: list[Text2SqlDataSample]) -> list[SchemaLinkingOutput]:
        return self.forward(data=data)

    def forward(self, data: list[Text2SqlDataSample]) -> list[SchemaLinkingOutput]:
        assert all(
            sample.database_schema is not None for sample in data
        ), "Database schema for sample is required for schema linking"

        responses = self.generator(data=data)

        db_infos = [get_database_info(sample.database_path) for sample in data]

        schema_linking_outputs = []
        for response, db_info in zip(responses, db_infos):
            table_to_columns = defaultdict(set)

            if response.response and response.response.schema_links:
                for table_name, column_name in response.response.schema_links:
                    table_to_columns[table_name].add(column_name)

            schema_links = [
                SchemaLink(table_name=table, columns=list(columns)) for table, columns in table_to_columns.items()
            ]

            postprocessed_schema_link = postprocess_schema_links(db_info=db_info, schema_links=schema_links)

            schema_linking_outputs.append(
                SchemaLinkingOutput(
                    question_id=int(response.input_data.id),
                    question=response.input_data.input_mappings["question"],
                    schema_links=postprocessed_schema_link if postprocessed_schema_link else [],
                )
            )

        return schema_linking_outputs
