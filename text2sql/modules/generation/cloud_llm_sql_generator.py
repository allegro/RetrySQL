import re
from typing import Type

from allms.domain.response import ResponseData
from allms.models import AbstractModel
from pydantic import BaseModel

from text2sql.datasets.domain.model import Text2SqlDataSample
from text2sql.modules.generation.cloud_llm_generator import CloudLlmGenerator
from text2sql.modules.llm_input.domain.model import GeneratedSqlOutput
from text2sql.modules.llm_input.llm_input_creator import LlmInputCreator


class CloudLlmSqlGenerator(CloudLlmGenerator):

    def __init__(
        self,
        llm: AbstractModel,
        llm_input_creator: LlmInputCreator,
        output_model: Type[BaseModel] = GeneratedSqlOutput,
        system_prompt: str | None = None,
        extract_sql_from_error: bool = True,
    ) -> None:
        super().__init__(
            llm=llm,
            llm_input_creator=llm_input_creator,
            output_model=output_model,
            system_prompt=system_prompt,
        )
        self._extract_sql_from_error = extract_sql_from_error

    def __call__(self, data: list[Text2SqlDataSample]) -> list[ResponseData]:
        return self.forward(data=data)

    def forward(self, data: list[Text2SqlDataSample]) -> list[ResponseData]:
        responses = super().forward(data=data)

        if self._extract_sql_from_error:
            responses = [self.extract_sql_from_response_error(response) for response in responses]

        return responses

    @staticmethod
    def extract_sql_from_response_error(response: ResponseData) -> ResponseData:
        response_error = response.error

        if response_error and response_error != "None":
            # try to extract SQL query from error message (when JsonOutputParser error occurred)
            sql_match = re.search(r'\{\s*"sql":\s*"(.*?)"\s*}', response_error, re.DOTALL)

            if sql_match:
                sql_block = sql_match.group(1).replace("\n", " ").strip()
                response.response = {"sql": sql_block}

        return response
