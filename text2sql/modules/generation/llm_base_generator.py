from typing import Type

from allms.domain.response import ResponseData
from allms.models import AbstractModel
from pydantic import BaseModel

from text2sql.datasets.domain.model import Text2SqlDataSample
from text2sql.modules.llm_input.llm_input_creator import LlmInputCreator
from text2sql.llms.domain.models import OpenSourceLlm


class LlmBaseGenerator:

    def __init__(
        self,
        llm: AbstractModel | OpenSourceLlm,
        llm_input_creator: LlmInputCreator | None = None,
        output_model: Type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._llm = llm
        self._llm_input_creator = llm_input_creator
        self._output_model = output_model
        self._system_prompt = system_prompt

    def __call__(self, data: list[Text2SqlDataSample]) -> list[ResponseData]:
        return self.forward(data=data)

    def forward(self, data: list[Text2SqlDataSample]) -> list[ResponseData]:
        pass
