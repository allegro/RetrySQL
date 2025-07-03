from string import Formatter
from typing import Type

from allms.domain.input_data import InputData
from pydantic import BaseModel

from text2sql.datasets.domain.model import Text2SqlDataSample
from text2sql.modules.llm_input.domain.model import SingleLlmInput
from text2sql.modules.llm_input.prompt_templates.prompt_template import PromptTemplate


class LlmInputCreator:
    def __init__(
        self,
        prompt_template: PromptTemplate,
        use_cot: bool,
        use_knowledge: bool,
    ) -> None:
        self.prompt_template = prompt_template
        self.use_cot = use_cot
        self.use_knowledge = use_knowledge

    def create(
        self,
        data: Text2SqlDataSample,
        output_model: Type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> SingleLlmInput:
        knowledge = data.knowledge if self.use_knowledge else None

        prompt_template = self.prompt_template.create(
            use_cot=self.use_cot,
            use_knowledge=self.use_knowledge,
            knowledge=knowledge,
        )

        prompt_template_keys = [prompt_key for _, prompt_key, _, _ in Formatter().parse(prompt_template) if prompt_key]
        input_data = self._prepare_input_data(data=data, keys=prompt_template_keys)

        return SingleLlmInput(
            input_data=input_data,
            prompt_template=prompt_template,
            output_model=output_model,
            system_prompt=system_prompt,
        )

    @staticmethod
    def _prepare_input_data(data: Text2SqlDataSample, keys: list[str]) -> InputData:
        data_dict = data.dict()
        input_mappings = {key: data_dict[key] for key in keys}

        return InputData(
            id=str(data.question_id),
            input_mappings=input_mappings,
        )
