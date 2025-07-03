from collections import defaultdict
from typing import Type

import pydash
from allms.domain.response import ResponseData
from allms.models import AbstractModel
from pydantic import BaseModel

from text2sql.modules.generation.llm_base_generator import LlmBaseGenerator
from text2sql.datasets.domain.model import Text2SqlDataSample
from text2sql.modules.llm_input.domain.model import BatchLlmInput, SingleLlmInput
from text2sql.modules.llm_input.llm_input_creator import LlmInputCreator


class CloudLlmGenerator(LlmBaseGenerator):

    def __init__(
        self,
        llm: AbstractModel,
        llm_input_creator: LlmInputCreator,
        output_model: Type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(
            llm=llm,
            llm_input_creator=llm_input_creator,
            output_model=output_model,
            system_prompt=system_prompt
        )

    def __call__(self, data: list[Text2SqlDataSample]) -> list[ResponseData]:
        return self.forward(data=data)

    def forward(self, data: list[Text2SqlDataSample]) -> list[ResponseData]:
        llm_inputs = [
            self._llm_input_creator.create(
                data=sample,
                output_model=self._output_model,
                system_prompt=self._system_prompt,
            )
            for sample in data
        ]
        batched_llm_inputs = self._batch_queries(single_inputs=llm_inputs)

        return (
            pydash.chain(batched_llm_inputs)
            .map(lambda batch: self._generate_responses(data=batch))
            .flatten()
            .sort_by(lambda x: int(x.input_data.id))
            .value()
        )

    def _generate_responses(self, data: BatchLlmInput) -> list[ResponseData]:
        return self._llm.generate(
            prompt=data.prompt_template,
            input_data=data.input_data,
            output_data_model_class=data.output_model,
            system_prompt=data.system_prompt,
        )

    @staticmethod
    def _batch_queries(single_inputs: list[SingleLlmInput]) -> list[BatchLlmInput]:
        batched_input_data = defaultdict(list)
        for single_input in single_inputs:
            batched_input_data[(single_input.prompt_template, single_input.output_model)].append(
                single_input.input_data
            )

        batches = [
            BatchLlmInput(
                input_data=input_data,
                prompt_template=prompt_template,
                output_model=output_model_cls,
            )
            for (prompt_template, output_model_cls), input_data in batched_input_data.items()
        ]

        return batches
