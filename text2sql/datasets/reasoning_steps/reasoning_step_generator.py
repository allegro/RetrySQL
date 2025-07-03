import json
import logging
import re

from allms.domain.input_data import InputData
from allms.domain.response import ResponseData
from allms.models import AbstractModel
from upath import UPath

from text2sql.commons.io_utils import save_objects_to_jsonl_file
from text2sql.datasets.domain.model import BirdTrainDataSample
from text2sql.datasets.dto.bird_dataset import TrainBirdEntryWithReasoningStepsDto
from text2sql.datasets.reasoning_steps.prompt import REASONING_PROMPT

logger = logging.getLogger(__name__)


class ReasoningStepGenerator:
    def __init__(
            self,
            bird_dataset: list[BirdTrainDataSample],
            model: AbstractModel,
            output_dir: UPath
    ):
        self._bird_dataset = bird_dataset
        self._model = model
        self._output_dir = output_dir

    def __call__(self):
        return self.generate_reasoning_steps()

    def generate_reasoning_steps(self):
        input_data = self._prepare_input_data()

        logger.info("Generating reasoning steps")
        responses = self._model.generate(REASONING_PROMPT, input_data=input_data)
        self._save_raw_predictions(responses)

        bird_with_reasoning = self._map_responses_to_bird_dto(responses)
        self._save_bird_with_reasoning_steps(bird_with_reasoning)

    def _prepare_input_data(self):
        return [
            InputData(input_mappings={"sql_query": example.sql_query}, id=str(example.question_id))
            for example in self._bird_dataset
        ]

    def _map_responses_to_bird_dto(self, responses: list[ResponseData]) -> list[TrainBirdEntryWithReasoningStepsDto]:
        logger.info("Parsing responses to BIRD DTO class")
        ordered_responses = sorted(responses, key=lambda response: int(response.input_data.id))

        return [
            TrainBirdEntryWithReasoningStepsDto(
                db_id=bird_entry.database_name,
                question=bird_entry.question,
                evidence=bird_entry.knowledge,
                SQL=bird_entry.sql_query,
                reasoning_steps=self._postprocess_generated_reasoning_steps(response.response)
            )
            for response, bird_entry in zip(ordered_responses, self._bird_dataset)
        ]

    @staticmethod
    def _postprocess_generated_reasoning_steps(response: str | None) -> list[str]:
        return (
            re.sub(r"\n+", "\n", response).split("\n")
            if response is not None else []
        )

    def _save_bird_with_reasoning_steps(
            self,
            bird_with_reasoning: list[TrainBirdEntryWithReasoningStepsDto]
    ) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir.joinpath("train_with_reasoning_steps.json")

        logger.info(f"Saving BIRD with reasoning steps to {output_path}")

        bird_with_reasoning_json = json.dumps([entry.model_dump() for entry in bird_with_reasoning], indent=4)
        with output_path.open("w") as file:
            file.write(bird_with_reasoning_json)

    def _save_raw_predictions(self, raw_predictions: list[ResponseData]) -> None:
        output_path = self._output_dir.joinpath("raw_reasoning_steps_predictions.jsonl")

        logger.info(f"Saving raw predictions to {output_path}")

        save_objects_to_jsonl_file(
            [raw_prediction.model_dump() for raw_prediction in raw_predictions],
            output_path
        )
