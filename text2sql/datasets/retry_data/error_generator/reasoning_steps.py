import random

from text2sql.datasets.domain.enums import NumCorruptionsPerStepType, StepsToUseForCorruptionsType
from text2sql.datasets.domain.model import CorruptedSample, DatabaseSchema, \
    BirdTrainWithReasoningStepsDataSample
from text2sql.datasets.retry_data.error_generator.abstract import AbstractErrorGenerator


class ReasoningStepsErrorGenerator(AbstractErrorGenerator):
    def __init__(
            self,
            num_corruptions_per_step: NumCorruptionsPerStepType,
            steps_to_use_for_corruptions: StepsToUseForCorruptionsType
    ) -> None:
        self._num_corruptions_per_step = num_corruptions_per_step
        self._steps_to_use_for_corruptions = steps_to_use_for_corruptions

    def corrupt(
            self,
            data_sample: BirdTrainWithReasoningStepsDataSample,
            probability: float,
            bird_metadata: list[DatabaseSchema]
    ) -> CorruptedSample:
        corrupted_reasoning_steps = []
        for idx, reasoning_step in enumerate(data_sample.reasoning_steps):
            corruptions = self._corrupt_single_reasoning_step(
                reasoning_steps=data_sample.reasoning_steps,
                current_step_idx=idx,
                probability=probability
            )
            corrupted_reasoning_steps.extend(corruptions)
            corrupted_reasoning_steps.append(reasoning_step)

        return CorruptedSample(
            question_id=data_sample.question_id,
            database_name=data_sample.database_name,
            question=data_sample.question,
            knowledge=data_sample.knowledge,
            sql_query=data_sample.sql_query,
            reasoning_steps=corrupted_reasoning_steps,
        )

    def _get_steps_to_use_for_corruption(self, reasoning_steps: list[str], current_step_idx: int) -> list[str]:
        match self._steps_to_use_for_corruptions:
            case StepsToUseForCorruptionsType.FROM_FUTURE:
                return reasoning_steps[current_step_idx + 1:]
            case StepsToUseForCorruptionsType.FROM_PAST_AND_FUTURE:
                return reasoning_steps[:current_step_idx] + reasoning_steps[current_step_idx + 1:]
            case _:
                raise ValueError(f"Unknown steps to use for corruptions value: {self._steps_to_use_for_corruptions}")

    def _get_single_corruption(self, reasoning_steps: list[str], current_step_idx: int) -> str | None:
        steps_to_use_for_corruption = self._get_steps_to_use_for_corruption(reasoning_steps, current_step_idx)
        if not steps_to_use_for_corruption:
            return None

        corruption = random.choice(steps_to_use_for_corruption)

        return f"{corruption}[BACK]"

    def _corrupt_single_reasoning_step(
            self,
            reasoning_steps: list[str],
            current_step_idx: int,
            probability: float
    ) -> list[str]:
        corruptions = []
        while random.random() < probability:
            corruptions.append(
                self._get_single_corruption(reasoning_steps, current_step_idx)
            )
            if self._num_corruptions_per_step == NumCorruptionsPerStepType.SINGLE:
                break

        return list(filter(None, corruptions))