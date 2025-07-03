from abc import ABC, abstractmethod

from pydantic import BaseModel

from text2sql.datasets.domain.model import BirdTrainDataSample, BirdDevDataSample, BirdTrainWithReasoningStepsDataSample


class BaseBirdEntryDto(BaseModel, ABC):
    class Config:
        extra = "forbid"

    db_id: str
    question: str
    evidence: str
    SQL: str

    @abstractmethod
    def to_domain(self, question_id: int | None = None):
        ...


class TrainBirdEntryDto(BaseBirdEntryDto):
    def to_domain(self, question_id: int | None = None) -> BirdTrainDataSample:
        if question_id is None:
            raise ValueError(
                "Because raw train BIRD dataset doesn't have `question_id` it needs to be provided when calling this"
                " function"
            )
        return BirdTrainDataSample(
            question_id=question_id,
            question=self.question,
            database_name=self.db_id,
            sql_query=self.SQL,
            knowledge=self.evidence,
        )


class DevBirdEntryDto(BaseBirdEntryDto):
    question_id: int
    difficulty: str

    def to_domain(self, question_id: int | None = None) -> BirdDevDataSample:
        return BirdDevDataSample(
            question_id=self.question_id,
            question=self.question,
            database_name=self.db_id,
            sql_query=self.SQL,
            knowledge=self.evidence,
            difficulty=self.difficulty,
        )


class TrainBirdEntryWithReasoningStepsDto(TrainBirdEntryDto):
    reasoning_steps: list[str]

    def to_domain(self, *args, **kwargs) -> BirdTrainWithReasoningStepsDataSample:
        bird_train_sample = super().to_domain(*args, **kwargs)

        return BirdTrainWithReasoningStepsDataSample(
            **bird_train_sample.model_dump(),
            reasoning_steps=self.reasoning_steps
        )


class BirdDatasetDto(BaseModel):
    entries: list[TrainBirdEntryDto] | list[DevBirdEntryDto] | list[TrainBirdEntryWithReasoningStepsDto]

    def to_domain(self):
        return [
            entry.to_domain(question_id=idx) for idx, entry in enumerate(self.entries)
        ]