from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from text2sql.evaluation.domain.enums import QuestionDifficulty


class GroundTruth(BaseModel):
    question_id: int
    question_difficulty: QuestionDifficulty
    database_name: str
    sql_query: str


class GenerationResult(BaseModel):
    question_id: int
    sql_query: str


class ResultPair(BaseModel):
    question_id: int
    question_difficulty: QuestionDifficulty
    database_name: str
    ground_truth_query: str
    generated_query: str


@dataclass
class EvaluationResult:
    easy_accuracy: float
    medium_accuracy: float
    hard_accuracy: float
    overall_accuracy: float


class QueryExecutionResult(BaseModel):
    result_set: Any
    execution_time: float
