from dataclasses import dataclass

from text2sql.evaluation.domain.enums import QuestionDifficulty


@dataclass
class GroundTruthTestCase:
    database_name: str
    sql_query: str
    question_difficulty: QuestionDifficulty
