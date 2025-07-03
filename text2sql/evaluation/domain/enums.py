from text2sql.commons.list_enum import ListConvertableEnum


class SupportedBenchmarks(str, ListConvertableEnum):
    BIRD = "BIRD"
    SPIDER_V2 = "SPIDER_2"


class QuestionDifficulty(str, ListConvertableEnum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"