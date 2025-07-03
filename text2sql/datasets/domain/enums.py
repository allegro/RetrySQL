from enum import StrEnum

from text2sql.commons.list_enum import ListConvertableEnum

class NumCorruptionsPerStepType(StrEnum, ListConvertableEnum):
    SINGLE = "SINGLE"
    MULTIPLE = "MULTIPLE"


class StepsToUseForCorruptionsType(StrEnum, ListConvertableEnum):
    FROM_FUTURE = "FROM_FUTURE"
    FROM_PAST_AND_FUTURE = "FROM_PAST_AND_FUTURE"


class BirdSupportedColumnDataFormats(ListConvertableEnum):
    INTEGER = "integer"
    REAL = "real"
    DATE = "date"
    DATETIME = "datetime"
    TEXT = "text"
