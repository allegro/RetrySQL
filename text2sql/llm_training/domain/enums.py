from text2sql.commons.list_enum import ListConvertableEnum

class SupportedTrainingModes(str, ListConvertableEnum):
    PRETRAINING = "PRETRAINING"
    INSTRUCTION_FINE_TUNING = "INSTRUCTION_FINE_TUNING"


class ModelExecutionModes(str, ListConvertableEnum):
    TRAIN = "TRAIN"
    EVAL = "EVAL"

class PaddingSide(str, ListConvertableEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class PretrainingDataTypes(str, ListConvertableEnum):
    WITH_REASONING = "WITH_REASONING"
    WITHOUT_REASONING = "WITHOUT_REASONING"