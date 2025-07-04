from text2sql.commons.list_enum import ListConvertableEnum


class SupportedAlgorithms(str, ListConvertableEnum):
    EXACT_MATCHING = "EXACT_MATCHING"
    EDIT_DISTANCE = "EDIT_DISTANCE"
    NEAREST_NEIGHBOUR_SEARCH = "NEAREST_NEIGHBOUR_SEARCH"
    LLM_BASED = "LLM_BASED"


class AvailableEmbeddingModels(str, ListConvertableEnum):
    DUMMY_MODEL = "DUMMY_MODEL"
    TEXT_EMBEDDING_ADA_002 = "TEXT_EMBEDDING_ADA_002"
    TEXT_EMBEDDING_3_LARGE = "TEXT_EMBEDDING_3_LARGE"
    SNOWFLAKE_ARCTIC_EMBED_M = "SNOWFLAKE_ARCTIC_EMBED_M"
    SNOWFLAKE_ARCTIC_EMBED_M_LONG = "SNOWFLAKE_ARCTIC_EMBED_M_LONG"
    SNOWFLAKE_ARCTIC_EMBED_L = "SNOWFLAKE_ARCTIC_EMBED_L"
