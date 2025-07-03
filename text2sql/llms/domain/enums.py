from text2sql.commons.list_enum import ListConvertableEnum


class SupportedLlmProviders(str, ListConvertableEnum):
    azure = "azure"
    vertex_ai = "vertex-ai"


class SupportedLlms(str, ListConvertableEnum):
    gpt_4_32k = "gpt-4-32k"
    gpt_4o = "gpt-4o"
    gpt_4o_mini = "gpt-4o-mini"
    gemini_pro = "gemini-pro"
    gemini_flash = "gemini-flash"

    opencoder_1_5b = "opencoder-1.5b"
    opencoder_8b = "opencoder-8b"