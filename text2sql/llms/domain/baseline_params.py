from text2sql.llms.domain.enums import SupportedLlms
from text2sql.llms.domain.models import AzureOpenAIParams, VertexAIParams


class Gpt4BaselineParams(AzureOpenAIParams):
    max_concurrency: int | None = 5
    max_output_tokens: int = 2048
    request_timeout_s: int | None = 120
    model_total_max_tokens: int = 32000
    temperature: float = 0.0


class Gpt4oBaselineParams(AzureOpenAIParams):
    max_concurrency: int | None = 50
    max_output_tokens: int = 2048
    request_timeout_s: int | None = 120
    model_total_max_tokens: int = 128000
    temperature: float = 0.0


class Gpt4oMiniBaselineParams(AzureOpenAIParams):
    max_concurrency: int | None = 50
    max_output_tokens: int = 2048
    request_timeout_s: int | None = 120
    model_total_max_tokens: int = 128000
    temperature: float = 0.0


class GeminiProBaselineParams(VertexAIParams):
    max_concurrency: int | None = 5
    max_output_tokens: int = 2048
    model_total_max_tokens: int = 128000
    temperature: float = 0.0


class GeminiFlashBaselineParams(VertexAIParams):
    max_concurrency: int | None = 5
    max_output_tokens: int = 2048
    model_total_max_tokens: int = 128000
    temperature: float = 0.0


def get_baseline_llm_params(name: SupportedLlms) -> AzureOpenAIParams | VertexAIParams:
    match name:
        case SupportedLlms.gpt_4_32k:
            return Gpt4BaselineParams()
        case SupportedLlms.gpt_4o:
            return Gpt4oBaselineParams()
        case SupportedLlms.gpt_4o_mini:
            return Gpt4oMiniBaselineParams()
        case SupportedLlms.gemini_pro:
            return GeminiProBaselineParams()
        case SupportedLlms.gemini_flash:
            return GeminiFlashBaselineParams()
        case _:
            raise ValueError(
                "Unsupported model name: '%s'. Provide one of %s" % (name, SupportedLlms.get_values()),
            )
