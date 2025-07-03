from dataclasses import dataclass

from transformers import PreTrainedModel, PreTrainedTokenizer
from pydantic import BaseModel, Field


class LlmParams(BaseModel):
    temperature: float = 0.0
    max_output_tokens: int = Field(desc="The maximum number of tokens to generate by the model")
    model_total_max_tokens: int = Field(
        desc="Context length of the model - maximum number of input plus generated tokens",
    )
    max_concurrency: int | None = Field(default=50, desc="Maximum number of concurrent requests")
    max_retries: int | None = Field(default=8, desc="Maximum number of retries if a request fails")


class AzureOpenAIParams(LlmParams):
    request_timeout_s: int | None = Field(default=120, desc="Timeout for requests to the model in seconds")


class VertexAIParams(LlmParams):
    top_k: int | None = Field(
        default=40,
        desc="Changes how the model selects tokens for output. "
        "A top-k of 3 means that the next token is selected from among the 3 most probable tokens.",
    )
    top_p: float | None = Field(
        default=0.95,
        desc="Top-p changes how the model selects tokens for output. "
        "Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.",
    )
    verbose: bool | None = True


class BasePrompt(BaseModel):
    question_id: str
    prompt: str


class OpenAIEmbeddingModelEndpoint(BaseModel):
    provider: str
    deployment: str
    model: str
    openai_api_base: str
    openai_api_type: str
    request_timeout: float
    max_concurrency: int
    max_retries: int
    openai_api_key: str | None = None


@dataclass
class OpenSourceLlm:
    llm: PreTrainedModel
    tokenizer: PreTrainedTokenizer
