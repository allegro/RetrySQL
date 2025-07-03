from dataclasses import dataclass

import torch

@dataclass
class InferenceExample:
    question_id: str
    question: str
    prompt: str
    tokenized_prompt: dict[str, list[int]]

@dataclass
class InferenceBatch:
    question_ids: list[str]
    prompts: list[str]
    questions: list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

@dataclass
class ConfidenceScore:
    mean: float
    std: float

@dataclass
class ConfidenceScorePair:
    before: ConfidenceScore
    after: ConfidenceScore


@dataclass
class ConfidenceOutput:
    question_id: int
    probas: list[ConfidenceScorePair]
