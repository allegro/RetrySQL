from dataclasses import dataclass

@dataclass
class TokenizedPretrainDataSample:
    question_id: str
    tokenized_text: list[int]
    attention_mask: list[int]

@dataclass
class TokenizedLinearProbingDataSample:
    question_id: str
    tokenized_text: list[int]
    attention_mask: list[int]
    label: int

@dataclass
class TokenizationInputSample:
    question_id: str
    input_text: str

@dataclass
class LinearProbingInputSample:
    question_id: str
    input_text: str
    label: int

@dataclass
class InstructionFineTuningDataSample:
    question_id: str
    instruction: str
    expected_answer: str | None = None

@dataclass
class TokenizedInstructionFineTuningDataSample:
    question_id: str
    input_ids: list[int]
    attention_mask: list[int]
    target_ids: list[int] | None = None
