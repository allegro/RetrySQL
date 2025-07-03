from dataclasses import dataclass

from transformers.tokenization_utils import PaddingStrategy
from upath import UPath

from text2sql.llm_training.domain.enums import SupportedTrainingModes, PaddingSide

@dataclass
class PretrainingConfigurationDto:
    llm_training_data_path: UPath
    training_mode: SupportedTrainingModes
    hf_access_token: str
    per_device_batch_size: int
    seed: int
    pretrained_model_name: str
    pretrained_model_path: str
    epochs: int
    learning_rate: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    checkpoint_every_n_steps: int
    output_dir: UPath
    save_all_states: bool
    resume_from_checkpoint: UPath
    padding_strategy: PaddingStrategy
    padding_side: PaddingSide
    padding_multiple: int
    adamw_beta1: float
    adamw_beta2: float
    adamw_epsilon: float
    adamw_weight_decay: float
    resize_token_embeddings: bool