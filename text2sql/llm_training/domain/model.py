from __future__ import annotations
from dataclasses import dataclass

from transformers.tokenization_utils import PaddingStrategy
from upath import UPath

from text2sql.llm_training.domain.enums import SupportedTrainingModes
from text2sql.llm_training.configuration import PaddingSide, PretrainingConfigurationDto

@dataclass
class PaddingConfiguration:
    padding_strategy: PaddingStrategy
    padding_multiple: int
    padding_side: PaddingSide


@dataclass
class OptimizerConfiguration:
    beta_1: float
    beta_2: float
    epsilon: float
    weight_decay: float


@dataclass
class PretrainingConfiguration:
    llm_training_data_path: UPath
    training_mode: SupportedTrainingModes
    hf_access_token: str
    per_device_batch_size: int
    seed: int
    pretrained_model_name: str
    pretrained_model_path: str | None
    epochs: int
    learning_rate: float
    gradient_accumulation_steps: int
    warmup_ratio: float
    checkpoint_every_n_steps: int
    output_dir: UPath
    save_all_states: bool
    resume_from_checkpoint: None | UPath
    padding_configuration: PaddingConfiguration
    optimizer_configuration: OptimizerConfiguration
    resize_token_embeddings: bool = False

    @staticmethod
    def from_dto(dto: PretrainingConfigurationDto) -> PretrainingConfiguration:
        return PretrainingConfiguration(
            llm_training_data_path=dto.llm_training_data_path,
            training_mode=dto.training_mode,
            hf_access_token=dto.hf_access_token,
            per_device_batch_size=dto.per_device_batch_size,
            seed=dto.seed,
            pretrained_model_name=dto.pretrained_model_name,
            pretrained_model_path=dto.pretrained_model_path,
            epochs=dto.epochs,
            learning_rate=dto.learning_rate,
            gradient_accumulation_steps=dto.gradient_accumulation_steps,
            warmup_ratio=dto.warmup_ratio,
            checkpoint_every_n_steps=dto.checkpoint_every_n_steps,
            output_dir=dto.output_dir,
            save_all_states=dto.save_all_states,
            resume_from_checkpoint=dto.resume_from_checkpoint,
            padding_configuration=PaddingConfiguration(
                padding_strategy=dto.padding_strategy,
                padding_multiple=dto.padding_multiple,
                padding_side=dto.padding_side
            ),
            optimizer_configuration=OptimizerConfiguration(
                beta_1=dto.adamw_beta1,
                beta_2=dto.adamw_beta2,
                epsilon=dto.adamw_epsilon,
                weight_decay=dto.adamw_weight_decay
            ),
            resize_token_embeddings=dto.resize_token_embeddings
        )
    
    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "llm_training_data_path": str(self.llm_training_data_path),
            "training_mode": self.training_mode.value,
            "hf_access_token": self.hf_access_token,
            "per_device_batch_size": self.per_device_batch_size,
            "seed": self.seed,
            "pretrained_model_name": self.pretrained_model_name,
            "pretrained_model_path": self.pretrained_model_path,
            "epochs": self.epochs,
            "lr": self.learning_rate,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_ratio": self.warmup_ratio,
            "checkpoint_every_n_steps": self.checkpoint_every_n_steps,
            "output_dir": str(self.output_dir),
            "save_all_states": self.save_all_states,
            "resume_from_checkpoint": str(self.resume_from_checkpoint),
            "padding_configuration": {
                "padding_strategy": str(self.padding_configuration.padding_strategy),
                "padding_multiple": self.padding_configuration.padding_multiple,
                "padding_side": str(self.padding_configuration.padding_side)
            },
            "optimizer_configuration": {
                "beta_1": self.optimizer_configuration.beta_1,
                "beta_2": self.optimizer_configuration.beta_2,
                "epsilon": self.optimizer_configuration.epsilon,
                "weight_decay": self.optimizer_configuration.weight_decay
            },
            "resize_token_embeddings": self.resize_token_embeddings
        }

@dataclass
class LinearProbingConfiguration:
    # backbone
    backbone_model_name_or_path : str
    tokenizer_name_or_path : str

    # dataset
    path_to_dataset: str = ""
    train_val_split: float = 0.8
    num_workers: int = 1

    # training
    lr: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 3

    # pytorch-lightning trainer args
    num_sanity_val_steps: int = 0
    log_every_n_steps: int = 1
    val_check_interval: float = 0.25

