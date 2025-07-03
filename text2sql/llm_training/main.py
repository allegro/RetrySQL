import argparse
import json
import logging

from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PaddingStrategy
from upath import UPath

from text2sql.commons.io_utils import translate_gcs_dir_to_local, translate_gcs_file_to_local
from text2sql.commons.logging_utils import setup_logger
from text2sql.datasets.domain.model import PretrainDataSample
from text2sql.datasets.llm_training.instruction_fine_tuning_dataset import InstructionFineTuningDataset
from text2sql.datasets.llm_training.pretraining_dataset import PretrainingDataset
from text2sql.llm_training.configuration import PaddingSide, PretrainingConfigurationDto
from text2sql.llm_training.domain.constants import TokenizationConstants
from text2sql.llm_training.domain.enums import SupportedTrainingModes, ModelExecutionModes
from text2sql.llm_training.domain.model import PretrainingConfiguration
from text2sql.llm_training.trainer import Trainer

logger = logging.getLogger(__name__)


def parse_args() -> PretrainingConfigurationDto:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-mode",
        type=SupportedTrainingModes,
        choices=SupportedTrainingModes.get_values(),
        required=True,
        help="Decide whether to perform pre-training or fine-tuning."
    )
    parser.add_argument(
        "--llm-training-data-path",
        type=lambda path: translate_gcs_file_to_local(UPath(path)),
        required=True,
        help="Path to the data that will be used for either pre-training of supervised instruction fine-tuning"
    )
    parser.add_argument(
        "--hf-access-token",
        type=str,
        default="",
        help=("Some models are restricted/gated. To access them, you need to go to their huggingface page and accept"
              " the license/terms and then provide your huggingface access token here.")
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=2,
        help="Batch size per GPU."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=lambda path: translate_gcs_dir_to_local(UPath(path)),
        required=False,
        default=None,
        help="Path to a model to be used as a starting point for pre-training or fine-tuning."
    )
    parser.add_argument(
        "--pretrained-model-name",
        type=str,
        default="infly/OpenCoder-1.5B-Base",
        help="Huggingface model name to be used as a starting point for pre-training or fine-tuning."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs for pre-training."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate."
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help=("Number of gradient accumulation steps. The value provided here should be the same as the one provided in"
              " the accelerate config file.")
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Ratio of total training steps used for a linear warmup."
    )
    parser.add_argument(
        "--checkpoint-every-n-steps",
        type=int,
        default=300,
        help="Number of steps after which the checkpoint is saved."
    )
    parser.add_argument(
        "--output-dir",
        type=UPath,
        default="./output",
        help="Working directory."
    )
    parser.add_argument(
        "--save-all-states",
        action="store_true",
        help=("Whether to save the model, optimizer, and lr scheduler states during checkpointing in order to be able to"
             " resume training. Otherwise only the model params are saved.")
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=lambda path: translate_gcs_dir_to_local(UPath(path)),
        default=None,
        help="Path to a dir with checkpoint. If provided, the training will resume from this checkpoint"
    )
    parser.add_argument(
        "--padding-strategy",
        type=PaddingStrategy,
        choices=[PaddingStrategy.LONGEST, PaddingStrategy.MAX_LENGTH],
        default=PaddingStrategy.LONGEST,
        help="Padding strategy for the dataset."
    )
    parser.add_argument(
        "--padding-multiple",
        type=int,
        default=8,
        help="Pad to a multiple of this value."
    )
    parser.add_argument(
        "--padding-side",
        type=PaddingSide,
        choices=PaddingSide.get_values(),
        default=PaddingSide.RIGHT,
    )
    parser.add_argument(
        "--adamw-beta1",
        type=float,
        default=0.9,
        help="Sets beta1 for AdamW optimizer."
    )
    parser.add_argument(
        "--adamw-beta2",
        type=float,
        default=0.95,
        help="Sets beta2 for AdamW optimizer."
    )
    parser.add_argument(
        "--adamw-eps",
        type=float,
        default=1e-8,
        help="Sets eps for AdamW optimizer."
    )
    parser.add_argument(
        "--adamw-weight-decay",
        type=float,
        default=0.1,
        help="Sets weight decay for AdamW optimizer."
    )
    parser.add_argument(
        "--resize-token-embeddings",
        action=argparse.BooleanOptionalAction,
        help="If set, model's token embeddings will be resized to match the tokenizer's vocab size."
    )

    args = parser.parse_args()

    return PretrainingConfigurationDto(
        llm_training_data_path=args.llm_training_data_path,
        training_mode=args.training_mode,
        hf_access_token=args.hf_access_token,
        per_device_batch_size=args.per_device_batch_size,
        seed=args.seed,
        pretrained_model_name=args.pretrained_model_name,
        pretrained_model_path=args.pretrained_model_path,
        epochs=args.epochs,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        output_dir=args.output_dir,
        save_all_states=args.save_all_states,
        resume_from_checkpoint=args.resume_from_checkpoint,
        padding_strategy=args.padding_strategy,
        padding_multiple=args.padding_multiple,
        padding_side=args.padding_side,
        adamw_beta1=args.adamw_beta1,
        adamw_beta2=args.adamw_beta2,
        adamw_epsilon=args.adamw_eps,
        adamw_weight_decay=args.adamw_weight_decay,
        resize_token_embeddings=args.resize_token_embeddings
    )


def _get_model_and_tokenizer(
    pretrained_model_name: str,
    hf_access_token: str,
    training_mode: SupportedTrainingModes,
    pretrained_model_path: UPath | None = None,
    resize_token_embeddings: bool = False
):
    if pretrained_model_path is not None:
        logger.info(f"Loading {pretrained_model_path} model")
    else:
        logger.info(f"Loading model from HuggingFace model hub: {pretrained_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path.joinpath("tokenizer") if pretrained_model_path is not None else pretrained_model_name,
        token=hf_access_token,
        trust_remote_code=True,
        additional_special_tokens=[
            TokenizationConstants.BACK_TOKEN, TokenizationConstants.CONTEXT_TOKEN, TokenizationConstants.QUESTION_TOKEN,
            TokenizationConstants.REASONING_TOKEN, TokenizationConstants.SQL_TOKEN
        ]
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path.joinpath("model") if pretrained_model_path is not None else pretrained_model_name,
        token=hf_access_token,
        trust_remote_code=True
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    if training_mode == SupportedTrainingModes.PRETRAINING:
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
    elif training_mode == SupportedTrainingModes.INSTRUCTION_FINE_TUNING:
        model.config.eos_token_id = tokenizer.convert_tokens_to_ids(TokenizationConstants.EOT_TOKEN)
        model.config.bos_token_id = tokenizer.convert_tokens_to_ids(TokenizationConstants.BOT_TOKEN)
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")

    if resize_token_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def _load_pretrain_data(input_data_path: UPath) -> list[PretrainDataSample]:
    with input_data_path.open("r") as input_file:
        lines = input_file.readlines()
    return [PretrainDataSample(**json.loads(line)) for line in lines]


if __name__ == "__main__":
    setup_logger()

    config = PretrainingConfiguration.from_dto(parse_args())
    model, tokenizer = _get_model_and_tokenizer(
        pretrained_model_name=config.pretrained_model_name,
        pretrained_model_path=config.pretrained_model_path,
        hf_access_token=config.hf_access_token,
        resize_token_embeddings=config.resize_token_embeddings,
        training_mode=config.training_mode
    )

    config.hf_access_token = "***"
    logger.info(f"Arguments: {config}")

    dataset_class = PretrainingDataset if config.training_mode == SupportedTrainingModes.PRETRAINING else InstructionFineTuningDataset
    trainer = Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset_class(
            input_data=_load_pretrain_data(config.llm_training_data_path),
            tokenizer=tokenizer,
            mode=ModelExecutionModes.TRAIN
        ),
        accelerator=Accelerator(
            gradient_accumulation_plugin=GradientAccumulationPlugin(
                num_steps=config.gradient_accumulation_steps,
                sync_with_dataloader=False
            )
        ),
        writer=SummaryWriter(config.output_dir / "tensorboard_logs")
    )
    trainer.run_training()
