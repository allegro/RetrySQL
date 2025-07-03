import json
from dataclasses import dataclass, field
from typing import Optional, Union
from upath import UPath
from functools import partial

from transformers import HfArgumentParser, set_seed, AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PaddingStrategy
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig

from text2sql.datasets.domain.model import PretrainDataSample
from text2sql.llm_training.domain.enums import SupportedTrainingModes, ModelExecutionModes
from text2sql.datasets.llm_training.instruction_fine_tuning_dataset import InstructionFineTuningDataset
from text2sql.llm_training.domain.constants import TokenizationConstants
from text2sql.llm_training.configuration import PaddingSide
from text2sql.datasets.llm_training.batching import pad_batch


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    llm_training_data_path: str  = field(
        metadata={"help": "Path to the data that will be used for either pre-training of supervised instruction fine-tuning"}
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[Union[list[str], str]] = field(
        default="q_proj,v_proj,embed_tokens,lm_head",
        metadata={"help": "comma separated list of target modules to apply LoRA layers to"}
    )


def _load_pretrain_data(input_data_path: UPath) -> list[PretrainDataSample]:
    with input_data_path.open("r") as input_file:
        lines = input_file.readlines()
    return [PretrainDataSample(**json.loads(line)) for line in lines]


def get_instruction_fine_tuning_dataset(llm_training_data_path, tokenizer) -> InstructionFineTuningDataset:
    return InstructionFineTuningDataset(
            input_data=_load_pretrain_data(UPath(llm_training_data_path)),
            tokenizer=tokenizer,
            mode=ModelExecutionModes.TRAIN
        )

def setup_lora_training_model(args) -> tuple[PeftModel, PeftConfig, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        additional_special_tokens=[
            TokenizationConstants.BACK_TOKEN, TokenizationConstants.CONTEXT_TOKEN, TokenizationConstants.QUESTION_TOKEN,
            TokenizationConstants.REASONING_TOKEN, TokenizationConstants.SQL_TOKEN
        ]
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.convert_tokens_to_ids(TokenizationConstants.EOT_TOKEN)
    model.config.bos_token_id = tokenizer.convert_tokens_to_ids(TokenizationConstants.BOT_TOKEN)

    peft_config = LoraConfig(
        lora_dropout=args.lora_dropout,
        lora_alpha=args.lora_alpha,
        r=args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules
    )

    return get_peft_model(model, peft_config), peft_config, tokenizer


def main(model_args, training_args) -> None:
    set_seed(training_args.seed)

    model, peft_config, tokenizer = setup_lora_training_model(model_args)

    train_dataset = get_instruction_fine_tuning_dataset(model_args.llm_training_data_path, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        data_collator=partial(
            pad_batch,
            tokenizer=tokenizer,
            padding_strategy=PaddingStrategy.LONGEST,
            pad_to_multiple_of=8,
            label_mask=TokenizationConstants.LABEL_MASK,
            training_mode=SupportedTrainingModes.INSTRUCTION_FINE_TUNING,
            padding_side=PaddingSide.RIGHT.lower()
            )
        )
    trainer.accelerator.print(f"{trainer.model}")
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, SFTConfig))
    model_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, training_args)