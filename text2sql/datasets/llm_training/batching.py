import torch

from transformers import AutoTokenizer
from transformers.tokenization_utils import PaddingStrategy

from text2sql.llm_training.domain.enums import SupportedTrainingModes

def pad_batch(
        batch: dict[str, list[int]],
        tokenizer: AutoTokenizer,
        padding_strategy: PaddingStrategy,
        pad_to_multiple_of: int,
        label_mask: int,
        training_mode: SupportedTrainingModes,
        padding_side: str = "right"
) -> dict[str, torch.Tensor]:

    batch_size = len(batch)

    padded_input_ids_and_attention_mask = pad_input_ids_and_attention_mask(
        batch, tokenizer, padding_strategy, pad_to_multiple_of, padding_side
    )

    padded_labels = generate_labels(
        batch,
        padded_input_ids_and_attention_mask,
        training_mode,
        tokenizer,
        label_mask,
        padding_side,
        batch_size,
    )

    return {
        "input_ids": padded_input_ids_and_attention_mask["input_ids"],
        "attention_mask": padded_input_ids_and_attention_mask["attention_mask"],
        "labels": padded_labels
    }


def pad_input_ids_and_attention_mask(
        batch: dict[str, list[int]],
        tokenizer: AutoTokenizer,
        padding_strategy: PaddingStrategy,
        pad_to_multiple_of: int,
        padding_side: str
) -> dict[str, torch.Tensor]:
    return tokenizer.pad(
        encoded_inputs=[
            {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]}
            for example in batch
        ],
        padding=padding_strategy,
        pad_to_multiple_of=pad_to_multiple_of,
        padding_side=padding_side,
        return_tensors="pt"
    )


def generate_labels(
        original_batch: dict[str, list[int]],
        padded_batch: dict[str, torch.Tensor],
        training_mode: SupportedTrainingModes,
        tokenizer: AutoTokenizer,
        label_mask: int,
        padding_side: str,
        batch_size: int
) -> torch.Tensor:
    if training_mode == SupportedTrainingModes.PRETRAINING:
        return generate_pretraining_labels(padded_batch, label_mask)
    elif training_mode == SupportedTrainingModes.INSTRUCTION_FINE_TUNING:
        return generate_fine_tuning_labels(
            original_batch, padded_batch, tokenizer, label_mask, padding_side, batch_size
        )
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")


def generate_pretraining_labels(
        padded_batch: dict[str, torch.Tensor],
        label_mask: int
) -> torch.Tensor:
    labels = padded_batch["input_ids"].clone()
    labels[padded_batch["attention_mask"] == 0] = label_mask
    return labels


def generate_fine_tuning_labels(
        batch: dict[str, list[int]],
        padded_batch: dict[str, torch.Tensor],
        tokenizer: AutoTokenizer,
        label_mask: int,
        padding_side: str,
        batch_size: int
) -> torch.Tensor:
    labels_list = []
    for batch_id in range(batch_size):
        padding_length = len(padded_batch["input_ids"][batch_id]) - len(batch[batch_id]["input_ids"])
        padded_target_ids = pad_target_ids(batch[batch_id]["labels"], padding_length, tokenizer, padding_side)
        labels_list.append(padded_target_ids)

    labels = torch.tensor(labels_list, dtype=padded_batch["input_ids"].dtype)
    labels[padded_batch["attention_mask"] == 0] = label_mask
    return labels


def pad_target_ids(
        target_ids: list[int],
        padding_length: int,
        tokenizer: AutoTokenizer,
        padding_side: str
) -> list[int]:

    if padding_side == "left":
        return [tokenizer.pad_token_id] * padding_length + target_ids
    elif padding_side == "right":
        return target_ids + [tokenizer.pad_token_id] * padding_length
    else:
        raise ValueError(f"Unsupported padding side: {padding_side}")