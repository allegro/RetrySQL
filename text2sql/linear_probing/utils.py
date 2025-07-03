import torch

def padding_fn(batch, pad_token_id):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["label"] for item in batch]

    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = []
    attention_mask = []

    for ids in input_ids:
        padding_length = max_len - len(ids)
        padded_ids = ids + [pad_token_id] * padding_length
        padded_input_ids.append(padded_ids)
        attention_mask.append([1] * len(ids) + [0] * padding_length)

    padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "label": labels
    }
