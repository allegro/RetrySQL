import argparse
from upath import UPath
import json
import os

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader

from text2sql.linear_probing.model import LinearProbingModel
from text2sql.llm_training.domain.model import LinearProbingConfiguration
from text2sql.linear_probing.utils import padding_fn
from text2sql.commons.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--validation_set_path", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--backbone_model_name_or_path", type=str, default="infly/OpenCoder-1.5B-Base", help="Path to the backbone model")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="infly/OpenCoder-1.5B-Base", help="Path to the tokenizer")
    parser.add_argument("--plot_embeddings", action=argparse.BooleanOptionalAction, help="If set, t-SNE plotting will be triggered")
    args = parser.parse_args()
    return args


class InferenceDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def read_validation_set(path_to_eval_set: str) -> list[dict]:
    data = []
    with UPath(path_to_eval_set).open("r", encoding="utf-8") as json_file:
        for line in json_file:
            line = line.strip()
            try:
                json_data = json.loads(line)
                data.append(json_data)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSON reading error with object: {line}") from error
    return data


def are_embeddings_and_labels_already_stored(output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, "embeddings.npy")) and os.path.exists(os.path.join(output_dir, "labels.npy"))


def load_model(checkpoint_path: str, config: LinearProbingConfiguration) -> LinearProbingModel:
    logger.info("Loading Model from checkpoint")
    model = LinearProbingModel.load_from_checkpoint(checkpoint_path, config=config)
    logger.info("Model loaded!")
    model.eval()
    return model

def extract_embeddings_and_labels(
        model: LinearProbingModel,
        dataloader: torch.utils.data.DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    all_embeddings = []
    all_labels = []

    for batch_idx, batch in enumerate(dataloader):
        logger.info("Processing batch_idx:", batch_idx)

        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['label']
        embeddings = model.extract_last_token_embeddings(input_ids.to(model.device), attention_mask.to(model.device))

        all_embeddings.append(embeddings)
        all_labels.append(labels)

    return torch.cat(all_embeddings).detach().numpy(), torch.cat(all_labels).detach().numpy()


def save_embeddings(
        embeddings: np.ndarray,
        labels: np.ndarray,
        output_dir: str,
) -> None:
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(output_dir, "labels.npy"), labels)


def get_inference_dataloader(validation_set_path: str, batch_size: int) -> torch.utils.data.DataLoader:
    validation_set = read_validation_set(validation_set_path)
    inference_dataset = InferenceDataset(validation_set)
    return DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=padding_fn
    )


def plot_tsne_embeddings(embeddings: np.ndarray, labels: np.ndarray, output_dir: str) -> None:
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    df = pd.DataFrame({
        'Component 1': embeddings_2d[:, 0],
        'Component 2': embeddings_2d[:, 1],
        'Label': labels
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Component 1', y='Component 2', hue='Label', palette='colorblind', alpha=0.5, data=df)
    plt.xlabel('t-SNE component 1', fontsize=24)
    plt.ylabel('t-SNE component 2', fontsize=24)
    plt.legend(fontsize=20)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embeddings_plot.svg"), format='svg')
    plt.show()

def main():
    args = parse_args()

    if are_embeddings_and_labels_already_stored(args.output_dir):
        embeddings = np.load(os.path.join(args.output_dir, "embeddings.npy"))
        labels = np.load(os.path.join(args.output_dir, "labels.npy"))

    else:
        config = LinearProbingConfiguration(
            backbone_model_name_or_path=args.backbone_model_path,
            tokenizer_name_or_path=args.tokenizer_path
        )

        inference_dataloader = get_inference_dataloader(args.validation_set_path, args.batch_size)

        model = load_model(args.checkpoint_path, config)
        embeddings, labels = extract_embeddings_and_labels(model, inference_dataloader)
        save_embeddings(embeddings, labels, args.output_dir)

    if args.plot_embeddings:
        plot_tsne_embeddings(embeddings, labels, args.output_dir)


if __name__ == '__main__':
    main()
