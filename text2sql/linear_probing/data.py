import json
from upath import UPath
from random import shuffle
from typing import Any
from functools import partial

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from text2sql.datasets.domain.model import PretrainDataSample
from text2sql.llm_training.domain.enums import ModelExecutionModes
from text2sql.datasets.llm_training.linear_probing import LinearProbingDataset
from text2sql.llm_training.domain.constants import TokenizationConstants
from text2sql.llm_training.domain.model import LinearProbingConfiguration
from text2sql.linear_probing.utils import padding_fn
from text2sql.commons.logging_utils import get_logger

logger = get_logger(__name__)

class TokenizedData(Dataset):
    def __init__(self, dataset) -> None:
        self.raw_dataset = dataset
        self.dataset = [
            {
                "input_ids": sample.tokenized_text,
                "label": sample.label
            }
            for sample in self.raw_dataset
        ]


    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> dict[str, Any]:
        return self.dataset[idx]

    def shuffle(self) -> None:
        shuffle(self.dataset)

class LinearProbingDataModule(pl.LightningDataModule):
    def __init__(self, config: LinearProbingConfiguration):
        super().__init__()
        self.config = config

        self.tokenizer = self._get_tokenizer(self.config.tokenizer_name_or_path)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.full_dataset = TokenizedData(
            self.get_linear_probing_dataset(self.config.path_to_dataset, self.tokenizer)
            .tokenized_data
        )

        self.train_dataset = None
        self.val_dataset = None


    @staticmethod
    def _load_pretrain_data(input_data_path: UPath) -> list[PretrainDataSample]:
        with input_data_path.open("r") as input_file:
            lines = input_file.readlines()
        return [PretrainDataSample(**json.loads(line)) for line in lines][0:10]

    @staticmethod
    def _get_tokenizer(tokenizer_name_or_path: str):
          return AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            trust_remote_code=True,
            additional_special_tokens=[
                TokenizationConstants.BACK_TOKEN, TokenizationConstants.CONTEXT_TOKEN,
                TokenizationConstants.QUESTION_TOKEN,
                TokenizationConstants.REASONING_TOKEN, TokenizationConstants.SQL_TOKEN
            ]
        )

    @staticmethod
    def get_linear_probing_dataset(llm_training_data_path, tokenizer) -> LinearProbingDataset:
        linear_probing_dataset = LinearProbingDataset(
            input_data=LinearProbingDataModule._load_pretrain_data(UPath(llm_training_data_path)),
            tokenizer=tokenizer,
            mode=ModelExecutionModes.TRAIN
        )
        logger.info(f"Linear probing dataset created with {len(linear_probing_dataset)} samples")

        logger.info("Labels distribution:")
        num_of_wrong_sentences = 0
        num_of_correct_sentences = 0
        for sample in linear_probing_dataset.tokenized_data:
            if sample.label == 1:
                num_of_wrong_sentences += 1
            else:
                num_of_correct_sentences += 1
        logger.info(f"Number of wrong sentences: {num_of_wrong_sentences}")
        logger.info(f"Number of correct sentences: {num_of_correct_sentences}")

        return linear_probing_dataset

    def setup(self, stage=None) -> None:
        self.full_dataset.shuffle()
        self.train_dataset = self.full_dataset[:int(len(self.full_dataset) * self.config.train_val_split)]
        self.val_dataset = self.full_dataset[int(len(self.full_dataset) * self.config.train_val_split):]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=partial(padding_fn, pad_token_id=self.pad_token_id),
            num_workers=self.config.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            collate_fn=partial(padding_fn, pad_token_id=self.pad_token_id),
            num_workers=self.config.num_workers
        )