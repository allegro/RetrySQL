import os
import tempfile

import torch
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from numpy.random import randint
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Config, PreTrainedTokenizerFast
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GPT2LMHeadModel, BatchEncoding
from transformers.utils import PaddingStrategy
from upath import UPath

from text2sql.llm_training.domain.enums import SupportedTrainingModes, PaddingSide
from text2sql.llm_training.domain.model import OptimizerConfiguration, PretrainingConfiguration, PaddingConfiguration
from text2sql.llm_training.trainer import Trainer


class DummyDataset(Dataset):
    def __init__(self, data: list[str], tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenized_dataset = self._tokenize_data(data, tokenizer)
        self._length = len(self._tokenized_dataset["input_ids"])
        self.num_calls_counter = 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self.num_calls_counter += 1
        return {
            "input_ids": self._tokenized_dataset["input_ids"][idx],
            "attention_mask": self._tokenized_dataset["input_ids"][idx],
            "labels": self._tokenized_dataset["input_ids"][idx]
        }

    @staticmethod
    def _tokenize_data(data: list[str], tokenizer: PreTrainedTokenizerBase) -> BatchEncoding:
         return tokenizer(data, padding="longest", return_tensors="pt")


class TestTrainer:

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # GIVEN
            vocab_size = 100
            dataset_size = 10
            config = PretrainingConfiguration(
                training_mode=SupportedTrainingModes.PRETRAINING,
                per_device_batch_size=3,
                seed=42,
                epochs=2,
                learning_rate=1e-3,
                gradient_accumulation_steps=2,
                warmup_ratio=0.1,
                checkpoint_every_n_steps=2,
                output_dir=UPath(tmpdir),
                save_all_states=True,
                resume_from_checkpoint=None,
                llm_training_data_path=UPath("dummy"),
                hf_access_token="",
                pretrained_model_name="dummy",
                pretrained_model_path=None,
                padding_configuration=PaddingConfiguration(
                    padding_strategy=PaddingStrategy.LONGEST,
                    padding_multiple=8,
                    padding_side=PaddingSide.LEFT
                ),
                optimizer_configuration=OptimizerConfiguration(
                    beta_1=0.9,
                    beta_2=0.95,
                    epsilon=1e-8,
                    weight_decay=0.1
                )
            )

            model, tokenizer = self._get_model_and_tokenizer(vocab_size)
            dataset = self._get_dataset(tokenizer, vocab_size, dataset_size)

            batch_size = config.per_device_batch_size * config.gradient_accumulation_steps
            num_steps_per_epoch = len(dataset) // batch_size
            num_total_steps = num_steps_per_epoch * config.epochs

            expected_dataset_calls = len(dataset) // config.per_device_batch_size * config.per_device_batch_size * config.epochs
            expected_num_checkpoints = num_total_steps // config.checkpoint_every_n_steps + 1 * config.epochs
            log_dir = os.path.join(tmpdir, Trainer.TENSORBOARD_LOG_DIR)
            checkpoint_dir = os.path.join(tmpdir, Trainer.CHECKPOINT_DIR)

            # WHEN
            trainer = Trainer(
                config=config,
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                accelerator=Accelerator(
                    gradient_accumulation_plugin=GradientAccumulationPlugin(
                        num_steps=config.gradient_accumulation_steps,
                        sync_with_dataloader=False
                    )
                ),
                writer=SummaryWriter(log_dir=os.path.join(config.output_dir, "tensorboard_logs")),
            )
            trainer.run_training()

            # THEN
            assert os.path.exists(checkpoint_dir), "Checkpoint directory was not created."
            assert os.path.exists(log_dir), "TensorBoard logs directory was not created."
            assert dataset.num_calls_counter == expected_dataset_calls, "Dataset was not called the expected number of times."
            assert len(os.listdir(checkpoint_dir)) == expected_num_checkpoints, "Number of checkpoints isn't as expected."
            
            for single_checkpoint_dir in os.listdir(checkpoint_dir):
                assert os.path.exists(os.path.join(checkpoint_dir, single_checkpoint_dir, "model_weights_and_tokenizer")), "Model weights and tokenizer checkpoint missing."
                assert os.path.exists(os.path.join(checkpoint_dir, single_checkpoint_dir, "accelerate_states")), "Accelerate states checkpoint missing."
                assert os.path.exists(os.path.join(checkpoint_dir, single_checkpoint_dir, "config_files")), "Config files directory was not created."
                assert os.path.exists(os.path.join(checkpoint_dir, single_checkpoint_dir, "config_files", "accelerate_config.yaml")), "Accelerate config was not created."
                assert os.path.exists(os.path.join(checkpoint_dir, single_checkpoint_dir, "config_files", "main_config.yaml")), "Main config was not created."

    def _get_model_and_tokenizer(self, vocab_size: int) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        tokenizer = self._create_dummy_tokenizer(vocab_size)
        config = GPT2Config(
            n_layer=1, n_head=1, n_embd=32, vocab_size=vocab_size, eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id
        )
        model = GPT2LMHeadModel(config)
        model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    @staticmethod
    def _get_dataset(tokenizer: PreTrainedTokenizerBase, vocab_size: int, dataset_size: int) -> DummyDataset:
        data = [
            f"token_{randint(1, vocab_size)} token_{randint(1, vocab_size)} token_{randint(1, vocab_size)}"
            for _ in range(dataset_size)
        ]
        
        return DummyDataset(
            data=data,
            tokenizer=tokenizer
        )

    @staticmethod
    def _create_dummy_tokenizer(vocab_size: int) -> PreTrainedTokenizerBase:
        vocab = {"<bos>": 0, "<eos>": 0, "<unk>": 0, "<pad>": 0}
        vocab = {
            **vocab,
            **{f"token_{i}": i for i in range(1, vocab_size)}
        }

        tokenizer_backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        tokenizer_backend.pre_tokenizer = Whitespace()

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_backend,
            bos_token="<bos>",
            eos_token="<eos>",
            unk_token="<unk>",
            pad_token="<pad>"
        )

        return tokenizer
