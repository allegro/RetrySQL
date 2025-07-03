import json
import logging
from functools import partial

import gcsfs
import math
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.data_loader import DataLoaderAdapter
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import get_cosine_schedule_with_warmup
from upath import UPath

from text2sql.commons.io_utils import copy_local_dir_to_gcp_dir, remove_local_dir, save_configuration_to_yaml
from text2sql.llm_training.domain.model import PretrainingConfiguration
from text2sql.llm_training.domain.constants import TokenizationConstants
from text2sql.datasets.llm_training.batching import pad_batch


logger = logging.getLogger(__name__)


class Trainer:
    CHECKPOINT_DIR = "checkpoints"
    TENSORBOARD_LOG_DIR = "tensorboard_logs"
    METADATA_FILENAME = "metadata.json"

    def __init__(
            self,
            config: PretrainingConfiguration,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase,
            dataset: Dataset,
            accelerator: Accelerator,
            writer: SummaryWriter
    ) -> None:
        set_seed(config.seed)

        self._config = config
        self._model = model
        self._tokenizer = tokenizer
        self._dataset = dataset
        self._accelerator = accelerator
        self._writer = writer

    def __call__(self) -> None:
        self.run_training()

    def run_training(self) -> None:
        logger.info(f"accelerator.device: {self._accelerator.device}")
        logger.info(f"accelerator.is_main_process: {self._accelerator.is_main_process}")

        self._model, optimizer, dataloader, lr_scheduler = self._prepare_model_optimizer_dataloader_and_scheduler()

        if self._config.resume_from_checkpoint:
            global_step, resume_epoch, resume_batch = self._resume_from_checkpoint(len(dataloader))
        else:
            global_step, resume_epoch, resume_batch = 0, 0, 0

        accumulated_loss = 0.0
        self._model.train()

        for epoch in range(resume_epoch, self._config.epochs):
            if self._accelerator.is_main_process:
                logger.info(f"Starting training epoch {epoch}")

            if epoch == resume_epoch and resume_batch > 0:
                current_dataloader = self._accelerator.skip_first_batches(dataloader, resume_batch)
            else:
                current_dataloader = dataloader

            for batch in current_dataloader:
                with self._accelerator.accumulate(self._model):
                    outputs = self._model(**batch)
                    loss = outputs.loss
                    self._accelerator.backward(loss)
                    accumulated_loss += loss.item()

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if self._accelerator.sync_gradients:
                    device_avg_loss, global_avg_loss = self._compute_avg_losses(accumulated_loss)
                    accumulated_loss = 0.0

                    self._log_training_step(
                        global_step,
                        device_avg_loss,
                        global_avg_loss,
                        lr_scheduler.get_last_lr()[0]
                    )

                    if (global_step + 1) % self._config.checkpoint_every_n_steps == 0:
                        self._save_checkpoint(epoch, global_step)

                    global_step += 1

            if self._accelerator.is_main_process:
                logger.info("Saving checkpoint at the end of an epoch")

            self._save_checkpoint(epoch, global_step)

    def _prepare_model_optimizer_dataloader_and_scheduler(
            self
    ) -> tuple[torch.nn.Module, AcceleratedOptimizer, DataLoaderAdapter, AcceleratedScheduler]:
        self._model.gradient_checkpointing_enable()

        dataloader = DataLoader(
            self._dataset,
            batch_size=self._config.per_device_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=partial(
                pad_batch,
                tokenizer=self._tokenizer,
                padding_strategy=self._config.padding_configuration.padding_strategy,
                pad_to_multiple_of=self._config.padding_configuration.padding_multiple,
                label_mask=TokenizationConstants.LABEL_MASK,
                training_mode=self._config.training_mode,
                padding_side=self._config.padding_configuration.padding_side.lower()
            )
        )
        optimizer = AdamW(
            self._model.parameters(),
            lr=self._config.learning_rate,
            betas=(
                self._config.optimizer_configuration.beta_1,
                self._config.optimizer_configuration.beta_2
            ),
            eps=self._config.optimizer_configuration.epsilon,
            weight_decay=self._config.optimizer_configuration.weight_decay
        )

        total_batch_size = (
            self._config.per_device_batch_size * self._accelerator.num_processes
            * self._accelerator.gradient_accumulation_steps
        )
        num_steps_per_epoch = math.ceil(len(self._dataset) / total_batch_size)
        num_total_steps = num_steps_per_epoch * self._config.epochs
        num_warmup_steps = int(self._config.warmup_ratio * num_total_steps)

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_total_steps
        )

        if self._accelerator.is_main_process:
            logger.info(f"Total batch size: {total_batch_size}", )
            logger.info(f"Total steps: {num_total_steps}")
            logger.info(f"Steps per epoch: {num_steps_per_epoch}")
            logger.info(f"Warmup steps: {num_warmup_steps}")
            logger.info(f"Gradient accumulation steps: {self._accelerator.gradient_accumulation_steps}")

        return self._accelerator.prepare(
            self._model, optimizer, dataloader, lr_scheduler
        )

    def _resume_from_checkpoint(self, len_dataloader: int) -> tuple[int, int, int]:
        last_global_step = self._load_accelerate_states()
        resume_global_step = last_global_step + 1

        resume_epoch = resume_global_step * self._accelerator.gradient_accumulation_steps // len_dataloader
        resume_batch = resume_global_step * self._accelerator.gradient_accumulation_steps % len_dataloader

        if self._accelerator.is_main_process:
            logger.info(f"Loaded checkpoint saved at epoch: {resume_epoch}, batch: {resume_batch}")

        return resume_global_step, resume_epoch, resume_batch

    def _load_accelerate_states(self) -> int:
        if self._accelerator.is_main_process:
            logger.info(f"Loading accelerate states from {self._config.resume_from_checkpoint}...")

        self._accelerator.load_state(self._config.resume_from_checkpoint)

        metadata_path = self._config.resume_from_checkpoint / self.METADATA_FILENAME

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return metadata["last_global_step"]

    def _compute_avg_losses(self, accumulated_loss: float) -> tuple[float, float]:
        device_avg_loss = accumulated_loss / self._accelerator.gradient_accumulation_steps
        global_avg_loss = device_avg_loss

        if self._accelerator.num_processes > 1:
            avg_loss_tensor = torch.tensor(device_avg_loss, device=self._accelerator.device)
            global_avg_loss = self._accelerator.gather(avg_loss_tensor).mean().item()

        return device_avg_loss, global_avg_loss

    def _log_training_step(
            self,global_step: int,
            device_avg_loss: float,
            avg_loss: float,
            learning_rate: float
    ) -> None:
        if self._accelerator.is_main_process:
            perplexity = np.exp(avg_loss).item()
            self._writer.add_scalar(
                f"train/loss/avg-loss",
                avg_loss,
                global_step
            )
            self._writer.add_scalar(
                f"train/perplexity/avg-perplexity",
                perplexity,
                global_step
            )
            logger.info(f"step: {global_step} - loss: {avg_loss} - perplexity: {perplexity}")

        self._writer.add_scalar(
            f"train/loss/gpu-{self._accelerator.process_index}",
            device_avg_loss,
            global_step
        )

        self._writer.add_scalar(
            f"train/perplexity/gpu-{self._accelerator.process_index}",
            np.exp(device_avg_loss).item(),
            global_step
        )

        self._writer.add_scalar(
            f"train/learning-rate/gpu-{self._accelerator.process_index}",
            learning_rate,
            global_step
        )

    def _save_checkpoint(self, epoch: int, global_step: int) -> None:
        self._accelerator.wait_for_everyone()
        self._save_model_weights_and_tokenizer(epoch, global_step)
        self._save_configuration_files(epoch, global_step)

        if self._config.save_all_states:
            self._save_accelerate_states(epoch, global_step)

        if isinstance(self._config.output_dir.fs, gcsfs.core.GCSFileSystem) and self._accelerator.is_main_process:
            logger.info("Copying checkpoint to GCS")
            copy_local_dir_to_gcp_dir(
                self._get_checkpoint_base_path(epoch, global_step),
                self._config.output_dir / self.CHECKPOINT_DIR
            )
            remove_local_dir(self._get_checkpoint_base_path(epoch, global_step))

    def _save_configuration_files(self, epoch: int, global_step: int) -> None:
        if not self._accelerator.is_main_process:
            return

        checkpoint_base_path = self._get_checkpoint_base_path(epoch, global_step)
        checkpoint_path = checkpoint_base_path / "config_files"

        logger.info(f"Saving configuration files at {checkpoint_path}")

        save_configuration_to_yaml(
            filepath=checkpoint_path / "main_config.yaml",
            config_dict=self._config.as_dict()
        )

        with UPath("configs/llm_training/accelerate/default_config.yaml").open("r") as accelerate_config_file:
            with UPath.open(checkpoint_path / "accelerate_config.yaml", "w") as output_file:
                output_file.write(accelerate_config_file.read())

    def _save_model_weights_and_tokenizer(self, epoch: int, global_step: int) -> None:
        if not self._accelerator.is_main_process:
            return

        checkpoint_base_path = self._get_checkpoint_base_path(epoch, global_step)
        checkpoint_path = checkpoint_base_path / "model_weights_and_tokenizer"

        logger.info(f"Saving model weights and tokenizer at {checkpoint_path}")

        unwrapped_model = self._accelerator.unwrap_model(self._model)
        unwrapped_model.save_pretrained(
            str(checkpoint_path / "model"),
            is_main_process=self._accelerator.is_main_process,
            save_function=self._accelerator.save,
            state_dict=self._accelerator.get_state_dict(self._model),
            max_shard_size="100GB"
        )
        self._tokenizer.save_pretrained(str(checkpoint_path / "tokenizer"))

    def _save_accelerate_states(self, epoch: int, global_step: int) -> None:
        checkpoint_base_path = self._get_checkpoint_base_path(epoch, global_step)
        checkpoint_path = checkpoint_base_path / "accelerate_states"

        logger.info(f"Saving accelerate states at {checkpoint_path}")

        self._accelerator.save_state(str(checkpoint_path))

        if self._accelerator.is_main_process:
            metadata = {"last_global_step": global_step}

            with UPath.open(checkpoint_path / self.METADATA_FILENAME, "w") as file_handler:
                json.dump(metadata, file_handler)


    def _get_checkpoint_base_path(self, epoch: int, step: int) -> UPath:
        if isinstance(self._config.output_dir.fs, gcsfs.core.GCSFileSystem):
            output_dir = UPath(f"/tmp/pretraining_output/{self._config.output_dir.name}")
        else:
            output_dir = self._config.output_dir

        return output_dir / self.CHECKPOINT_DIR / f"epoch-{epoch}-step-{step}"
