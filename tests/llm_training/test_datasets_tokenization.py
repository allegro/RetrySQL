from upath import UPath
from functools import partial

import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils import PaddingStrategy
from torch.utils.data import DataLoader

from text2sql.llm_training.domain.constants import TokenizationConstants
from text2sql.llm_training.domain.enums import SupportedTrainingModes, ModelExecutionModes
from text2sql.datasets.llm_training.pretraining_dataset import PretrainingDataset
from text2sql.datasets.llm_training.instruction_fine_tuning_dataset import InstructionFineTuningDataset
from text2sql.datasets.llm_training.batching import pad_batch
from text2sql.llm_training.main import _load_pretrain_data


class TestDatasetsTokenization:

    def setup_method(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.add_special_tokens({
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>"
        })
        self.tokenizer.bos_token_id = self.tokenizer.convert_tokens_to_ids("<bos>")
        self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids("<eos>")
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<pad>")

        self.input_data = _load_pretrain_data(UPath("tests/resources/llm_training/pretraining_data_example.jsonl"))

    def test_pretraining_dataset_tokenization_for_training(self) -> None:
        # GIVEN
        mode = ModelExecutionModes.TRAIN
        pretraining_dataset = PretrainingDataset(
            input_data=self.input_data,
            tokenizer=self.tokenizer,
            mode=mode
        )

        # THEN
        assert len(pretraining_dataset) == 3
        assert pretraining_dataset[0] == {
            "input_ids": [self.tokenizer.bos_token_id, 1212, 318, 262, 717, 6152, 770, 318, 262, 717, 12405, self.tokenizer.eos_token_id],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        assert pretraining_dataset[1] == {
            "input_ids": [self.tokenizer.bos_token_id, 1212, 318, 262, 1218, 6152, 770, 318, 262, 1218, 12405, self.tokenizer.eos_token_id],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        assert pretraining_dataset[2] == {
            "input_ids": [self.tokenizer.bos_token_id, 1212, 318, 262, 2368, 6152, 770, 318, 262, 2368, 12405, self.tokenizer.eos_token_id],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        }

    def test_pretraining_dataset_tokenization_for_evaluation(self) -> None:
        # GIVEN
        mode = ModelExecutionModes.EVAL
        pretraining_dataset = PretrainingDataset(
            input_data=self.input_data,
            tokenizer=self.tokenizer,
            mode=mode
        )

        # THEN
        assert len(pretraining_dataset) == 3
        assert pretraining_dataset[0] == {
            "input_ids": [self.tokenizer.bos_token_id, 1212, 318, 262, 717, 6152, 198, 46506, 220],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        assert pretraining_dataset[1] == {
            "input_ids": [self.tokenizer.bos_token_id, 1212, 318, 262, 1218, 6152, 198, 46506, 220],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1]
        }
        assert pretraining_dataset[2] == {
            "input_ids": [self.tokenizer.bos_token_id, 1212, 318, 262, 2368, 6152, 198, 46506, 220],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1]
        }

    def test_instruction_fine_tuning_dataset_tokenization_for_training(self) -> None:
        # GIVEN
        mode = ModelExecutionModes.TRAIN
        instruction_finetuning_dataset = InstructionFineTuningDataset(
            input_data=self.input_data,
            tokenizer=self.tokenizer,
            mode=mode
        )

        # THEN
        assert len(instruction_finetuning_dataset) == 3
        assert instruction_finetuning_dataset.data_with_chat_template_applied[0] == '<|im_start|>user\nThis is the first prompt<|im_end|>\n<|im_start|>assistant\nThis is the first query\n<|im_end|>'
        assert instruction_finetuning_dataset.data_with_chat_template_applied[1] == '<|im_start|>user\nThis is the second prompt<|im_end|>\n<|im_start|>assistant\nThis is the second query\n<|im_end|>'
        assert instruction_finetuning_dataset.data_with_chat_template_applied[2] == '<|im_start|>user\nThis is the third prompt<|im_end|>\n<|im_start|>assistant\nThis is the third query\n<|im_end|>'
        assert instruction_finetuning_dataset[0] == {
            "input_ids": [27, 91, 320, 62, 9688, 91, 29, 7220, 198, 1212, 318, 262, 717, 6152, 27, 91, 320, 62, 437, 91,
                          29, 198, 27, 91, 320, 62, 9688, 91, 29, 562, 10167, 198,
                          1212, 318, 262, 717, 12405, 198, 27, 91, 320, 62, 437, 91, 29],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "labels": [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                           -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                           1212, 318, 262, 717, 12405, 198, 27, 91, 320, 62, 437, 91, 29]
        }
        assert instruction_finetuning_dataset[1] == {
            "input_ids": [27, 91, 320, 62, 9688, 91, 29, 7220, 198, 1212, 318, 262, 1218, 6152, 27, 91, 320, 62, 437, 91,
                          29, 198, 27, 91, 320, 62, 9688, 91, 29, 562, 10167, 198,
                          1212, 318, 262, 1218, 12405, 198, 27, 91, 320, 62, 437, 91, 29],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "labels": [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                           -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                           1212, 318, 262, 1218, 12405, 198, 27, 91, 320, 62, 437, 91, 29]
        }
        assert instruction_finetuning_dataset[2] == {
            "input_ids": [27, 91, 320, 62, 9688, 91, 29, 7220, 198, 1212, 318, 262, 2368, 6152, 27, 91, 320, 62, 437, 91,
                          29, 198, 27, 91, 320, 62, 9688, 91, 29, 562, 10167, 198,
                          1212, 318, 262, 2368, 12405, 198, 27, 91, 320, 62, 437, 91, 29],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "labels": [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                           -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                           1212, 318, 262, 2368, 12405, 198, 27, 91, 320, 62, 437, 91, 29]
        }

    def test_instruction_fine_tuning_dataset_tokenization_for_evaluation(self) -> None:
        # GIVEN
        mode = ModelExecutionModes.EVAL
        instruction_finetuning_dataset = InstructionFineTuningDataset(
            input_data=self.input_data,
            tokenizer=self.tokenizer,
            mode=mode
        )

        # THEN
        assert len(instruction_finetuning_dataset) == 3
        assert instruction_finetuning_dataset.data_with_chat_template_applied[0] == '<|im_start|>user\nThis is the first prompt<|im_end|>\n<|im_start|>assistant\nSELECT '
        assert instruction_finetuning_dataset.data_with_chat_template_applied[1] == '<|im_start|>user\nThis is the second prompt<|im_end|>\n<|im_start|>assistant\nSELECT '
        assert instruction_finetuning_dataset.data_with_chat_template_applied[2] == '<|im_start|>user\nThis is the third prompt<|im_end|>\n<|im_start|>assistant\nSELECT '
        assert instruction_finetuning_dataset[0] == {
            'input_ids': [27, 91, 320, 62, 9688, 91, 29, 7220, 198, 1212, 318, 262, 717, 6152, 27, 91, 320, 62, 437, 91, 29, 198, 27, 91, 320, 62, 9688, 91, 29, 562, 10167, 198, 46506, 220],
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'labels': None
        }
        assert instruction_finetuning_dataset[1] == {
            'input_ids': [27, 91, 320, 62, 9688, 91, 29, 7220, 198, 1212, 318, 262, 1218, 6152, 27, 91, 320, 62, 437, 91, 29, 198, 27, 91, 320, 62, 9688, 91, 29, 562, 10167, 198, 46506, 220],
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'labels': None
        }

        assert instruction_finetuning_dataset[2] == {
            'input_ids': [27, 91, 320, 62, 9688, 91, 29, 7220, 198, 1212, 318, 262, 2368, 6152, 27, 91, 320, 62, 437, 91, 29, 198, 27, 91, 320, 62, 9688, 91, 29, 562, 10167, 198, 46506, 220],
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'labels': None
        }


    def test_pretraining_dataset_padding(self) -> None:
        # GIVEN
        mode = ModelExecutionModes.TRAIN
        pretraining_dataset = PretrainingDataset(
            input_data=self.input_data,
            tokenizer=self.tokenizer,
            mode=mode
        )

        # AND
        dataloader = DataLoader(
            pretraining_dataset,
            batch_size=3,
            shuffle=False,
            drop_last=False,
            collate_fn=partial(
                pad_batch,
                tokenizer=self.tokenizer,
                padding_strategy=PaddingStrategy.LONGEST,
                pad_to_multiple_of=8,
                label_mask=-100,
                training_mode=SupportedTrainingModes.PRETRAINING
            )
        )

        # WHEN
        expected_input_ids = torch.tensor(
                    [[self.tokenizer.bos_token_id,  1212,   318,   262,   717,  6152,   770,   318,   262,   717, 12405, self.tokenizer.eos_token_id,
                      self.tokenizer.pad_token_id, self.tokenizer.pad_token_id, self.tokenizer.pad_token_id, self.tokenizer.pad_token_id],
                     [self.tokenizer.bos_token_id,  1212,   318,   262,  1218,  6152,   770,   318,   262,  1218, 12405, self.tokenizer.eos_token_id,
                      self.tokenizer.pad_token_id, self.tokenizer.pad_token_id, self.tokenizer.pad_token_id, self.tokenizer.pad_token_id],
                     [self.tokenizer.bos_token_id,  1212,   318,   262,  2368,  6152,   770,   318,   262,  2368, 12405, self.tokenizer.eos_token_id,
                      self.tokenizer.pad_token_id, self.tokenizer.pad_token_id, self.tokenizer.pad_token_id, self.tokenizer.pad_token_id]
                     ]

        )
        expected_attention_mask = torch.tensor(
                    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]
        )
        expected_labels = torch.tensor(
                    [[self.tokenizer.bos_token_id,  1212,   318,   262,   717,  6152,   770,   318,   262,   717, 12405, self.tokenizer.eos_token_id, -100, -100, -100, -100],
                     [self.tokenizer.bos_token_id,  1212,   318,   262,  1218,  6152,   770,   318,   262,  1218, 12405, self.tokenizer.eos_token_id, -100, -100, -100, -100],
                     [self.tokenizer.bos_token_id,  1212,   318,   262,  2368,  6152,   770,   318,   262,  2368, 12405, self.tokenizer.eos_token_id, -100, -100, -100, -100]]
        )


        for idx, batch in enumerate(dataloader):
            assert idx == 0
            assert torch.equal(batch["input_ids"], expected_input_ids)
            assert torch.equal(batch["attention_mask"], expected_attention_mask)
            assert torch.equal(batch["labels"], expected_labels)

    def test_instruction_fine_tuning_dataset_padding(self) -> None:
        # GIVEN
        mode = ModelExecutionModes.TRAIN
        instruction_finetuning_dataset = InstructionFineTuningDataset(
            input_data=self.input_data,
            tokenizer=self.tokenizer,
            mode=mode
        )

        # AND
        dataloader = DataLoader(
            instruction_finetuning_dataset,
            batch_size=3,
            shuffle=False,
            drop_last=False,
            collate_fn=partial(
                pad_batch,
                tokenizer=self.tokenizer,
                padding_strategy=PaddingStrategy.LONGEST,
                pad_to_multiple_of=2,
                label_mask=TokenizationConstants.LABEL_MASK,
                training_mode=SupportedTrainingModes.INSTRUCTION_FINE_TUNING
            )
        )

        # WHEN
        expected_input_ids = torch.tensor(
            # ------------------------------ example 1 ------------------------------
            # <|im_start|>user\n<PROMPT><|im_end|>\n tokens
            [[27,    91,   320,    62,  9688,    91,    29,  7220,   198,  1212, 318,   262,   717,  6152,    27, 91,
                 320,    62,   437,    91, 29,   198,    27,    91,   320,    62,  9688,    91,    29,   562, 10167, 198,
            # <|im_start|>assistant\n<QUERY><|im_end|> tokens
                 1212,   318,   262,   717, 12405,   198,    27,    91, 320,    62,   437,    91,    29,
            # <PAD>
              50259
              ],
            # -----------------------------------------------------------------------
            # ------------------------------ example 2 ------------------------------
            # <|im_start|>user\n<PROMPT><|im_end|>\n tokens
             [27,    91,   320,    62,  9688,    91,    29,  7220,   198,  1212, 318,   262,  1218,  6152,    27,    91,
                 320,    62,   437,    91, 29,   198,    27,    91,   320,    62,  9688,    91,    29,   562, 10167,   198,
            # <|im_start|>assistant\n<QUERY><|im_end|> tokens
                 1212,   318,   262,  1218, 12405,   198,    27,    91, 320,    62,   437,    91,    29,
            # <PAD>
              50259
              ],
            # -----------------------------------------------------------------------
            # ------------------------------ example 3 ------------------------------
            # <|im_start|>user\n<PROMPT><|im_end|>\n tokens
             [27,    91,   320,    62,  9688,    91,    29,  7220,   198,  1212, 318,   262,  2368,  6152,    27,    91,
                 320,    62,   437,    91, 29,   198,    27,    91,   320,    62,  9688,    91,    29,   562, 10167,   198,
            # <|im_start|>assistant\n<QUERY><|im_end|> tokens
                 1212,   318,   262,  2368, 12405,   198,    27,    91, 320,    62,   437,    91,    29,
            # <PAD>
              50259]
             ]
            # ----------------------------------------------------------------------
        )
        # All ones everywhere, apart from "0" for padding tokens
        expected_attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]
        )

        expected_labels = torch.tensor(
            # ------------------------------ example 1 ------------------------------
            # <|im_start|>user\n<PROMPT><|im_end|>\n tokens set to -100
            [[ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, -100,  -100,  -100,  -100,  -100, -100,
               -100,  -100,  -100,  -100, -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, -100,  -100,
            # <|im_start|>assistant\n<QUERY><|im_end|> tokens left unchanged
               1212,   318,   262,   717, 12405,   198,    27,    91, 320,    62,   437,    91,    29,
            # <PAD>
               -100
               ],
            # -----------------------------------------------------------------------
            # ------------------------------ example 2 ------------------------------
            # <|im_start|>user\n<PROMPT><|im_end|>\n tokens set to -100
             [ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, -100,  -100,  -100,  -100,  -100,  -100,  -100,
               -100,  -100,  -100, -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, -100,  -100,
            # <|im_start|>assistant\n<QUERY><|im_end|> tokens left unchanged
               1212,   318,   262,  1218, 12405,   198,    27,    91, 320,    62,   437,    91,    29,
            # <PAD>
               -100
               ],
             # -----------------------------------------------------------------------
             # ------------------------------ example 3 ------------------------------
             # <pad> and <|im_start|>user\n<PROMPT><|im_end|>\n tokens set to -100
             [ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, -100,  -100,  -100,  -100,  -100,  -100,  -100,
               -100,  -100,  -100, -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, -100,  -100,
            # <|im_start|>assistant\n<QUERY><|im_end|> tokens left unchanged
               1212,   318,   262,  2368, 12405,   198,    27,    91, 320,    62,   437,    91,    29,
            # <PAD>
               -100
               ]
             ]
        )

        for idx, batch in enumerate(dataloader):
            assert idx == 0
            assert torch.equal(batch["input_ids"], expected_input_ids)
            assert torch.equal(batch["attention_mask"], expected_attention_mask)
            assert torch.equal(batch["labels"], expected_labels)
