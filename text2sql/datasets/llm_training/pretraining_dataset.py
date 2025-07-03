from transformers import PreTrainedTokenizer

from text2sql.datasets.domain.model import PretrainDataSample
from text2sql.datasets.llm_training.base_dataset import BaseDataset
from text2sql.datasets.llm_training.domain import TokenizationInputSample, TokenizedPretrainDataSample
from text2sql.llm_training.domain.enums import ModelExecutionModes


class PretrainingDataset(BaseDataset[TokenizedPretrainDataSample]):
    def __init__(
        self,
            mode: ModelExecutionModes,
            tokenizer: PreTrainedTokenizer,
            input_data: list[PretrainDataSample]
    ) -> None:
        super().__init__(input_data=input_data, tokenizer=tokenizer, mode=mode)

    def _tokenize_dataset(
            self,
            input_data: list[PretrainDataSample]
    ) -> list[TokenizedPretrainDataSample]:
        tokenization_input = self._prepare_tokenization_samples(input_data)

        tokenized_text = self._tokenizer([sample.input_text for sample in tokenization_input])

        tokenized_dataset = []
        for sample, input_ids in zip(tokenization_input, tokenized_text["input_ids"]):
            # Training mode: <bos> input_text <eos>
            # Eval mode: <bos> input_text
            if self._mode == ModelExecutionModes.TRAIN:
                input_ids_with_special_tokens = [self._tokenizer.bos_token_id] + input_ids + [self._tokenizer.eos_token_id]
            elif self._mode == ModelExecutionModes.EVAL:
                input_ids_with_special_tokens = [self._tokenizer.bos_token_id] + input_ids
            else:
                raise ValueError(f"Unsupported mode: {self._mode}")

            extended_attention_mask = [1] * len(input_ids_with_special_tokens)

            tokenized_dataset.append(
                TokenizedPretrainDataSample(
                    question_id=sample.question_id,
                    tokenized_text=input_ids_with_special_tokens,
                    attention_mask=extended_attention_mask
                )
            )

        return tokenized_dataset

    def _prepare_tokenization_samples(self, input_data: list[PretrainDataSample]) -> list[TokenizationInputSample]:
        return [
            TokenizationInputSample(
                question_id=str(sample.question_id),
                input_text=(
                    self._prepare_train_input_text(sample) if self._mode == ModelExecutionModes.TRAIN
                    else self._prepare_eval_input_text(sample)
                )
            )
            for sample in input_data
        ]

    @staticmethod
    def _prepare_train_input_text(input_sample: PretrainDataSample) -> str:
        if input_sample.question is not None:
            reasoning_steps_joined = " ".join(input_sample.reasoning_steps)
            return f"[CONTEXT] {input_sample.prompt}[QUESTION] {input_sample.question}\n[REASONING] {reasoning_steps_joined} [SQL] {input_sample.query}"

        return f"{input_sample.prompt} {input_sample.query}"

    @staticmethod
    def _prepare_eval_input_text(input_sample: PretrainDataSample) -> str:
        if input_sample.question is not None:
            return f"[CONTEXT] {input_sample.prompt}[QUESTION] {input_sample.question}\n[REASONING] "

        return f"{input_sample.prompt}\nSELECT "

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        input_ids = self.tokenized_data[index].tokenized_text
        attention_mask = self.tokenized_data[index].attention_mask

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def __len__(self) -> int:
        return len(self.tokenized_data)
