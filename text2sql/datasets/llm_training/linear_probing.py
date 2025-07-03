from transformers import PreTrainedTokenizer

from text2sql.datasets.domain.model import PretrainDataSample
from text2sql.datasets.llm_training.base_dataset import BaseDataset
from text2sql.datasets.llm_training.domain import LinearProbingInputSample, TokenizedLinearProbingDataSample
from text2sql.llm_training.domain.enums import ModelExecutionModes


class LinearProbingDataset(BaseDataset[TokenizedLinearProbingDataSample]):
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
    ) -> list[TokenizedLinearProbingDataSample]:
        tokenization_input = self._prepare_tokenization_samples(input_data)

        tokenized_text = self._tokenizer([sample.input_text for sample in tokenization_input])

        tokenized_dataset = []
        for sample, input_ids in zip(tokenization_input, tokenized_text["input_ids"]):
            input_ids_with_special_tokens = [self._tokenizer.bos_token_id] + input_ids
            extended_attention_mask = [1] * len(input_ids_with_special_tokens)

            tokenized_dataset.append(
                TokenizedLinearProbingDataSample(
                    question_id=sample.question_id,
                    tokenized_text=input_ids_with_special_tokens,
                    attention_mask=extended_attention_mask,
                    label=sample.label
                )
            )

        return tokenized_dataset

    def _prepare_linear_probing_input_and_labels(self, reasoning_steps: list[str]) -> list[tuple[str, int]]:

        LABEL_WRONG = 1
        LABEL_CORRECT = 0

        output_list = []

        for item in reasoning_steps:
            if '[BACK]' in item:
                parts = item.split(' [BACK] ')
                wrong_part = parts[0].strip()
                correct_part = parts[1].strip()

                output_list.append((wrong_part, LABEL_WRONG))
                output_list.append((correct_part, LABEL_CORRECT))
            else:
                output_list.append((item.strip(), LABEL_CORRECT))

        final_output = []
        current_context = ""

        index = 0
        for text, status in output_list:
            if status == LABEL_WRONG and index == 0:
                final_output.append((text, LABEL_WRONG))
            elif status == LABEL_WRONG and index > 0:
                last_correct_context = final_output[index - 1][0]
                final_output.append((last_correct_context + " " + text, LABEL_WRONG))
            else:
                if current_context:
                    current_context += " " + text
                else:
                    current_context = text
                final_output.append((current_context, LABEL_CORRECT))

            index += 1

        return final_output


    def _prepare_tokenization_samples(self, input_data: list[PretrainDataSample]) -> list[LinearProbingInputSample]:

        tokenization_samples = []
        for sample in input_data:
            question_id = str(sample.question_id)
            prefix = f"[CONTEXT] {sample.prompt}[QUESTION] {sample.question}\n[REASONING]"

            reasoning_steps = sample.reasoning_steps
            linear_probing_examples = self._prepare_linear_probing_input_and_labels(reasoning_steps)

            for example in linear_probing_examples:
                input_text = f"{prefix} {example[0]}"
                tokenization_samples.append(
                    LinearProbingInputSample(
                        question_id=question_id,
                        input_text=input_text,
                        label=example[1]
                    )
                )

        return tokenization_samples


    def __getitem__(self, index: int) -> dict[str, list[int]]:
        input_ids = self.tokenized_data[index].tokenized_text
        attention_mask = self.tokenized_data[index].attention_mask
        labels = self.tokenized_data[index].label

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def __len__(self) -> int:
        return len(self.tokenized_data)
