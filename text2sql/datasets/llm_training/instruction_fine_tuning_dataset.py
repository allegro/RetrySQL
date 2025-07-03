from copy import deepcopy

from transformers import PreTrainedTokenizer

from text2sql.datasets.domain.model import PretrainDataSample
from text2sql.datasets.llm_training.base_dataset import BaseDataset
from text2sql.datasets.llm_training.domain import InstructionFineTuningDataSample, \
    TokenizedInstructionFineTuningDataSample
from text2sql.llm_training.domain.constants import TokenizationConstants
from text2sql.llm_training.domain.enums import ModelExecutionModes


class InstructionFineTuningDataset(BaseDataset[TokenizedInstructionFineTuningDataSample]):
    def __init__(
            self,
            mode: ModelExecutionModes,
            tokenizer: PreTrainedTokenizer,
            input_data: list[PretrainDataSample]
    ) -> None:
        super().__init__(input_data=input_data, tokenizer=tokenizer, mode=mode)

    def _setup_tokenizer_chat_template(self) -> None:
        # Source: https://huggingface.co/docs/transformers/main/en/chat_templating#advanced-adding-and-editing-chat-templates
        # We need to overwrite the chat template to remove `You are OpenCoder, created by OpenCoder Team...` section from the authors' chat template
        self._tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if loop.first and messages[0]['role'] != 'system' %}"
            "{% endif %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )

    def _build_instruction_prompt(self, prompt: str):
        prompt_message_only = [{"role": "user", "content": prompt}]
        return self._tokenizer.apply_chat_template(prompt_message_only, tokenize=False, add_generation_prompt=True)

    def _tokenize_fn(self, sequences: list[str]) -> dict[str, list[list[int]]]:
        tokenized_list = self._tokenizer([text for text in sequences])
        return {"input_ids": tokenized_list.input_ids}

    def _tokenize_for_evaluation(
            self,
            tokenization_input: list[InstructionFineTuningDataSample]
    ) -> list[TokenizedInstructionFineTuningDataSample]:

        instruction_prompt_with_answer_beginning_added = [
            self._build_instruction_prompt(example.instruction) for example in tokenization_input
        ]
        instruction_prompt_with_answer_beginning_added = [
            f"{instruction_prompt}{example.expected_answer}"
            for instruction_prompt, example in zip(instruction_prompt_with_answer_beginning_added, tokenization_input)
        ]
        self.data_with_chat_template_applied = instruction_prompt_with_answer_beginning_added

        tokenized_instructions = self._tokenize_fn(self.data_with_chat_template_applied)
        return [
            TokenizedInstructionFineTuningDataSample(
                question_id=sample.question_id,
                input_ids=input_ids,
                attention_mask=[1] * len(input_ids)
            )
            for sample, input_ids in zip(tokenization_input, tokenized_instructions["input_ids"])
        ]

    @staticmethod
    def _create_target_ids(
            full_examples_tokenized: dict[str, list[list[int]]],
            sources_only_tokenized: dict[str, list[list[int]]]
    ) -> list[list[int]]:
        all_input_ids = full_examples_tokenized["input_ids"]
        all_target_ids = deepcopy(all_input_ids)
        for sequence_id, target_ids in enumerate(all_target_ids):
            source_len = len(sources_only_tokenized["input_ids"][sequence_id])
            target_ids[:source_len] = [TokenizationConstants.LABEL_MASK] * source_len
        return all_target_ids

    def _tokenize_for_training(
            self,
            tokenization_input: list[InstructionFineTuningDataSample]
    ) -> list[TokenizedInstructionFineTuningDataSample]:

        sources = [self._build_instruction_prompt(example.instruction) for example in tokenization_input]
        targets = [f"{example.expected_answer}\n{TokenizationConstants.EOT_TOKEN}" for example in tokenization_input]
        self.data_with_chat_template_applied = [source + target for source, target in zip(sources, targets)]

        full_examples_tokenized = self._tokenize_fn(self.data_with_chat_template_applied)
        sources_only_tokenized = self._tokenize_fn(sources)

        all_input_ids = full_examples_tokenized["input_ids"]
        all_target_ids = self._create_target_ids(full_examples_tokenized, sources_only_tokenized)

        return [
            TokenizedInstructionFineTuningDataSample(
                question_id=sample.question_id,
                input_ids=input_ids,
                attention_mask=[1] * len(input_ids),
                target_ids=target_ids
            )
            for sample, input_ids, target_ids in zip(tokenization_input, all_input_ids, all_target_ids)
        ]

    def _tokenize_dataset(
            self,
            input_data: list[PretrainDataSample]
    ) -> list[TokenizedInstructionFineTuningDataSample]:
        self._setup_tokenizer_chat_template()
        tokenization_input = self._prepare_instruction_samples(input_data)

        # INPUT: <|im_start|>user\nprompt<|im_end|>\n<|im_start|>assistant\n
        # TARGET: no targets
        if self._mode == ModelExecutionModes.EVAL:
            return self._tokenize_for_evaluation(tokenization_input)

        # INPUT: <|im_start|>user\nprompt<|im_end|>\n<|im_start|>assistant\nexpected_answer<|im_end|>
        # TARGET: <-100, ..., -100>expected_answer<|im_end|>
        elif self._mode == ModelExecutionModes.TRAIN:
            return self._tokenize_for_training(tokenization_input)
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

    def _prepare_instruction_samples(
            self,
            input_data: list[PretrainDataSample]
    ) -> list[InstructionFineTuningDataSample]:
        instruction_samples = []
        for sample in input_data:
            instruction, expected_answer = (
                self._prepare_train_instruction_and_expected_answer(sample) if self._mode == ModelExecutionModes.TRAIN
                else self._prepare_eval_instruction_and_expected_answer(sample)
            )
            instruction_samples.append(
                InstructionFineTuningDataSample(
                    question_id=str(sample.question_id),
                    instruction=instruction,
                    expected_answer=expected_answer
                )
            )
        return instruction_samples

    @staticmethod
    def _prepare_train_instruction_and_expected_answer(input_sample: PretrainDataSample) -> tuple[str, str]:
        if input_sample.question is not None:
            reasoning_steps_joined = " ".join(input_sample.reasoning_steps)
            instruction = f"[CONTEXT] {input_sample.prompt}[QUESTION] {input_sample.question}"
            expected_answer = f"[REASONING] {reasoning_steps_joined} [SQL] {input_sample.query}"
        else:
            instruction = input_sample.prompt
            expected_answer = input_sample.query

        return instruction, expected_answer

    @staticmethod
    def _prepare_eval_instruction_and_expected_answer(input_sample: PretrainDataSample) -> tuple[str, str]:
        if input_sample.question is not None:
            instruction = f"[CONTEXT] {input_sample.prompt}[QUESTION] {input_sample.question}"
            expected_answer = f"[REASONING] "
        else:
            instruction = input_sample.prompt
            expected_answer = "SELECT "

        return instruction, expected_answer


    def __getitem__(self, index: int) -> dict[str, list[int]]:
        input_ids = self.tokenized_data[index].input_ids
        attention_mask = self.tokenized_data[index].attention_mask
        target_ids = self.tokenized_data[index].target_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target_ids
        }

    def __len__(self) -> int:
        return len(self.tokenized_data)