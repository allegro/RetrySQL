import torch
from allms.domain.response import ResponseData
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from dataclasses import asdict

from text2sql.commons.constants import BACK_TOKEN
from text2sql.commons.io_utils import save_objects_to_jsonl_file
from text2sql.commons.logging_utils import get_logger
from text2sql.datasets.domain.model import PretrainDataSample
from text2sql.datasets.domain.model import Text2SqlDataSample
from text2sql.datasets.llm_training.instruction_fine_tuning_dataset import InstructionFineTuningDataset
from text2sql.datasets.llm_training.pretraining_dataset import PretrainingDataset
from text2sql.llm_training.domain.constants import TokenizationConstants
from text2sql.llm_training.domain.enums import SupportedTrainingModes, ModelExecutionModes, PretrainingDataTypes
from text2sql.llms.configuration import OpenSourceLlmConfiguration
from text2sql.llms.domain.models import OpenSourceLlm
from text2sql.modules.generation.domain import InferenceBatch, InferenceExample, ConfidenceScore, ConfidenceOutput, ConfidenceScorePair
from text2sql.modules.generation.llm_base_generator import LlmBaseGenerator
from text2sql.modules.llm_input.prompt_templates.sql_generation_prompt_templates import BaselineSqlPromptTemplate

logger = get_logger(__name__)


class OpenSourceLlmSqlGenerator(LlmBaseGenerator):
    TOKEN_RADIUS = 10

    def __init__(
        self,
        llm: OpenSourceLlm,
        llm_config: OpenSourceLlmConfiguration,
        sql_generation_prompt_template: BaselineSqlPromptTemplate
    ) -> None:
        super().__init__(llm=llm)
        self._model: PreTrainedModel = llm.llm
        self._tokenizer: PreTrainedTokenizer = llm.tokenizer
        self._back_token_id = self._tokenizer.convert_tokens_to_ids(BACK_TOKEN)
        self._llm_config = llm_config
        self._setup_model_and_tokenizer_for_evaluation()

        # Currently not used - will be used in the future in case we start supporting batch_size > 1
        self._max_input_length = self._llm_config.model_configuration.max_model_context_length - self._llm_config.generation_configuration.max_new_tokens
        self._stopping_generation_token_id = self._set_stopping_generation_token_id()
        self._sql_generation_prompt_template = sql_generation_prompt_template

        self._tokenized_eval_set_class = (
            PretrainingDataset if self._llm_config.model_configuration.pre_trained_model_mode == SupportedTrainingModes.PRETRAINING
            else InstructionFineTuningDataset
        )

    def _setup_model_and_tokenizer_for_evaluation(self) -> None:
        self._model.eval()

        self._model.config.pad_token_id = self._tokenizer.pad_token_id

        training_mode = self._llm_config.model_configuration.pre_trained_model_mode
        if training_mode == SupportedTrainingModes.PRETRAINING:
            self._model.config.eos_token_id = self._tokenizer.eos_token_id
            self._model.config.bos_token_id = self._tokenizer.bos_token_id
        elif training_mode == SupportedTrainingModes.INSTRUCTION_FINE_TUNING:
            self._model.config.eos_token_id = self._tokenizer.convert_tokens_to_ids(TokenizationConstants.EOT_TOKEN)
            self._model.config.bos_token_id = self._tokenizer.convert_tokens_to_ids(TokenizationConstants.BOT_TOKEN)
        else:
            raise ValueError(f"Unsupported training mode: {training_mode}")

    def _set_stopping_generation_token_id(self) -> int:
        return (
            self._tokenizer.eos_token_id if self._llm_config.model_configuration.pre_trained_model_mode == SupportedTrainingModes.PRETRAINING
            else self._tokenizer.convert_tokens_to_ids(TokenizationConstants.EOT_TOKEN)
        )


    def __call__(self, data: list[Text2SqlDataSample]) -> list[ResponseData]:
        return self.forward(data=data)
    
    def _compute_mean_proba_around_back_tokens(self, sequence, logits):
        confidence_scores = torch.softmax(logits, dim=2)
        max_confidence_scores = torch.max(confidence_scores, dim=2).values
        mean_over_beam_searches = torch.mean(max_confidence_scores, dim=1).tolist()
        std_dev_over_beam_searches = torch.std(max_confidence_scores, dim=1).tolist()
        all_aggregate_scores = list(zip(mean_over_beam_searches, std_dev_over_beam_searches))
        token_proba_pairs = list(zip(sequence, all_aggregate_scores[1:]))

        back_token_indices = []

        for index, token in enumerate(sequence):
            if token == self._back_token_id:
                back_token_indices.append(index)

        all_probas = []

        for back_token_index in back_token_indices:
            before_back_token = token_proba_pairs[back_token_index-self.TOKEN_RADIUS:back_token_index]
            after_back_token = token_proba_pairs[back_token_index+1:back_token_index+self.TOKEN_RADIUS]

            mean_before = 100 * torch.mean(torch.tensor([mean for _, (mean, _) in before_back_token])).item()
            std_before = 100 * torch.mean(torch.tensor([std for _, (_, std) in before_back_token])).item()
            mean_after = 100 * torch.mean(torch.tensor([mean for _, (mean, _) in after_back_token])).item()
            std_after = 100 * torch.mean(torch.tensor([std for _, (_, std) in after_back_token])).item()

            all_probas.append(
                ConfidenceScorePair(
                    before=ConfidenceScore(
                        mean_before,
                        std_before
                    ),
                    after=ConfidenceScore(
                        mean_after,
                        std_after
                    )
                )
            )

        return all_probas

    def forward(self, data: list[Text2SqlDataSample]) -> list[ResponseData]:
        dataloader = self._get_dataloader(data=data)

        generated_sqls = []
        all_outputs = []

        for batch in tqdm(dataloader):
            outputs = self.generate_sql_tokens(batch=batch)
            trimmed_and_reshaped_outputs = self.trim_and_reshape_outputs(batch=batch, outputs=outputs.sequences).cpu()

            probas_for_sequence = self._compute_mean_proba_around_back_tokens(
                trimmed_and_reshaped_outputs.tolist()[0],
                torch.stack(outputs.logits).cpu()
            )

            for batch_id in range(self._llm_config.generation_configuration.batch_size):
                question_id = str(batch.question_ids[batch_id])
                question = batch.questions[batch_id]

                all_outputs.append(asdict(ConfidenceOutput(question_id, probas_for_sequence)))

                generated_sql = self.decode_sql_tokens(trimmed_and_reshaped_outputs[batch_id])
                post_processed_sql = self.postprocess(generated_sql, question)

                if self._llm_config.model_configuration.verbose:
                    logger.info(
                        f"Question ID: {question_id} - Generated SQL: {generated_sql}"
                    )

                generated_sqls.append(ResponseData(
                    input_data={
                        "id": question_id,
                        "input_mappings": {"prompt": batch.prompts[batch_id]}},
                    response={"sql": post_processed_sql}
                ))

        if self._llm_config.output_path is not None:
            save_objects_to_jsonl_file(
                all_outputs,
                self._llm_config.output_path
            )

        return generated_sqls

    def postprocess(
        self,
        generated_sql: str,
        question: str
    ) -> str:
        if self._llm_config.post_processing_configuration.add_select_statement_to_the_generated_sql:
            generated_sql = f"SELECT {generated_sql}"

        if self._llm_config.post_processing_configuration.normalize_generated_sql:
            generated_sql = generated_sql.replace("\t", " ")
            generated_sql = generated_sql.replace("\n", " ")
            generated_sql = generated_sql.replace(r'\"', '"')
            generated_sql = generated_sql.replace("``", "`")
            generated_sql = generated_sql.replace(f"{self._tokenizer.convert_ids_to_tokens(self._tokenizer.eos_token_id)}", "")
            generated_sql = generated_sql.replace(TokenizationConstants.EOT_TOKEN, "")

        if self._llm_config.model_configuration.pretraining_data_type == PretrainingDataTypes.WITH_REASONING:
            generated_sql = generated_sql.split(TokenizationConstants.SQL_TOKEN)[-1].strip()

        if self._llm_config.post_processing_configuration.split_output_at_question:
            generated_sql = generated_sql.split(question)[1]

        return generated_sql.strip()

    def generate_sql_tokens(self, batch: InferenceBatch) -> torch.Tensor:
        return self._model.generate(
            input_ids=batch.input_ids.to(self._model.device),
            attention_mask=batch.attention_mask.to(self._model.device),
            max_new_tokens=self._llm_config.generation_configuration.max_new_tokens,
            temperature=self._llm_config.generation_configuration.temperature,
            top_k=self._llm_config.generation_configuration.top_k,
            top_p=self._llm_config.generation_configuration.top_p,
            repetition_penalty=self._llm_config.generation_configuration.repetition_penalty,
            do_sample=self._llm_config.generation_configuration.do_sample,
            num_beams=self._llm_config.generation_configuration.num_beams,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._stopping_generation_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True
        )

    def decode_sql_tokens(self, sql_tokens: torch.Tensor) -> str:
        return self._tokenizer.decode(
            sql_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )
    
    def trim_and_reshape_outputs(self, batch: InferenceBatch, outputs: torch.Tensor) -> torch.Tensor:
        if self._llm_config.post_processing_configuration.trim_output_from_input_sequence:
            outputs = self._trim_input(
                input_attention_mask=batch.attention_mask,
                outputs=outputs
            )

        return outputs.reshape(
            self._llm_config.generation_configuration.batch_size,
            outputs.shape[1]
        )

    @staticmethod
    def _trim_input(input_attention_mask: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        input_lengths = (input_attention_mask == 1).sum(dim=1)
        return torch.stack([output[length:] for output, length in zip(outputs, input_lengths)])

    def _prepare_inference_dataset(self, data: list[Text2SqlDataSample]) -> list[InferenceExample]:
        is_with_reasoning = (
            self._llm_config.model_configuration.pretraining_data_type == PretrainingDataTypes.WITH_REASONING
        )

        question_ids = [example.question_id for example in data]
        questions = [example.question for example in data]
        prompts = [self._get_single_example_prompt(example, is_with_reasoning) for example in data]

        remapped_input_data = [
            PretrainDataSample(
                question_id=question_id,
                prompt=prompt,
                question=question if is_with_reasoning else None
            )
            for question_id, question, prompt in zip(question_ids, questions, prompts)
        ]
        tokenized_dataset = self._tokenized_eval_set_class(
            input_data=remapped_input_data,
            tokenizer=self._tokenizer,
            mode=ModelExecutionModes.EVAL
        )

        return [
            InferenceExample(
                question_id=question_id,
                question=question,
                prompt=prompt,
                tokenized_prompt=tokenized_example
            )
            for question_id, question, prompt, tokenized_example in zip(question_ids, questions, prompts, tokenized_dataset)
        ]

    def _get_dataloader(self, data: list[Text2SqlDataSample]) -> DataLoader:
        inference_dataset = self._prepare_inference_dataset(data=data)
        return DataLoader(
            inference_dataset,
            collate_fn=self._prepare_batch,
            batch_size=self._llm_config.generation_configuration.batch_size,
            shuffle=False
        )

    def _get_single_example_prompt(self, example: Text2SqlDataSample, is_with_reasoning: bool) -> str:
        inference_prompt_template = self._sql_generation_prompt_template.create_without_select(
            use_cot=False,
            use_knowledge=True,
            knowledge=example.knowledge,
            with_question=False if is_with_reasoning else True
        )

        filled_template_text = (
            (
                inference_prompt_template.replace("{question}", example.question)
                if not is_with_reasoning else inference_prompt_template
            )
            .replace("{database_schema}", example.database_schema)
            .replace("{knowledge}", example.knowledge)
        )

        return filled_template_text

    @staticmethod
    def _prepare_batch(examples: list[InferenceExample]) -> InferenceBatch:
        return InferenceBatch(
            question_ids=[example.question_id for example in examples],
            prompts=[example.prompt for example in examples],
            questions=[example.question for example in examples],
            input_ids=torch.tensor([example.tokenized_prompt["input_ids"] for example in examples], dtype=torch.int),
            attention_mask=torch.tensor([example.tokenized_prompt["attention_mask"] for example in examples], dtype=torch.int)
        )
