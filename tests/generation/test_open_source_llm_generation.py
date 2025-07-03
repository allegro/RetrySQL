import pytest
from upath import UPath

from transformers import AutoModelForCausalLM, AutoTokenizer

from text2sql.llms.domain.models import OpenSourceLlm
from text2sql.llms.configuration import OpenSourceLlmConfiguration
from text2sql.modules.generation.domain import InferenceBatch
from text2sql.modules.generation.open_source_llm_sql_generator import OpenSourceLlmSqlGenerator
from text2sql.modules.llm_input.prompt_templates.sql_generation_prompt_templates import BaselineSqlPromptTemplate
from tests.helpers.builders import OpenSourceLlmConfigurationBuilder

class TestOpenSourceLlmGeneration:

    def setup_method(self) -> None:
        self.llm = self.get_small_generative_model_for_testing_purposes()
        self.sql_generation_prompt_template = BaselineSqlPromptTemplate()

        self.bird_dev_examples_path = UPath("tests/resources/bird_dev_dataset_extract/dev.json")
        self.db_root_dirpath = UPath("tests/resources/bird_dev_dataset_extract/dev_databases")

    @staticmethod
    def get_small_generative_model_for_testing_purposes() -> OpenSourceLlm:
        return OpenSourceLlm(
            llm=AutoModelForCausalLM.from_pretrained("openai-community/gpt2"),
            tokenizer=AutoTokenizer.from_pretrained("openai-community/gpt2"),
        )

    def get_generator(
            self, open_source_llm_config: OpenSourceLlmConfiguration
    ) -> OpenSourceLlmSqlGenerator:
        return OpenSourceLlmSqlGenerator(
            llm=self.llm,
            llm_config=open_source_llm_config,
            sql_generation_prompt_template=self.sql_generation_prompt_template,
        )

    @pytest.mark.parametrize(
        "generated_sql, add_select_statement_to_the_generated_sql, normalize_generated_sql, expected_sql",
        [
            ("col_a FROM table", True, False, "SELECT col_a FROM table"),
            ("SELECT col_a FROM table", False, False, "SELECT col_a FROM table"),
            ("SELECT col_a\nFROM table", False, True, "SELECT col_a FROM table"),
            ("SELECT col_a\t\t\t\t\tFROM table", False, True, "SELECT col_a     FROM table"),
            ("SELECT ``col_a`` FROM ``table``", False, True, "SELECT `col_a` FROM `table`"),
            # '<|endoftext|>' == </s> for GPT-2 model
            ("SELECT col_a FROM table <|endoftext|>", False, True, "SELECT col_a FROM table"),
            ("SELECT col_a FROM table <|im_end|>", False, True, "SELECT col_a FROM table"),

        ]
    )
    def test_postprocess(
            self,
            generated_sql,
            add_select_statement_to_the_generated_sql,
            normalize_generated_sql,
            expected_sql
    ) -> None:
        # GIVEN
        config = (
            OpenSourceLlmConfigurationBuilder()
            .with_post_processing_config(
                add_select=add_select_statement_to_the_generated_sql,
                normalize_sql=normalize_generated_sql
            )
            .build()
        )
        # AND
        generator = self.get_generator(config)

        # WHEN
        result = generator.postprocess(generated_sql, question="")

        # THEN
        assert result == expected_sql

    def test_split_output_at_question(self) -> None:
        # GIVEN
        config = (
            OpenSourceLlmConfigurationBuilder()
            .with_post_processing_config(
                add_select=True,
                normalize_sql=True,
                split_output_at_question=True
            )
            .build()
        )
        # AND
        generator = self.get_generator(config)

        # WHEN
        result = generator.postprocess(
            generated_sql="CREATE TABLE example_table (col_a INT, col_b TEXT); -- External knowledge: Bazinga! -- Question: What is in example_table? SELECT * FROM example_table", 
            question="What is in example_table?")

        # THEN
        assert result == "SELECT * FROM example_table"

    @pytest.mark.parametrize(
        "input_prompt, trim_output_from_input_sequence, expected_output",
        [
            ("CREATE TABLE ...", False, "CREATE TABLE ...\n\nThe following table contains the table of contents for the table.\n\nTABLE Name Description Name"),
            ("CREATE TABLE ...", True, "\n\nThe following table contains the table of contents for the table.\n\nTABLE Name Description Name"),
        ]
    )
    def test_trimming_of_input_tokens_from_the_output(
            self,
            input_prompt,
            trim_output_from_input_sequence,
            expected_output
    ) -> None:
        # GIVEN
        config = (
            OpenSourceLlmConfigurationBuilder()
            .with_post_processing_config(
                trim_output=trim_output_from_input_sequence,
            )
            .build()
        )
        # AND
        generator = self.get_generator(config)
        # AND
        tokenized_prompts = generator._tokenizer([input_prompt], return_tensors="pt")
        batch = InferenceBatch(
            question_ids=[1],
            prompts=[input_prompt],
            questions=[""],
            input_ids=tokenized_prompts.input_ids,
            attention_mask=tokenized_prompts.attention_mask
        )

        # WHEN
        outputs = generator.generate_sql_tokens(batch)
        trimmed_and_reshaped_outputs = generator.trim_and_reshape_outputs(batch=batch, outputs=outputs.sequences)

        # THEN
        for output in trimmed_and_reshaped_outputs:
            generated_sql = generator._tokenizer.decode(
                output.squeeze(0),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            assert generated_sql == expected_output
