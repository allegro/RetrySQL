import argparse
import asyncio
from dataclasses import asdict

from pydash import chain
from upath import UPath

from text2sql.commons.db_utils.create_table_schema_prompt import CreateTableSchemaPrompt
from text2sql.commons.db_utils.database_info import get_database_info
from text2sql.commons.io_utils import save_objects_to_jsonl_file, save_configuration_to_yaml
from text2sql.commons.logging_utils import get_logger
from text2sql.datasets.domain.enums import NumCorruptionsPerStepType, StepsToUseForCorruptionsType
from text2sql.datasets.domain.model import CorruptedSample, DatabaseSchema, PretrainDataSample, \
    RetryDataGenerationConfig, SchemaLinkingDataSample, Text2SqlDataSample
from text2sql.datasets.retry_data.error_generator.abstract import AbstractErrorGenerator
from text2sql.datasets.retry_data.error_generator.reasoning_steps import ReasoningStepsErrorGenerator
from text2sql.datasets.schema_linking.parsers.bird_dataset_parsing_utils import (
    read_bird_database_schemas,
    read_bird_ground_truth,
)
from text2sql.datasets.schema_linking.parsers.schema_link_parser import SchemaLinkParser
from text2sql.modules.llm_input.prompt_templates.sql_generation_prompt_templates import BaselineSqlPromptTemplate
from text2sql.settings import DATASETS_DIR

logger = get_logger()


def parse_arguments() -> RetryDataGenerationConfig:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--bird-databases-path",
        type=UPath,
        help="Path to the directory with BIRD databases",
        required=True
    )
    argument_parser.add_argument(
        "--ground-truth-path",
        type=UPath,
        help="Ground truth dataset file",
        required=True,
        default=DATASETS_DIR.joinpath("BIRD_dev", "dev.json"),
    )
    argument_parser.add_argument(
        "--bird-metadata-path",
        type=UPath,
        help="File with all BIRD tables and columns",
        required=True,
        default=DATASETS_DIR.joinpath("BIRD_dev", "dev_tables.json"),
    )
    argument_parser.add_argument(
        "--error-probability",
        type=float,
        help="Error probability, in range (0, 1).",
        required=True
    )
    argument_parser.add_argument(
        "--num-corruptions-per-step",
        type=NumCorruptionsPerStepType,
        help=("Used only if --correction-type is REASONING_STEPS. Controls whether single or multiple corruptions "
              " should be added per step."),
        default=NumCorruptionsPerStepType.SINGLE,
        choices=NumCorruptionsPerStepType.get_values(),
        required=False
    )
    argument_parser.add_argument(
        "--steps-to-use-for-corruptions",
        type=StepsToUseForCorruptionsType,
        help=("Used only if --correction-type is REASONING_STEPS. Controls whether only steps from future or steps "
              " from past and future should be used when corrupting the reasoning steps."),
        default=StepsToUseForCorruptionsType.FROM_FUTURE,
        choices=StepsToUseForCorruptionsType.get_values(),
        required=False
    )
    argument_parser.add_argument(
        "--output-directory",
        type=UPath,
        help="Output directory for the generated data",
        required=True
    )
    argument_parser.add_argument(
        "--multiply-factor",
        type=int,
        help="Multiply factor for how many time to corrupt each example",
        required=False,
        default=1
    )
    argument_parser.add_argument(
        "--databases-to-skip",
        type=list[str],
        help="List of databases to skip",
        required=False,
        default=["mondial_geo"]
    )
    args = argument_parser.parse_args()

    return RetryDataGenerationConfig(
        bird_databases_path=args.bird_databases_path,
        ground_truth_path=args.ground_truth_path,
        bird_metadata_path=args.bird_metadata_path,
        error_probability=args.error_probability,
        multiply_factor=args.multiply_factor,
        num_corruptions_per_step=args.num_corruptions_per_step,
        steps_to_use_for_corruptions=args.steps_to_use_for_corruptions,
        output_directory=args.output_directory,
        databases_to_skip=args.databases_to_skip
    )
    

async def corrupt_example(
    example: Text2SqlDataSample, 
    error_generator: AbstractErrorGenerator, 
    error_probability: float, 
    bird_metadata: list[DatabaseSchema],
    multiply_factor: int
) -> asyncio.Future:
    loop = asyncio.get_event_loop()
    
    corrupted_examples = [
        await loop.run_in_executor(
            None, 
            error_generator.corrupt,
            example, 
            error_probability, 
            bird_metadata
        )
        for _ in range(multiply_factor)
    ]

    return corrupted_examples


async def process_corrupted_example(
    corrupted_sample: CorruptedSample,
    schema_linking_ground_truth: list[SchemaLinkingDataSample],
    bird_databases_path: UPath
) -> PretrainDataSample:
    ground_truth_schema_links = (
        chain(schema_linking_ground_truth)
        .filter(lambda schema_linking_data_sample: schema_linking_data_sample.question_id == corrupted_sample.question_id)
        .head()
        .value()
        .schema_links
    )

    db_info = get_database_info(
        bird_databases_path / corrupted_sample.database_name / f"{corrupted_sample.database_name}.sqlite"
    )

    required_database_schema = CreateTableSchemaPrompt().create_schema_prompt(
        db_info=db_info,
        schema_links=ground_truth_schema_links
    )

    baseline_prompt_template_text = BaselineSqlPromptTemplate().create_without_select(
        use_cot=False, 
        use_knowledge=True, 
        knowledge=corrupted_sample.knowledge,
        with_question=False if corrupted_sample.reasoning_steps is not None else True
    )
    filled_template_text = (
        (
            baseline_prompt_template_text.replace("{question}", corrupted_sample.question)
            if corrupted_sample.reasoning_steps is None else baseline_prompt_template_text
        )
        .replace("{database_schema}", required_database_schema)
        .replace("{knowledge}", corrupted_sample.knowledge)
    )

    return PretrainDataSample(
        question_id=corrupted_sample.question_id,
        prompt=filled_template_text,
        query=corrupted_sample.sql_query,
        reasoning_steps=corrupted_sample.reasoning_steps,
        question=corrupted_sample.question if corrupted_sample.reasoning_steps is not None else None,
    )


async def corrupt_dataset(
    dataset: list[Text2SqlDataSample], 
    error_generator: AbstractErrorGenerator, 
    error_probability: float, 
    bird_metadata: list[DatabaseSchema],
    multiply_factor: int
):
    tasks = [
        corrupt_example(
            example=example, 
            error_generator=error_generator, 
            error_probability=error_probability, 
            bird_metadata=bird_metadata, 
            multiply_factor=multiply_factor
        ) 
        for example in dataset
    ]
    
    corrupted_examples = await asyncio.gather(*tasks)

    return corrupted_examples


async def join_queries_with_questions_and_schemas(
    corrupted_samples: list[CorruptedSample],
    schema_linking_ground_truth: list[SchemaLinkingDataSample],
    bird_databases_path: UPath
) -> list[PretrainDataSample]:
    tasks = [
        process_corrupted_example(
            corrupted_sample=corrupted_sample,
            schema_linking_ground_truth=schema_linking_ground_truth,
            bird_databases_path=bird_databases_path
        )
        for corrupted_sample in corrupted_samples
    ]

    pretrain_data = await asyncio.gather(*tasks)

    return pretrain_data


def generate_pretrain_data_with_corrections(
    bird_dataset: list[Text2SqlDataSample],
    bird_metadata: list[DatabaseSchema],
    error_probability: float,
    multiply_factor: int,
    bird_databases_path: UPath,
    num_corruptions_per_step: NumCorruptionsPerStepType | None = None,
    steps_to_use_for_corruptions: StepsToUseForCorruptionsType | None = None
) -> list[dict[str, str]]:
    schema_linking_ground_truth = SchemaLinkParser().run(
        bird_ground_truth_data=bird_dataset,
        bird_metadata=bird_metadata,
    )

    error_generator = (
        ReasoningStepsErrorGenerator(
            num_corruptions_per_step=num_corruptions_per_step,
            steps_to_use_for_corruptions=steps_to_use_for_corruptions
        )
    )


    logger.info("Generating errors and corrections...")

    processing_results = asyncio.get_event_loop().run_until_complete(
        corrupt_dataset(
            dataset=bird_dataset, 
            error_generator=error_generator, 
            error_probability=error_probability, 
            bird_metadata=bird_metadata,
            multiply_factor=multiply_factor
        )
    )

    flattened_corrupted_queries = chain(processing_results).flatten().value()

    logger.info("Making full pretrain data...")

    pretrain_data = asyncio.get_event_loop().run_until_complete(
        join_queries_with_questions_and_schemas(
            corrupted_samples=flattened_corrupted_queries,
            schema_linking_ground_truth=schema_linking_ground_truth,
            bird_databases_path=bird_databases_path
        )
    )

    return pretrain_data


if __name__ == "__main__":
    config = parse_arguments()

    logger.info(f"Generation config: {asdict(config)}")

    bird_dataset = read_bird_ground_truth(config.ground_truth_path)

    logger.info(f"Input dataset size: {len(bird_dataset)}")

    filtered_bird_dataset = (
        chain(bird_dataset)
        .filter(lambda example: example.database_name not in config.databases_to_skip)
        .value()
    )

    logger.info(f"Filtered dataset size, after filtering out databases {config.databases_to_skip}: {len(filtered_bird_dataset)}")

    bird_metadata = read_bird_database_schemas(config.bird_metadata_path)

    pretrain_data = generate_pretrain_data_with_corrections(
        bird_dataset=filtered_bird_dataset,
        bird_metadata=bird_metadata,
        error_probability=config.error_probability,
        multiply_factor=config.multiply_factor,
        bird_databases_path=config.bird_databases_path,
        num_corruptions_per_step=config.num_corruptions_per_step,
        steps_to_use_for_corruptions=config.steps_to_use_for_corruptions
    )

    logger.info(f"Retry data dataset size: {len(pretrain_data)}")

    save_configuration_to_yaml(filepath=config.output_directory / "config.yaml", config_dict=asdict(config))
    save_objects_to_jsonl_file(list(map(lambda sample: asdict(sample), pretrain_data)), config.output_directory / "pretrain_data.jsonl")
