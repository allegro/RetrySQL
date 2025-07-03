import argparse
from upath import UPath

from text2sql.commons.db_utils.create_table_schema_prompt import CreateTableSchemaPrompt
from text2sql.commons.db_utils.schema_prompt_strategy import SchemaPromptStrategy
from text2sql.commons.io_utils import (
    save_objects_to_jsonl_file,
    load_configuration_to_target_dataclass,
    read_jsonl,
    translate_gcs_dir_to_local
)
from text2sql.datasets.bird_datasets import BirdDataset
from text2sql.datasets.domain.model import BirdDevDataSample
from text2sql.evaluation.run_benchmark_evaluation import run_evaluation
from text2sql.llms.configuration import OpenSourceLlmConfigurationDto, OpenSourceLlmConfiguration
from text2sql.llms.domain.enums import SupportedLlms
from text2sql.llms.domain.models import OpenSourceLlm
from text2sql.llms.model_factory import LlmModelFactory
from text2sql.modules.llm_input.prompt_templates.sql_generation_prompt_templates import BaselineSqlPromptTemplate
from text2sql.modules.pipelines.domain import PipelineResults
from text2sql.modules.pipelines.open_source_llm_pipeline import OpenSourceLlmPipeline
from text2sql.modules.schema_linking.domain.model import SchemaLinkingOutput
from text2sql.settings import DATASETS_DIR, PREDICTIONS_DIR


def save_pipeline_results_separately(pipeline_result: PipelineResults, output_dir: UPath) -> None:
    if pipeline_result.schema_linking_responses:
        save_objects_to_jsonl_file(
            objs=[schema_link_response.dict() for schema_link_response in pipeline_result.schema_linking_responses],
            file_path=output_dir.joinpath("schema_linking_responses.jsonl"),
        )

    save_objects_to_jsonl_file(
        objs=[sql_generation_response_candidates.dict() for sql_generation_response_candidates
              in pipeline_result.sql_generation_responses],
        file_path=output_dir.joinpath("sql_generation_responses.jsonl"),
    )


def parse_args() -> argparse.Namespace:
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--llm",
        type=SupportedLlms,
        required=True,
        choices=SupportedLlms.get_values(),
        help="Supported LLMs",
    )
    args_parser.add_argument(
        "--path-to-open-source-llm-config",
        type=UPath,
        required=True,
        help="Path to open source llm config. See examples in `configs/open_source_llms`",
    )
    args_parser.add_argument(
        "--run-evaluation",
        action=argparse.BooleanOptionalAction,
        help="If set, evaluation at the end of the pipeline"
    )
    args_parser.add_argument(
        "--is-lora-model",
        action=argparse.BooleanOptionalAction,
        help="If set, LoRA model loading will be triggered. Use it only if you are aiming to run the pipeline for a LoRA trained model"
    )
    args_parser.add_argument(
        "--path-to-pre-computed-schema-links",
        type=UPath,
        default=UPath("resources/schema_links/ground_truth_schema_links_bird_dev.jsonl"),
        required=False,
        help="Path to pre-computed schema-links",
    )
    args_parser.add_argument(
        "--data-output-dir", 
        type=UPath, 
        default=PREDICTIONS_DIR
    )
    args_parser.add_argument(
        "--eval-path", 
        type=UPath, 
        default=DATASETS_DIR.joinpath("BIRD_dev", "dev.json")
    )
    args_parser.add_argument(
        "--db-root-dir", 
        type=UPath, 
        default=DATASETS_DIR.joinpath("BIRD_dev")
    )

    return args_parser.parse_args()


def load_bird_data(eval_path: UPath, db_root_dir: UPath) -> list[BirdDevDataSample]:
    return BirdDataset().load_dev(
        samples_path=eval_path,
        db_root_dirpath=db_root_dir.joinpath("dev_databases"),
    )

def get_llm_model(
        llm: SupportedLlms,
        config_path: UPath,
        is_lora_model: bool = False
) -> tuple[OpenSourceLlm, OpenSourceLlmConfiguration]:
    open_source_model = LlmModelFactory.create(
        open_source_model_config_path=config_path,
        name=llm,
        is_lora_model=is_lora_model
    )
    open_source_model_config = load_configuration_to_target_dataclass(
        path_to_config_yaml_file=config_path,
        target_dataclass=OpenSourceLlmConfigurationDto,
    )
    open_source_model_config.to_domain().validate_configuration()
    return open_source_model, open_source_model_config


def setup_schema_linking(path_to_pre_computed_schema_links: UPath) -> list[SchemaLinkingOutput] | None:
    if path_to_pre_computed_schema_links:
        return read_jsonl(path_to_pre_computed_schema_links, class_schema=SchemaLinkingOutput)
    return None


def create_pipeline(
        model: OpenSourceLlm,
        config: OpenSourceLlmConfiguration,
        schema_links: list[SchemaLinkingOutput],
        schema_prompt_creator: SchemaPromptStrategy,
        sql_prompt_template: BaselineSqlPromptTemplate
) -> OpenSourceLlmPipeline:
    return OpenSourceLlmPipeline(
        open_source_llm=model,
        open_source_llm_config=config,
        schema_prompt_creator=schema_prompt_creator,
        sql_generation_prompt_template=sql_prompt_template,
        schema_linker=None,
        pre_computed_schema_links=schema_links
    )


if __name__ == "__main__":
    args = parse_args()

    translated_db_root_dir = translate_gcs_dir_to_local(args.db_root_dir)
    bird_data = load_bird_data(args.eval_path, translated_db_root_dir)
    model, config = get_llm_model(args.llm, args.path_to_open_source_llm_config, args.is_lora_model)

    schema_links = setup_schema_linking(args.path_to_pre_computed_schema_links)
    schema_prompt_creator = CreateTableSchemaPrompt()
    sql_prompt_template = BaselineSqlPromptTemplate()

    pipeline = create_pipeline(
        model, 
        config, 
        schema_links, 
        schema_prompt_creator, 
        sql_prompt_template
    )
    pipeline_results = pipeline(data=bird_data)
    save_pipeline_results_separately(pipeline_results, args.data_output_dir)

    if args.run_evaluation:
        run_evaluation(
            results_file_path=args.data_output_dir.joinpath("sql_generation_responses.jsonl"),
            ground_truth_path=args.eval_path,
            bird_base_path=translated_db_root_dir
        )
