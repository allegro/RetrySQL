import argparse

from upath import UPath

from text2sql.commons.db_utils.create_table_schema_prompt import CreateTableSchemaPrompt
from text2sql.commons.io_utils import save_predictions, read_jsonl
from text2sql.datasets.bird_datasets import BirdDataset
from text2sql.llms.domain.baseline_params import get_baseline_llm_params
from text2sql.llms.domain.enums import SupportedLlms
from text2sql.llms.domain.models import LlmParams
from text2sql.llms.model_factory import LlmModelFactory
from text2sql.modules.pipelines.baseline_pipeline import BaselinePipeline
from text2sql.evaluation.run_benchmark_evaluation import run_evaluation
from text2sql.modules.schema_linking.domain.model import SchemaLinkingOutput
from text2sql.settings import DATASETS_DIR, PREDICTIONS_DIR


def get_output_file_name(llm: str, params: LlmParams) -> str:
    use_cot = "cot" if args.use_cot else ""
    use_pydantic = "pydantic-mapping" if args.use_pydantic else ""
    use_knowledge = "with-knowledge" if args.use_knowledge else ""
    num_example_rows = f"example-rows-{args.example_rows}" if args.example_rows else ""

    options = [use_cot, use_knowledge, num_example_rows, use_pydantic]
    options = [option for option in options if option != ""]

    prefix = "-".join(options)

    return (
        f"reimplemented-baseline-{llm}-temperature-{params.temperature}-max-output-{params.max_output_tokens}"
        f"-{prefix}.jsonl"
    )

def load_pre_computed_schema_links(path: UPath) -> list[SchemaLinkingOutput]:
    return read_jsonl(path, class_schema=SchemaLinkingOutput)

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--llm",
        type=SupportedLlms,
        required=True,
        choices=SupportedLlms.get_values(),
        help="Supported LLMs",
    )
    args_parser.add_argument(
        "--run-evaluation",
        action=argparse.BooleanOptionalAction,
        help="If set, evaluation at the end of the pipeline"
    )
    args_parser.add_argument(
        "--path-to-pre-computed-schema-links",
        type=UPath,
        default=UPath("resources/schema_links/ground_truth_schema_links_bird_dev.jsonl"),
        required=False,
        help="Path to pre-computed schema-links",
    )
    args_parser.add_argument("--use-knowledge", action=argparse.BooleanOptionalAction)
    args_parser.add_argument("--use-cot", action=argparse.BooleanOptionalAction)
    args_parser.add_argument("--use-pydantic", action=argparse.BooleanOptionalAction)
    args_parser.add_argument("--example-rows", type=int, default=0)
    args_parser.add_argument("--data-output-dir", type=UPath, default=PREDICTIONS_DIR)
    args_parser.add_argument("--eval-path", type=UPath, default=DATASETS_DIR.joinpath("BIRD_dev", "dev.json"))
    args_parser.add_argument("--db-root-dir", type=UPath, default=DATASETS_DIR.joinpath("BIRD_dev"))
    args = args_parser.parse_args()

    bird_dev_data = BirdDataset().load_dev(
        samples_path=args.eval_path,
        db_root_dirpath=args.db_root_dir.joinpath("dev_databases"),
    )

    # -- GET LLM MODEL --
    llm_params = get_baseline_llm_params(args.llm)
    model = LlmModelFactory.create(name=args.llm, params=llm_params)

    # -- BIRD BASELINE PIPELINE --
    schema_prompt_creator = CreateTableSchemaPrompt()
    baseline_pipeline = BaselinePipeline(
        llm=model,
        schema_prompt_creator=schema_prompt_creator,
        use_pydantic_output=args.use_pydantic,
        use_cot=args.use_cot,
        use_knowledge=args.use_knowledge,
        num_example_rows=args.example_rows,
        shuffle_schema_cols=False,
        pre_computed_schema_links=(
            load_pre_computed_schema_links(args.path_to_pre_computed_schema_links) if args.path_to_pre_computed_schema_links
            else None
        ),
    )

    # -- GENERATE SQL RESPONSES --
    pipeline_results = baseline_pipeline(data=bird_dev_data)

    output_file_name = get_output_file_name(llm=args.llm, params=llm_params)
    save_predictions(
        pipeline_results.sql_generation_responses,
        output_dir=args.data_output_dir,
        output_file_name=output_file_name,
    )

    if args.run_evaluation:
        run_evaluation(
            results_file_path=args.data_output_dir.joinpath(output_file_name),
            ground_truth_path=args.eval_path,
            bird_base_path=args.db_root_dir,
        )