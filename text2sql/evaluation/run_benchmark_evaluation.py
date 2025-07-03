import argparse
import json

from upath import UPath

from text2sql.commons.io_utils import translate_gcs_dir_to_local
from text2sql.commons.logging_utils import get_logger
from text2sql.evaluation.domain.enums import QuestionDifficulty, SupportedBenchmarks
from text2sql.evaluation.domain.model import GenerationResult, GroundTruth
from text2sql.evaluation.metrics.bird_metrics import BirdMetrics
from text2sql.settings import DATASETS_DIR

logger = get_logger(__name__)


def map_difficulty(difficulty: str) -> QuestionDifficulty:
    match difficulty:
        case "simple":
            return QuestionDifficulty.EASY
        case "moderate":
            return QuestionDifficulty.MEDIUM
        case "challenging":
            return QuestionDifficulty.HARD
        case _:
            raise ValueError(f"Unknown difficulty: {difficulty}")

def read_bird_dev_dataset(ground_truth_path: UPath) -> list[GroundTruth]:
    with ground_truth_path.open(mode="r") as ground_truth_file:
        ground_truth_json = json.load(ground_truth_file)

    return [
        GroundTruth(
            question_id=data_object["question_id"],
            database_name=data_object["db_id"],
            question_difficulty=map_difficulty(data_object["difficulty"]),
            sql_query=data_object["SQL"],
        )
        for data_object in ground_truth_json
    ]


def read_generation_results(results_file_path: UPath) -> list[GenerationResult]:
    with results_file_path.open(mode="r") as results_file:
        lines = results_file.readlines()

    result_objects = [json.loads(line) for line in lines]

    return [
        GenerationResult(
            question_id=int(result_object["input_data"]["id"]),
            sql_query=(result_object["response"]["sql"] if result_object["response"] is not None else "")
        )
        for result_object in result_objects
    ]

def run_evaluation(
    results_file_path: UPath,
    ground_truth_path: UPath,
    bird_base_path: UPath,
    cpu_count: int = 8,
    comparison_timeout: float = 30.0,
    benchmark: SupportedBenchmarks = SupportedBenchmarks.BIRD,
    calculate_ves: bool = False
) -> None:

    if benchmark == SupportedBenchmarks.BIRD:
        generation_results = read_generation_results(results_file_path=results_file_path)
        bird_ground_truth = read_bird_dev_dataset(ground_truth_path)
        translated_bird_base_path = translate_gcs_dir_to_local(bird_base_path)

        bird_metrics = BirdMetrics(
            base_path=translated_bird_base_path,
            cpu_count=cpu_count,
            timeout_in_seconds=comparison_timeout,
        )

        execution_accuracy = bird_metrics.calculate_execution_accuracy(
            results=generation_results,
            ground_truth=bird_ground_truth,
        )

        logger.info(f"Execution Accuracy Easy: {execution_accuracy.easy_accuracy:.2f}")
        logger.info(f"Execution Accuracy Medium: {execution_accuracy.medium_accuracy:.2f}")
        logger.info(f"Execution Accuracy Hard: {execution_accuracy.hard_accuracy:.2f}")
        logger.info(f"Execution Accuracy Overall: {execution_accuracy.overall_accuracy:.2f}")

        if calculate_ves:
            valid_efficiency_score = bird_metrics.calculate_valid_efficiency_score(
                results=generation_results,
                ground_truth=bird_ground_truth,
            )
            logger.info(f"Reward-based Valid Efficiency Score: {valid_efficiency_score:.2f}")

    elif benchmark == SupportedBenchmarks.SPIDER_V2:
        logger.error("SPIDER v2 support is not implemented yet!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-file-path",
        type=UPath,
        help="JSONL file with results",
        required=True,
    )
    parser.add_argument(
        "--ground-truth-path",
        type=UPath,
        help="Ground truth data file",
        default=DATASETS_DIR.joinpath("BIRD_dev", "dev.json"),
    )
    parser.add_argument(
        "--bird-base-path",
        type=UPath,
        help="Path to a folder with BIRD dev databases",
        default=DATASETS_DIR.joinpath("BIRD_dev"),
    )
    parser.add_argument(
        "--cpu-count",
        type=int,
        help="Number of CPU cores used for query execution",
        default=8,
    )
    parser.add_argument(
        "--comparison-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds for a evaluating a single example",
    )
    parser.add_argument(
        "--benchmark",
        choices=SupportedBenchmarks.get_values(),
        type=SupportedBenchmarks,
        help="Benchmark for which metrics will be calculated",
        default=SupportedBenchmarks.BIRD,
    )
    parser.add_argument(
        "--calculate-ves",
        action="store_true",
        required=False,
        default=False,
        help="Flag for enabling VES calculation in BIRD",
    )
    parser.add_argument(
        "--is-from-open-source-pipeline",
        action="store_true",
        required=False,
        default=False,
        help="Flag for enabling mapping between open source model output and evaluation input",
    )

    args = parser.parse_args()

    run_evaluation(
        results_file_path=args.results_file_path,
        ground_truth_path=args.ground_truth_path,
        bird_base_path=args.bird_base_path,
        cpu_count=args.cpu_count,
        comparison_timeout=args.comparison_timeout,
        benchmark=args.benchmark,
        calculate_ves=args.calculate_ves
    )
