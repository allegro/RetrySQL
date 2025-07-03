import argparse
import logging

from upath import UPath

from text2sql.commons.logging_utils import setup_logger
from text2sql.datasets.domain.model import ReasoningStepsGenerationConfig
from text2sql.datasets.reasoning_steps.reasoning_step_generator import ReasoningStepGenerator
from text2sql.datasets.schema_linking.parsers.bird_dataset_parsing_utils import read_bird_ground_truth
from text2sql.llms.domain.baseline_params import get_baseline_llm_params
from text2sql.llms.domain.enums import SupportedLlms
from text2sql.llms.model_factory import LlmModelFactory

logger = logging.getLogger(__name__)


def parse_args() -> ReasoningStepsGenerationConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bird-dataset-path",
        type=UPath,
        help="Path to the BIRD train JSON file",
        required=True
    )
    parser.add_argument(
        "--llm",
        type=SupportedLlms,
        required=True,
        choices=SupportedLlms.get_values(),
        help="LLM to use for generating reasoning steps",
    )
    parser.add_argument(
        "--output-dir",
        type=UPath,
        help="Path to a dir where BIRD dataset enriched with reasoning steps will be saved",
        required=True
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of BIRD samples for which to generate reasoning steps",
    )
    args = parser.parse_args()

    return ReasoningStepsGenerationConfig.from_args(args)


if __name__ == "__main__":
    setup_logger()
    config = parse_args()

    logger.info(f"Config: {config}")

    generator = ReasoningStepGenerator(
        bird_dataset=read_bird_ground_truth(
            ground_truth_path=config.bird_dataset_path,
            limit=config.limit,
        ),
        model=LlmModelFactory.create(
            name=config.llm,
            params=get_baseline_llm_params(config.llm)
        ),
        output_dir=config.output_dir
    )
    generator.generate_reasoning_steps()
