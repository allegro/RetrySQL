import argparse

from upath import UPath

from text2sql.commons.io_utils import read_jsonl
from text2sql.commons.logging_utils import get_logger
from text2sql.datasets.domain.model import SchemaLinkingDataSample
from text2sql.evaluation.metrics.schema_linking_metrics import SchemaLinkingMetrics
from text2sql.modules.schema_linking.domain.model import SchemaLinkingOutput
from text2sql.settings import DATASETS_DIR

logger = get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-file-path",
        type=UPath,
        help="JSONL file with results",
        required=True,
    )
    parser.add_argument(
        "--ground-truth-path",
        type=UPath,
        help="Ground truth data file",
        default=DATASETS_DIR.joinpath("BIRD_dev", "schema_linking_dataset.jsonl"),
    )
    args = parser.parse_args()

    predictions = read_jsonl(
        args.predictions_file_path,
        class_schema=SchemaLinkingOutput,
    )
    ground_truths = read_jsonl(
        args.ground_truth_path,
        class_schema=SchemaLinkingDataSample,
    )

    schema_linking_metrics = SchemaLinkingMetrics(
        predictions=predictions,
        ground_truths=ground_truths,
    )

    schema_linking_metrics.compute_metrics()
