import argparse

from upath import UPath

from text2sql.commons.io_utils import save_objects_to_jsonl_file
from text2sql.datasets.schema_linking.parsers.bird_dataset_parsing_utils import (
    read_bird_database_schemas,
    read_bird_ground_truth,
)
from text2sql.datasets.schema_linking.parsers.schema_link_parser import SchemaLinkParser
from text2sql.settings import DATASETS_DIR

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--ground-truth-path",
        type=UPath,
        help="Ground truth dataset file",
        default=DATASETS_DIR.joinpath("BIRD_dev", "dev.json"),
    )
    argument_parser.add_argument(
        "--bird-metadata-path",
        type=UPath,
        help="File with all BIRD tables and columns",
        default=DATASETS_DIR.joinpath("BIRD_dev", "dev_tables.json"),
    )
    argument_parser.add_argument(
        "--output-file-path",
        type=UPath,
        help="Output file",
        required=False,
        default=DATASETS_DIR.joinpath("BIRD_dev", "schema_linking_dataset.jsonl"),
    )
    args = argument_parser.parse_args()

    bird_dev_dataset = read_bird_ground_truth(args.ground_truth_path)
    bird_metadata = read_bird_database_schemas(args.bird_metadata_path)

    dataset_creator = SchemaLinkParser()
    schema_linking_dataset = dataset_creator.run(
        bird_ground_truth_data=bird_dev_dataset,
        bird_metadata=bird_metadata,
    )

    save_objects_to_jsonl_file([example.dict() for example in schema_linking_dataset], args.output_file_path)
