import argparse

from upath import UPath

from text2sql.datasets.schema_linking.parsers.bird_parser_for_retrieval import SchemaLinkingDataDumperForRetrieval
from text2sql.settings import DATASETS_DIR

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--bird-ground-truth-path",
        type=UPath,
        help="Path to the BIRD ground truth JSON file",
        default=DATASETS_DIR.joinpath("BIRD_dev", "dev.json"),
    )
    argument_parser.add_argument(
        "--database-root-path",
        type=UPath,
        help="Root path to the BIRD databases directory",
        default=DATASETS_DIR.joinpath("BIRD_dev", "dev_databases"),
    )
    argument_parser.add_argument(
        "--bird-metadata-path",
        type=UPath,
        help="Path to the BIRD metadata JSON file",
        default=DATASETS_DIR.joinpath("BIRD_dev", "dev_tables.json"),
    )
    argument_parser.add_argument(
        "--output-dir-path",
        type=UPath,
        help="Output directory path where the raw data will be stored",
        default=DATASETS_DIR.joinpath("raw_dataset_for_retrieval"),
    )

    args = argument_parser.parse_args()

    raw_data_dumper = SchemaLinkingDataDumperForRetrieval(
        bird_ground_truth_path=args.bird_ground_truth_path,
        database_root_path=args.database_root_path,
        bird_metadata_path=args.bird_metadata_path,
        output_dir_path=args.output_dir_path,
    )
    raw_data_dumper.dump()
