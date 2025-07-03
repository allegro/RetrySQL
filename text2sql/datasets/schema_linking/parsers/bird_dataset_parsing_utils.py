import json

from pydash import chain
from upath import UPath

from text2sql.datasets.domain.model import (
    BirdDevDataSample,
    BirdTrainDataSample,
    DatabaseSchema,
    SchemaColumn,
    TableColumnPair,
)
from text2sql.datasets.dto.bird_dataset import BirdDatasetDto


def read_bird_ground_truth(
    ground_truth_path: UPath, limit: int | None = None
) -> list[BirdDevDataSample] | list[BirdTrainDataSample]:
    with ground_truth_path.open("r") as ground_truth_file:
        ground_truth_json = json.load(ground_truth_file)

    return BirdDatasetDto(entries=ground_truth_json[:limit]).to_domain()


def read_bird_database_schemas(metadata_path: UPath) -> list[DatabaseSchema]:

    def _get_primary_and_foreign_key_information(column_index) -> tuple[bool, bool]:
        is_primary = column_index in primary_key_indices_set
        is_foreign = column_index in foreign_key_indices_set
        return is_primary, is_foreign

    def _build_primary_and_foreign_key_indices_sets() -> tuple[set[int], set[int]]:

        primary_keys_set = set()
        for primary_key_index in primary_key_indices:
            if isinstance(primary_key_index, list):
                primary_keys_set.update(primary_key_index)
            else:
                primary_keys_set.add(primary_key_index)

        foreign_keys_set = set()
        for fk in foreign_key_indices:
            foreign_keys_set.update(fk)

        return primary_keys_set, foreign_keys_set

    with metadata_path.open("r") as metadata_file:
        metadata_for_all_databases = json.load(metadata_file)

    full_metadata = []

    for database_metadata in metadata_for_all_databases:
        database_name = database_metadata["db_id"]
        tables = database_metadata["table_names_original"]
        columns = database_metadata["column_names_original"]
        primary_key_indices = database_metadata["primary_keys"]
        foreign_key_indices = database_metadata["foreign_keys"]

        primary_key_indices_set, foreign_key_indices_set = _build_primary_and_foreign_key_indices_sets()

        table_column_map = (
            chain(columns)
            .filter(lambda column: column[0] != -1)
            .map(
                lambda column: TableColumnPair(
                    table_name=tables[column[0]].lower(),
                    expanded_column=SchemaColumn(
                        column_name=column[1].lower(),
                        is_primary_key=_get_primary_and_foreign_key_information(columns.index(column))[0],
                        is_foreign_key=_get_primary_and_foreign_key_information(columns.index(column))[1],
                    ),
                )
            )
            .group_by(lambda table_column_pair: table_column_pair.table_name)
            .map_values(
                lambda table_column_pairs: [
                    table_column_pair.expanded_column for table_column_pair in table_column_pairs
                ]
            )
            .value()
        )

        full_metadata.append(DatabaseSchema(database_name=database_name, tables=table_column_map))

    return full_metadata
