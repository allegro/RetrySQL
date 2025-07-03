from copy import deepcopy

from text2sql.commons.db_utils.domain.model import DatabaseInfo
from text2sql.commons.logging_utils import get_logger
from text2sql.datasets.domain.model import SchemaLink, BirdDevDataSample
from text2sql.modules.schema_linking.domain.model import SchemaLinkingOutput
from text2sql.commons.db_utils.create_table_schema_prompt import SchemaPromptStrategy


logger = get_logger(__name__)


def normalize_schema_links(schema_links: list[SchemaLink]) -> list[SchemaLink]:
    for link in schema_links:
        link.table_name = link.table_name.strip().replace('"', "").replace("`", "").lower()
        link.columns = [col.strip().replace('"', "").replace("`", "").lower() for col in link.columns]

    return schema_links


def postprocess_schema_links(db_info: DatabaseInfo, schema_links: list[SchemaLink]) -> list[SchemaLink]:
    if len(schema_links) == 0:
        return []

    schema_links = normalize_schema_links(schema_links=schema_links)

    adjusted_links = []
    for link in schema_links:
        linked_table = link.table_name

        # Check if the table actually exists in database schema
        if link.table_name not in db_info.tables.keys():
            logger.debug(
                f"Schema link for table '{linked_table}' provided "
                f"but table does not exist in the database '{db_info.name}'."
            )
            continue

        table_columns = db_info.tables.get(linked_table).columns.values()
        existing_column_names = [column.name for column in table_columns]

        # Filter schema links columns to those that actually exists in the table
        link.columns = [col for col in link.columns if col in existing_column_names]

        # In case the column list is empty, add all column names from database info schema
        if len(link.columns) == 0:
            link.columns = existing_column_names
            adjusted_links.append(link)
            continue

        # Add missing primary and foreign keys to schema link's columns
        primary_foreign_keys = {col.name for col in table_columns if col.is_primary_key or col.is_foreign_key}
        missing_cols = primary_foreign_keys - set(link.columns)
        
        if missing_cols:
            link.columns.extend(missing_cols)

        adjusted_links.append(link)

    return adjusted_links


def prepare_schema_links_for_update(
        question_id: int, 
        question_links_map: dict[int, list[SchemaLink]],
        db_info: DatabaseInfo,
        schema_link_postprocessing: bool
) -> list[SchemaLink]:
    schema_links_for_question = question_links_map[question_id]

    if schema_link_postprocessing:
        return postprocess_schema_links(
            db_info=db_info,
            schema_links=deepcopy(schema_links_for_question),
        )
    
    return schema_links_for_question


def update_schema_in_data_samples(
    data: list[BirdDevDataSample], 
    schema_links_for_each_sample: list[list[SchemaLink]],
    schema_prompt_creator: SchemaPromptStrategy,
    database_infos: dict[str, DatabaseInfo],
    shuffle_schema_columns: bool = False
) -> list[BirdDevDataSample]:
    data_copy = deepcopy(data)

    for sample, schema_links in zip(data_copy, schema_links_for_each_sample):
        if len(schema_links) > 0:
            sample.database_schema = schema_prompt_creator.create_schema_prompt(
                db_info=database_infos[sample.database_name],
                schema_links=schema_links,
                shuffle_cols=shuffle_schema_columns,
            )

    return data_copy


def update_db_schema_with_predicted_schema_links(
        data: list[BirdDevDataSample],
        schema_linking_responses_raw: list[SchemaLinkingOutput],
        schema_prompt_creator: SchemaPromptStrategy,
        database_infos: dict[str, DatabaseInfo],
        schema_link_postprocessing: bool = True,
        shuffle_schema_columns: bool = False,
) -> tuple[list[BirdDevDataSample], list[SchemaLinkingOutput]]:
    data_copy = deepcopy(data)

    schema_links_data_id_map = {
        schema_res.question_id: schema_res.schema_links for schema_res in schema_linking_responses_raw
    }

    processed_schema_links = [
        prepare_schema_links_for_update(
            question_id=sample.question_id,
            question_links_map=schema_links_data_id_map,
            db_info=database_infos[sample.database_name],
            schema_link_postprocessing=schema_link_postprocessing
        ) for sample in data_copy
    ]

    updated_data = update_schema_in_data_samples(
        data=data,
        schema_links_for_each_sample=processed_schema_links,
        schema_prompt_creator=schema_prompt_creator,
        database_infos=database_infos,
        shuffle_schema_columns=shuffle_schema_columns
    )

    schema_linking_responses = [
        SchemaLinkingOutput(
            question_id=sample.question_id,
            question=sample.question,
            schema_links=schema_links,
        ) for sample, schema_links in zip(data_copy, processed_schema_links)
    ]

    return updated_data, schema_linking_responses
