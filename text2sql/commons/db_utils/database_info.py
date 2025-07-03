from collections import defaultdict

from upath import UPath

from text2sql.commons.db_utils.domain.model import (
    ColumnInfo,
    DatabaseInfo,
    ForeignColumn,
    ForeignKey,
    PrimaryKey,
    TableInfo,
)
from text2sql.commons.db_utils.execution import execute_sql
from text2sql.commons.logging_utils import get_logger


logger = get_logger(__name__)


def get_database_info(db_path: UPath, num_example_rows: int = 0) -> DatabaseInfo:
    table_names = get_table_names(db_path=db_path)
    table_schemas = {
        table: get_table_info(db_path=db_path, table_name=table, num_example_rows=num_example_rows)
        for table in table_names
    }

    return DatabaseInfo(name=db_path.stem, tables=table_schemas)


def get_table_names(db_path: UPath) -> list[str]:
    query = "SELECT name FROM sqlite_master WHERE type='table'"
    table_names = execute_sql(db_path=db_path, sql=query, fetch="all")

    return [
        table[0].replace('"', "").replace("`", "").lower() for table in table_names if table[0] != "sqlite_sequence"
    ]


def get_table_info(db_path: UPath, table_name: str, num_example_rows: int = 0) -> TableInfo:
    # table constraints
    primary_key = get_primary_key(db_path=db_path, table_name=table_name)
    foreign_keys = get_foreign_keys(db_path=db_path, table_name=table_name)

    # columns definition
    fk_local_columns = [fk_col.local_column for fk in foreign_keys for fk_col in fk.column_links]
    table_info = execute_sql(db_path=db_path, sql=f'PRAGMA table_info("{table_name}");', fetch="all")

    column_infos = dict()
    for col_id, col_name, data_format, not_null, default_value, pk_pos_index in table_info:
        is_primary_key = pk_pos_index > 0

        is_foreign_key = any(fk_col == col_name for fk_col in fk_local_columns)

        column_infos[int(col_id)] = ColumnInfo(
            name=col_name.replace('"', "").replace("`", "").lower(),
            cid=int(col_id),
            is_primary_key=is_primary_key,
            is_foreign_key=is_foreign_key,
            data_format=data_format.lower(),
            default_value=default_value,
            is_not_null=(not_null == 1),
        )

    return TableInfo(
        name=table_name,
        columns=column_infos,
        primary_key=primary_key,
        foreign_keys=foreign_keys,
        example_rows=(
            get_table_rows(db_path=db_path, table_name=table_name, num_example_rows=num_example_rows)
            if num_example_rows > 0
            else None
        ),
    )


def get_primary_key(db_path: UPath, table_name: str) -> PrimaryKey:
    table_info = execute_sql(db_path=db_path, sql=f'PRAGMA table_info("{table_name}");', fetch="all")

    pk_columns = []
    for col_id, col_name, data_format, not_null, default_value, pk_pos_index in table_info:
        if pk_pos_index > 0:
            pk_columns.append((pk_pos_index, col_name.replace('"', "").replace("`", "").lower()))

    # sort by primary column position index in case of compound primary keys
    pk_columns.sort(key=lambda x: x[0])
    pk_columns = [pk[1] for pk in pk_columns]

    return PrimaryKey(columns=pk_columns)


def get_foreign_keys(db_path: UPath, table_name: str) -> list[ForeignKey]:
    foreign_keys_info = execute_sql(db_path=db_path, sql=f'PRAGMA foreign_key_list("{table_name}");', fetch="all")

    foreign_keys_dict = defaultdict(list)
    for fk_id, seq_order, ref_table, local_col, ref_col, *_ in foreign_keys_info:
        foreign_keys_dict[(fk_id, ref_table)].append((seq_order, local_col, ref_col))

    foreign_keys = []
    for (fk_id, ref_table), fk_col_link_info in foreign_keys_dict.items():
        fk_col_link_info.sort(key=lambda x: x[0])

        if any(col[2] is None for col in fk_col_link_info):
            ref_table_info = execute_sql(db_path=db_path, sql=f'PRAGMA table_info("{ref_table}");', fetch="all")
            ref_table_pk = [col[1] for col in ref_table_info if col[5] > 0]

            try:
                assert len(fk_col_link_info) == 1 and len(ref_table_pk) == 1, (
                    "In SQLite the NULL value in 'to' referenced column from query 'PRAGMA foreign_key_list()'"
                    " is allowed for single-column foreign key."
                )
            except AssertionError as e:
                logger.error(f"Error getting foreign keys for DB, table: {db_path}, {table_name}")
                continue
            
            fk_col_link_info[0] = (fk_col_link_info[0][0], fk_col_link_info[0][1], ref_table_pk[0])

        foreign_keys.append(
            ForeignKey(
                referenced_table=ref_table.replace('"', "").replace("`", "").lower(),
                column_links=[
                    ForeignColumn(
                        local_column=col[1].replace('"', "").replace("`", "").lower(),
                        referenced_column=col[2].replace('"', "").replace("`", "").lower(),
                    )
                    for col in fk_col_link_info
                ],
            )
        )

    return foreign_keys


def get_table_rows(db_path: UPath, table_name: str, num_example_rows: int) -> list[tuple[str]]:
    query = f'SELECT * FROM "{table_name}" LIMIT {num_example_rows}'
    result = execute_sql(db_path=db_path, sql=query, fetch="all")

    return result
