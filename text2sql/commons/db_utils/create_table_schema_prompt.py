import random

from text2sql.commons.db_utils.domain.model import DatabaseInfo
from text2sql.commons.db_utils.schema_prompt_strategy import SchemaPromptStrategy
from text2sql.datasets.domain.model import SchemaLink


class CreateTableSchemaPrompt(SchemaPromptStrategy):
    def create_schema_prompt(
        self,
        db_info: DatabaseInfo,
        schema_links: list[SchemaLink],
        shuffle_cols: bool = False,
    ) -> str:
        table_filtered_columns = (
            {link.table_name: link.columns for link in schema_links}
            if len(schema_links) > 0
            else {table_name: [] for table_name, table_info in db_info.tables.items()}
        )

        table_schemas = dict()
        for table_name, filter_column_names in table_filtered_columns.items():
            create_prompt, column_ids = self._create_table_ddl_statement(
                db_info=db_info,
                table_name=table_name,
                filter_columns=filter_column_names,
                shuffle_cols=shuffle_cols,
            )
            table_schemas[table_name] = create_prompt

            # if database info object includes the example rows, add them to the schema prompt
            example_rows = db_info.tables[table_name].example_rows
            if not example_rows:
                continue

            rows_prompt = self._create_example_rows_prompt(
                db_info=db_info,
                table_name=table_name,
                column_ids=column_ids,
            )
            verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(
                len(example_rows),
                table_name,
                len(example_rows),
                rows_prompt,
            )
            table_schemas[table_name] = f"{create_prompt} \n {verbose_prompt}"

        return "\n\n".join(table_schemas.values())

    @staticmethod
    def _create_table_ddl_statement(
        db_info: DatabaseInfo,
        table_name: str,
        filter_columns: list[str],
        shuffle_cols: bool = False,
    ) -> tuple[str, list[int]]:
        table_info = db_info.tables[table_name]
        primary_key = table_info.primary_key
        foreign_keys = table_info.foreign_keys
        columns = (
            [col for col in table_info.columns.values() if col.name in filter_columns]
            if len(filter_columns) > 0
            else list(table_info.columns.values())
        )
        if shuffle_cols:
            random.shuffle(columns)

        # take order of columns by column id (cid)
        column_ids = [col.cid for col in columns]

        # Column definitions
        column_definitions = []
        for col in columns:
            col_def = f"`{col.name}` {col.data_format.upper()}"
            if col.default_value is not None:
                col_def += f"  default {col.default_value}"

            if col.is_not_null:
                col_def += "  not null"
            else:
                col_def += "  null"

            if col.is_primary_key and len(primary_key.columns) == 1:
                col_def += "  PRIMARY KEY"

            column_definitions.append(col_def)

        # Table constraints -- foreign keys
        table_constraints = []
        for fk in foreign_keys:
            local_cols = [col_link.local_column for col_link in fk.column_links]

            # check if all local columns in the foreign key are in the table's filter_columns
            if not all(col in filter_columns for col in local_cols):
                continue

            local_cols = [f"`{local_col}`" for local_col in local_cols]
            ref_cols = [f"`{col_link.referenced_column}`" for col_link in fk.column_links]

            foreign_key_constraint = (
                f'FOREIGN KEY ({", ".join(local_cols)}) REFERENCES `{fk.referenced_table}` ({", ".join(ref_cols)})'
            )
            table_constraints.append(foreign_key_constraint)

        # Table constraints -- compound primary key
        if len(primary_key.columns) > 1:
            # check if all columns in the primary key are in the filter_columns, if not skip the compound primary key
            if all(col in filter_columns for col in primary_key.columns):
                pk_columns = [f"`{col}`" for col in primary_key.columns]
                primary_key_constraint = f"PRIMARY KEY ({', '.join(pk_columns)})"
                table_constraints.append(primary_key_constraint)

        # Combine column definitions and constraints
        create_table_body = ",\n".join(column_definitions + table_constraints)
        create_table_statement = f"CREATE TABLE `{table_name}`\n(\n{create_table_body}\n);"

        return create_table_statement, column_ids

    @staticmethod
    def _create_example_rows_prompt(
        db_info: DatabaseInfo,
        table_name: str,
        column_ids: list[int],
    ) -> str:
        # get column names & values filtering and ordering by provided column_ids
        columns = [db_info.tables[table_name].columns[col_id] for col_id in column_ids]
        column_names = [col.name for col in columns]

        table_example_rows = db_info.tables[table_name].example_rows
        example_values = [tuple(row[col_id] for col_id in column_ids) for row in table_example_rows]

        rows = []
        # Determine the maximum width of each column
        widths = [
            max(len(str(value[i])) for value in example_values + [column_names]) for i in range(len(column_names))
        ]

        # Print the column names
        header = "".join(f"{column.rjust(width)} " for column, width in zip(column_names, widths))
        # Print the values
        for value in example_values:
            row = "".join(f"{str(v).rjust(width)} " for v, width in zip(value, widths))
            rows.append(row)

        rows = "\n".join(rows)
        final_output = header + "\n" + rows
        return final_output
