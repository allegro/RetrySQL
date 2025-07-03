from pydantic import BaseModel, Field


class PrimaryKey(BaseModel):
    columns: list[str]


class ForeignColumn(BaseModel):
    local_column: str
    referenced_column: str


class ForeignKey(BaseModel):
    referenced_table: str
    column_links: list[ForeignColumn]


class ColumnInfo(BaseModel):
    cid: int = Field(description="Column ID representing the column’s position in the table.")
    name: str
    is_primary_key: bool = False
    is_foreign_key: bool = False
    is_not_null: bool = False
    data_format: str | None = None
    default_value: str | None = None


class TableInfo(BaseModel):
    name: str
    columns: dict[int, ColumnInfo]
    primary_key: PrimaryKey
    foreign_keys: list[ForeignKey]
    example_rows: list[tuple] | None = None


class DatabaseInfo(BaseModel):
    name: str
    tables: dict[str, TableInfo]
