from typing import Type

from allms.domain.input_data import InputData
from allms.domain.response import ResponseData
from pydantic import BaseModel, Field


class GeneratedSqlOutput(BaseModel):
    sql: str = Field(description="Generated SQL query")


class TableAndColumnSchemaLinkingOutput(BaseModel):
    schema_links: list[tuple[str, str]] = Field(
        description="List of selected relevant table-column pairs (schema links) "
        "where each pair is represented as a tuple (table_name, column_name).",
    )


class BatchLlmInput(BaseModel):
    input_data: list[InputData]
    prompt_template: str | None = None
    output_model: Type[BaseModel] | None = None
    system_prompt: str | None = None


class SingleLlmInput(BaseModel):
    input_data: InputData
    prompt_template: str | None = None
    output_model: Type[BaseModel] | None = None
    system_prompt: str | None = None

