from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field
from sqlglot import Expression
from upath import UPath

from text2sql.datasets.domain.enums import NumCorruptionsPerStepType, StepsToUseForCorruptionsType
from text2sql.llms.domain.enums import SupportedLlms


class ColumnDescriptor(BaseModel):
    original_column_name: str
    column_description: str
    data_format: str


class SchemaColumn(BaseModel):
    column_name: str
    is_primary_key: bool = False
    is_foreign_key: bool = False


class DatabaseColumn(BaseModel):
    database_name: str
    table_name: str
    column: SchemaColumn


class DatabaseSchema(BaseModel):
    database_name: str
    tables: dict[str, list[SchemaColumn]]


class SchemaLink(BaseModel):
    table_name: str = Field(description="Table name")
    columns: list[str] = Field(description="List of column names")


class TableColumnPair(BaseModel):
    table_name: str = Field(description="Table name")
    expanded_column: SchemaColumn


class TableWithAlias(BaseModel):
    name: str
    alias: str | None = None


# Dataset samples
class Text2SqlDataSample(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    question_id: int
    database_name: str
    question: str
    sql_query: str
    knowledge: str | None = None  # this field is present in both Bird and Spider_2-lite datasets
    database_path: UPath | None = None
    database_schema: str | None = None


class BirdDevDataSample(Text2SqlDataSample):
    difficulty: str


class BirdTrainDataSample(Text2SqlDataSample): ...


class BirdTrainWithReasoningStepsDataSample(BirdTrainDataSample):
    reasoning_steps: list[str]


class SchemaLinkingDataSample(BaseModel):
    question_id: int
    question: str
    schema_links: list[SchemaLink]


class SchemaLinkingQueryRawData(BaseModel):
    question_id: int
    question: str
    external_knowledge: str


class SchemaLinkingDocumentRawData(BaseModel):
    document_id: int
    table_name: str
    col_name: str
    is_primary_key: bool
    is_foreign_key: bool
    col_description: str = ""
    col_data_format: str = ""
    col_values: list[Any] | None = Field(default=None, allow_none=True)


@dataclass
class CorruptedSample:
    question_id: int
    database_name: str
    question: str
    knowledge: str
    sql_query: str
    reasoning_steps: list[str] | None = None


@dataclass
class PretrainDataSample:
    question_id: int
    prompt: str
    query: str | None = None
    reasoning_steps: list[str] | None = None
    question: str | None = None


@dataclass
class RetryDataGenerationConfig:
    class Config:
        arbitrary_types_allowed = True

    bird_databases_path: UPath
    ground_truth_path: UPath
    bird_metadata_path: UPath
    error_probability: float
    multiply_factor: int
    num_corruptions_per_step: NumCorruptionsPerStepType
    steps_to_use_for_corruptions: StepsToUseForCorruptionsType
    output_directory: UPath
    databases_to_skip: list[str]


@dataclass
class ExpressionPair:
    original_expression: Expression
    corrupted_expression: Expression


@dataclass
class ReplacementPair:
    source_substring: str
    target_substring: str


@dataclass
class DatabaseTablesAndColumns:
    tables: list[str]
    columns: list[str]


@dataclass
class ReasoningStepsGenerationConfig:
    bird_dataset_path: UPath
    llm: SupportedLlms
    output_dir: UPath
    limit: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> ReasoningStepsGenerationConfig:
        dict_args = {field: getattr(args, field) for field in cls.__dataclass_fields__ if hasattr(args, field)}
        return cls(**dict_args)