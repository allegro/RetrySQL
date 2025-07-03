from __future__ import annotations

import sqlite3
from typing import Any

from upath import UPath
from transformers.tokenization_utils import PaddingStrategy

from text2sql.modules.schema_linking.configuration import (
    DocumentConfiguration,
    EmbeddingModelConfiguration,
    EmbeddingsJobConfiguration,
    QueryConfiguration,
)
from text2sql.llm_training.domain.enums import SupportedTrainingModes, PretrainingDataTypes
from text2sql.modules.schema_linking.domain.enums import AvailableEmbeddingModels
from text2sql.llms.configuration import (
    ModelConfiguration,
    GenerationConfiguration,
    PostProcessingConfiguration,
    OpenSourceLlmConfiguration
)


class DatabaseBuilder:
    def __init__(self, name: str):
        self._name = name
        self._sql_statements = []

    def with_table(
        self,
        table_name: str,
        column_names: list[str],
        column_types: list[str],
    ) -> DatabaseBuilder:
        self._sql_statements.append(
            f"""
            CREATE TABLE {table_name} (
                {
            ", ".join([f"{column_name} {column_type}" for column_name, column_type in zip(column_names, column_types)])
            }
            )
            """
        )

        return self

    def with_values(self, table_name: str, columns: list[str], values: list[Any]):
        wrapped_values = [f'"{value}"' for value in values]
        self._sql_statements.append(
            f"""
            INSERT INTO {table_name} ({",".join(columns)}) VALUES ({",".join(wrapped_values)})
            """
        )

        return self

    def build(self, storage_directory: UPath) -> None:
        storage_directory.mkdir(parents=True, exist_ok=True)

        database_connection = sqlite3.connect(storage_directory.joinpath(f"{self._name}.sqlite"))

        for sql_statement in self._sql_statements:
            database_connection.execute(sql_statement)

        database_connection.commit()
        database_connection.close()


class EmbeddingModelConfigurationBuilder:
    model_name: AvailableEmbeddingModels = AvailableEmbeddingModels.DUMMY_MODEL
    max_sequence_length: int = 512
    batch_size: int = 32
    embedding_size: int = 128
    path_to_endpoint_config: UPath = UPath(".")

    @staticmethod
    def default() -> EmbeddingModelConfigurationBuilder:
        return EmbeddingModelConfigurationBuilder()

    def with_model_name(self, name: AvailableEmbeddingModels) -> EmbeddingModelConfigurationBuilder:
        self.model_name = name
        return self

    def with_batch_size(self, batch_size: int) -> EmbeddingModelConfigurationBuilder:
        self.batch_size = batch_size
        return self

    def with_max_seq_length(self, max_seq_length: int) -> EmbeddingModelConfigurationBuilder:
        self.max_sequence_length = max_seq_length
        return self

    def with_embedding_size(self, size: int) -> EmbeddingModelConfigurationBuilder:
        self.embedding_size = size
        return self

    def build(self) -> EmbeddingModelConfiguration:
        return EmbeddingModelConfiguration(
            model_name=self.model_name,
            batch_size=self.batch_size,
            max_sequence_length=self.max_sequence_length,
            embedding_size=self.embedding_size,
            path_to_endpoint_config=self.path_to_endpoint_config,
        )


class QueryConfigurationBuilder:
    include_external_knowledge: bool = True
    max_neighbour_count: int = 1

    @staticmethod
    def default() -> QueryConfigurationBuilder:
        return QueryConfigurationBuilder()

    def with_max_neighbour_count(self, max_neighbour_count: int) -> QueryConfigurationBuilder:
        self.max_neighbour_count = max_neighbour_count
        return self

    def with_include_external_knowledge(self, include: bool) -> QueryConfigurationBuilder:
        self.include_external_knowledge = include
        return self

    def build(self) -> QueryConfiguration:
        return QueryConfiguration(
            include_external_knowledge=self.include_external_knowledge, max_neighbour_count=self.max_neighbour_count
        )


class DocumentConfigurationBuilder:
    include_column_description: bool = True
    include_column_data_format: bool = True
    include_column_values: bool = True
    include_only_values_representative: bool = False
    exclude_all_primary_and_foreign_keys: bool = False
    exclude_only_numerical_primary_and_foreign_keys: bool = False
    exclude_primary_and_foreign_keys_with_uuid_values: bool = False
    exclude_all_columns_with_id_in_the_name: bool = False

    @staticmethod
    def default() -> DocumentConfigurationBuilder:
        return DocumentConfigurationBuilder()

    def with_column_description(self, include: bool) -> DocumentConfigurationBuilder:
        self.include_column_description = include
        return self

    def with_column_data_format(self, include: bool) -> DocumentConfigurationBuilder:
        self.include_column_data_format = include
        return self

    def with_column_values(self, include: bool) -> DocumentConfigurationBuilder:
        self.include_column_values = include
        return self

    def with_only_column_values_representatives(self, include: bool) -> DocumentConfigurationBuilder:
        self.include_only_values_representative = include
        return self

    def with_exclude_all_primary_and_foreign_keys(self, exclude: bool) -> DocumentConfigurationBuilder:
        self.exclude_all_primary_and_foreign_keys = exclude
        return self

    def with_exclude_only_numerical_primary_and_foreign_keys(self, exclude: bool) -> DocumentConfigurationBuilder:
        self.exclude_only_numerical_primary_and_foreign_keys = exclude
        return self

    def with_exclude_primary_and_foreign_keys_with_uuid_values(self, exclude: bool) -> DocumentConfigurationBuilder:
        self.exclude_primary_and_foreign_keys_with_uuid_values = exclude
        return self

    def with_exclude_columns_with_id_in_name(self, exclude: bool) -> DocumentConfigurationBuilder:
        self.exclude_all_columns_with_id_in_the_name = exclude
        return self

    def build(self) -> DocumentConfiguration:
        return DocumentConfiguration(
            include_column_description=self.include_column_description,
            include_column_data_format=self.include_column_data_format,
            include_column_values=self.include_column_values,
            include_only_values_representative=self.include_only_values_representative,
            exclude_all_primary_and_foreign_keys=self.exclude_all_primary_and_foreign_keys,
            exclude_only_numerical_primary_and_foreign_keys=self.exclude_only_numerical_primary_and_foreign_keys,
            exclude_primary_and_foreign_keys_with_uuid_values=self.exclude_primary_and_foreign_keys_with_uuid_values,
            exclude_all_columns_with_id_in_the_name=self.exclude_all_columns_with_id_in_the_name,
        )


class EmbeddingsJobConfigurationBuilder:
    path_to_raw_queries: UPath = UPath(".")
    path_to_raw_documents: UPath = UPath(".")
    model_configuration: EmbeddingModelConfiguration = EmbeddingModelConfigurationBuilder.default().build()
    query_configuration: QueryConfiguration = QueryConfigurationBuilder.default().build()
    document_configuration: DocumentConfiguration = DocumentConfigurationBuilder.default().build()

    @staticmethod
    def default() -> EmbeddingsJobConfigurationBuilder:
        return EmbeddingsJobConfigurationBuilder()

    def with_path_to_raw_queries(self, path: UPath) -> EmbeddingsJobConfigurationBuilder:
        self.path_to_raw_queries = path
        return self

    def with_path_to_raw_documents(self, path: UPath) -> EmbeddingsJobConfigurationBuilder:
        self.path_to_raw_documents = path
        return self

    def with_model_configuration(self, config: EmbeddingModelConfiguration) -> EmbeddingsJobConfigurationBuilder:
        self.model_configuration = config
        return self

    def with_query_configuration(self, config: QueryConfiguration) -> EmbeddingsJobConfigurationBuilder:
        self.query_configuration = config
        return self

    def with_document_configuration(self, config: DocumentConfiguration) -> EmbeddingsJobConfigurationBuilder:
        self.document_configuration = config
        return self

    def build(self) -> EmbeddingsJobConfiguration:
        return EmbeddingsJobConfiguration(
            path_to_raw_queries=self.path_to_raw_queries,
            path_to_raw_documents=self.path_to_raw_documents,
            embedding_model_configuration=self.model_configuration,
            query_configuration=self.query_configuration,
            document_configuration=self.document_configuration,
        )

class OpenSourceLlmConfigurationBuilder:
    def __init__(self):
        self._model_name = "default-model"
        self._weights_path = None
        self._tokenizer_path = None
        self._max_model_context_length = 50
        self._padding_strategy = PaddingStrategy.MAX_LENGTH
        self._verbose = False
        self._pre_trained_model_mode = SupportedTrainingModes.PRETRAINING
        self._pretraining_data_type = PretrainingDataTypes.WITHOUT_REASONING

        self._max_new_tokens = 20
        self._batch_size = 1
        self._temperature = 0.0
        self._top_k = 50
        self._top_p = 0.95
        self._repetition_penalty = 1.0
        self._do_sample = False
        self._num_beams = 1

        self._trim_output_from_input_sequence = False
        self._add_select_statement_to_generated_sql = False
        self._normalize_generated_sql = False
        self._split_output_at_question = False

    def with_model_name(self, model_name: str):
        self._model_name = model_name
        return self

    def with_weights_path(self, weights_path: str | None):
        self._weights_path = UPath(weights_path) if weights_path else None
        return self

    def with_tokenizer_path(self, tokenizer_path: str | None):
        self._tokenizer_path = UPath(tokenizer_path) if tokenizer_path else None
        return self

    def with_max_model_context_length(self, max_length: int):
        self._max_model_context_length = max_length
        return self
    
    def with_padding_strategy(self, padding_strategy: PaddingStrategy):
        self._padding_strategy = padding_strategy
        return self

    def with_verbose(self, verbose: bool):
        self._verbose = verbose
        return self

    def with_pre_trained_model_mode(self, pre_trained_model_mode: SupportedTrainingModes):
        self._pre_trained_model_mode = pre_trained_model_mode
        return self

    def with_pre_training_data_type(self, pretraining_data_type: PretrainingDataTypes):
        self._pretraining_data_type = pretraining_data_type
        return self

    def with_generation_config(
        self,
        max_new_tokens: int = 20,
        batch_size: int = 1,
        temperature: float = 0.0,
        top_k: float = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        num_beams: int = 1
    ):
        self._max_new_tokens = max_new_tokens
        self._batch_size = batch_size
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty
        self._do_sample = do_sample
        self._num_beams = num_beams
        return self

    def with_post_processing_config(
        self,
        trim_output: bool = False,
        add_select: bool = False,
        normalize_sql: bool = False,
        split_output_at_question: bool = False,
    ):
        self._trim_output_from_input_sequence = trim_output
        self._add_select_statement_to_generated_sql = add_select
        self._normalize_generated_sql = normalize_sql
        self._split_output_at_question = split_output_at_question
        return self

    def build(self):
        model_config = ModelConfiguration(
            model_name=self._model_name,
            weights_path=self._weights_path,
            tokenizer_path=self._tokenizer_path,
            max_model_context_length=self._max_model_context_length,
            padding_strategy=self._padding_strategy,
            verbose=self._verbose,
            pre_trained_model_mode=self._pre_trained_model_mode,
            pretraining_data_type=self._pretraining_data_type,
        )

        generation_config = GenerationConfiguration(
            max_new_tokens=self._max_new_tokens,
            batch_size=self._batch_size,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
            repetition_penalty=self._repetition_penalty,
            do_sample=self._do_sample,
            num_beams=self._num_beams
        )

        post_processing_config = PostProcessingConfiguration(
            trim_output_from_input_sequence=self._trim_output_from_input_sequence,
            add_select_statement_to_the_generated_sql=self._add_select_statement_to_generated_sql,
            normalize_generated_sql=self._normalize_generated_sql,
            split_output_at_question=self._split_output_at_question,
        )

        return OpenSourceLlmConfiguration(
            model_configuration=model_config,
            generation_configuration=generation_config,
            post_processing_configuration=post_processing_config,
        )

