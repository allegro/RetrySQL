from __future__ import annotations

from dataclasses import dataclass

from upath import UPath

from text2sql.llms.domain.enums import SupportedLlms
from text2sql.modules.schema_linking.domain.enums import AvailableEmbeddingModels, SupportedAlgorithms


@dataclass
class EmbeddingModelConfiguration:
    model_name: AvailableEmbeddingModels
    batch_size: int
    embedding_size: int
    max_sequence_length: int
    chunk_size: int | None = None
    staging_output_dir: UPath | None = None
    path_to_endpoint_config: UPath | None = None
    num_workers: int | None = None


@dataclass
class EmbeddingModelConfigurationDto:
    model_name: str
    batch_size: int
    embedding_size: int
    chunk_size: int | None = None
    staging_output_dir: str | None = None
    max_sequence_length: int | None = None
    path_to_endpoint_config: str | None = None
    num_workers: int | None = None

    def to_domain(self) -> EmbeddingModelConfiguration:
        return EmbeddingModelConfiguration(
            model_name=AvailableEmbeddingModels(self.model_name),
            batch_size=self.batch_size,
            embedding_size=self.embedding_size,
            chunk_size=self.chunk_size if self.chunk_size else None,
            staging_output_dir=UPath(self.staging_output_dir) if self.staging_output_dir else None,
            max_sequence_length=self.max_sequence_length,
            path_to_endpoint_config=UPath(self.path_to_endpoint_config) if self.path_to_endpoint_config else None,
            num_workers=self.num_workers if self.num_workers else None,
        )


@dataclass
class QueryConfiguration:
    include_external_knowledge: bool
    max_neighbour_count: int


@dataclass
class DocumentConfiguration:
    include_column_description: bool
    include_column_data_format: bool
    include_column_values: bool
    include_only_values_representative: bool

    exclude_all_primary_and_foreign_keys: bool
    exclude_only_numerical_primary_and_foreign_keys: bool
    exclude_primary_and_foreign_keys_with_uuid_values: bool
    exclude_all_columns_with_id_in_the_name: bool


@dataclass
class EmbeddingsJobConfiguration:
    path_to_raw_queries: UPath
    path_to_raw_documents: UPath

    embedding_model_configuration: EmbeddingModelConfiguration
    query_configuration: QueryConfiguration
    document_configuration: DocumentConfiguration


@dataclass
class EmbeddingsJobConfigurationDto:
    path_to_raw_queries: str
    path_to_raw_documents: str

    embedding_model_configuration: EmbeddingModelConfigurationDto
    query_configuration: QueryConfiguration
    document_configuration: DocumentConfiguration

    def to_domain(self) -> EmbeddingsJobConfiguration:
        return EmbeddingsJobConfiguration(
            path_to_raw_queries=UPath(self.path_to_raw_queries),
            path_to_raw_documents=UPath(self.path_to_raw_documents),
            embedding_model_configuration=self.embedding_model_configuration.to_domain(),
            query_configuration=self.query_configuration,
            document_configuration=self.document_configuration,
        )


@dataclass
class HeuristicBasedAlgorithmConfigurationDto:
    ground_truth_path: str
    bird_metadata_path: str

    edit_distance: int
    min_token_length: int
    use_external_knowledge: bool

    def to_domain(self) -> HeuristicBasedAlgorithmConfiguration:
        return HeuristicBasedAlgorithmConfiguration(
            ground_truth_path=UPath(self.ground_truth_path),
            bird_metadata_path=UPath(self.bird_metadata_path),
            edit_distance=self.edit_distance,
            min_token_length=self.min_token_length,
            use_external_knowledge=self.use_external_knowledge,
        )


@dataclass
class HeuristicBasedAlgorithmConfiguration:
    ground_truth_path: UPath
    bird_metadata_path: UPath

    edit_distance: int
    min_token_length: int
    use_external_knowledge: bool


@dataclass
class LlmBasedAlgorithmConfigurationDto:
    llm: str
    use_cot: bool
    use_knowledge: bool

    path_to_bird_dev: str
    db_root_dirpath: str

    def to_domain(self) -> LlmBasedAlgorithmConfiguration:
        return LlmBasedAlgorithmConfiguration(
            llm=SupportedLlms(self.llm),
            use_cot=self.use_cot,
            use_knowledge=self.use_knowledge,
            path_to_bird_dev=UPath(self.path_to_bird_dev),
            db_root_dirpath=UPath(self.db_root_dirpath),
        )


@dataclass
class LlmBasedAlgorithmConfiguration:
    llm: SupportedLlms
    use_cot: bool
    use_knowledge: bool

    path_to_bird_dev: UPath
    db_root_dirpath: UPath


@dataclass
class SchemaLinkingExperimentConfiguration:
    schema_linking_algorithm_name: SupportedAlgorithms

    heuristic_based_algorithm_configuration: HeuristicBasedAlgorithmConfiguration | None = None
    embedding_based_algorithm_configuration: EmbeddingsJobConfiguration | None = None
    llm_based_algorithm_configuration: LlmBasedAlgorithmConfigurationDto | None = None


@dataclass
class SchemaLinkingExperimentConfigurationDto:
    schema_linking_algorithm_name: str

    heuristic_based_algorithm_configuration: HeuristicBasedAlgorithmConfigurationDto | None = None
    embedding_based_algorithm_configuration: EmbeddingsJobConfigurationDto | None = None
    llm_based_algorithm_configuration: LlmBasedAlgorithmConfigurationDto | None = None

    def to_domain(self) -> SchemaLinkingExperimentConfiguration:
        return SchemaLinkingExperimentConfiguration(
            schema_linking_algorithm_name=SupportedAlgorithms(self.schema_linking_algorithm_name),
            heuristic_based_algorithm_configuration=(
                self.heuristic_based_algorithm_configuration.to_domain()
                if self.heuristic_based_algorithm_configuration
                else None
            ),
            embedding_based_algorithm_configuration=(
                self.embedding_based_algorithm_configuration.to_domain()
                if self.embedding_based_algorithm_configuration
                else None
            ),
            llm_based_algorithm_configuration=(
                self.llm_based_algorithm_configuration.to_domain() if self.llm_based_algorithm_configuration else None
            ),
        )
