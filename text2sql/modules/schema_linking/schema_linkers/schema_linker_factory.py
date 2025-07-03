from text2sql.llms.domain.baseline_params import get_baseline_llm_params
from text2sql.llms.model_factory import LlmModelFactory
from text2sql.modules.schema_linking.configuration import SchemaLinkingExperimentConfiguration
from text2sql.modules.schema_linking.domain.enums import SupportedAlgorithms
from text2sql.modules.schema_linking.schema_linkers.embedding_based.schema_linker_nn_search import (
    SchemaLinkerNearestNeighborSearch,
)
from text2sql.modules.schema_linking.schema_linkers.heuristic_based.schema_linker_edit_distance import (
    SchemaLinkerEditDistance,
)
from text2sql.modules.schema_linking.schema_linkers.llm_based.schema_linker_llm_table_and_column import (
    LlmSchemaLinkerTableAndColumn,
)
from text2sql.modules.schema_linking.schema_linkers.schema_linker_base import SchemaLinkerBase


class SchemaLinkerFactory:
    @staticmethod
    def get(
        experiment_configuration: SchemaLinkingExperimentConfiguration,
    ) -> SchemaLinkerBase:
        match experiment_configuration.schema_linking_algorithm_name:
            case SupportedAlgorithms.EXACT_MATCHING:
                algorithm_config = experiment_configuration.heuristic_based_algorithm_configuration
                return SchemaLinkerEditDistance(
                    use_external_knowledge=algorithm_config.use_external_knowledge,
                    edit_distance=0,
                    min_token_length=algorithm_config.min_token_length,
                )
            case SupportedAlgorithms.EDIT_DISTANCE:
                algorithm_config = experiment_configuration.heuristic_based_algorithm_configuration
                return SchemaLinkerEditDistance(
                    use_external_knowledge=algorithm_config.use_external_knowledge,
                    edit_distance=algorithm_config.edit_distance,
                    min_token_length=algorithm_config.min_token_length,
                )
            case SupportedAlgorithms.NEAREST_NEIGHBOUR_SEARCH:
                algorithm_config = experiment_configuration.embedding_based_algorithm_configuration
                return SchemaLinkerNearestNeighborSearch(
                    use_external_knowledge=algorithm_config.query_configuration.include_external_knowledge
                )
            case SupportedAlgorithms.LLM_BASED:
                algorithm_config = experiment_configuration.llm_based_algorithm_configuration
                llm = LlmModelFactory.create(
                    name=algorithm_config.llm, params=get_baseline_llm_params(algorithm_config.llm)
                )
                return LlmSchemaLinkerTableAndColumn(
                    llm=llm, use_cot=algorithm_config.use_cot, use_knowledge=algorithm_config.use_knowledge
                )
            case _:
                raise ValueError(
                    f"No schema-linker-type defined for {experiment_configuration.schema_linking_algorithm_name}"
                )
