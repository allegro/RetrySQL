from text2sql.modules.schema_linking.configuration import SchemaLinkingExperimentConfiguration
from text2sql.modules.schema_linking.data_processors.base_processor import BaseSchemaLinkingDataProcessor
from text2sql.modules.schema_linking.data_processors.heuristic_based_methods_data_processor import (
    HeuristicBasedMethodsDataProcessor,
)
from text2sql.modules.schema_linking.data_processors.llm_based_methods_data_processor import LlmBasedMethodDataProcessor
from text2sql.modules.schema_linking.data_processors.query_and_document_processor import QueryAndDocumentProcessor
from text2sql.modules.schema_linking.domain.enums import SupportedAlgorithms


class SchemaLinkingDataProcessorFactory:
    @staticmethod
    def get(config: SchemaLinkingExperimentConfiguration) -> BaseSchemaLinkingDataProcessor | None:
        match config.schema_linking_algorithm_name:
            case SupportedAlgorithms.EDIT_DISTANCE | SupportedAlgorithms.EXACT_MATCHING:
                return HeuristicBasedMethodsDataProcessor(config=config.heuristic_based_algorithm_configuration)
            case SupportedAlgorithms.NEAREST_NEIGHBOUR_SEARCH:
                return QueryAndDocumentProcessor(config=config.embedding_based_algorithm_configuration)
            case SupportedAlgorithms.LLM_BASED:
                return LlmBasedMethodDataProcessor(config=config.llm_based_algorithm_configuration)
            case _:
                raise ValueError(
                    f"No schema linking data processor configured for {config.schema_linking_algorithm_name}"
                )
