import argparse
import json
from dataclasses import asdict

from faiss import Index, write_index
from upath import UPath

from text2sql.commons.io_utils import (
    copy_file,
    load_configuration_to_target_dataclass,
    read_jsonl,
    save_configuration_to_yaml,
    save_embeddings,
    save_objects_to_jsonl_file,
)
from text2sql.commons.logging_utils import get_logger
from text2sql.datasets.domain.model import SchemaLinkingDataSample
from text2sql.evaluation.metrics.schema_linking_metrics import SchemaLinkingMetrics
from text2sql.modules.schema_linking.configuration import SchemaLinkingExperimentConfigurationDto
from text2sql.modules.schema_linking.data_processors.data_processor_factory import SchemaLinkingDataProcessorFactory
from text2sql.modules.schema_linking.domain.enums import SupportedAlgorithms
from text2sql.modules.schema_linking.domain.model import Embedding, EmbeddingJobExample, QueryAndDocumentEmbeddings
from text2sql.modules.schema_linking.repository.index_builder import IndexBuilder
from text2sql.modules.schema_linking.schema_linkers.embedding_based.embedders.embbeder_base import EmbedderBase
from text2sql.modules.schema_linking.schema_linkers.embedding_based.embedders.embedder_factory import (
    EmbeddingModelFactory,
)
from text2sql.modules.schema_linking.schema_linkers.schema_linker_factory import SchemaLinkerFactory

logger = get_logger(__name__)


def save_index(index: Index, target_path: UPath) -> None:
    local_index_path = UPath("temp")
    local_index_path.mkdir(exist_ok=True, parents=True)
    temporary_file_path = local_index_path.joinpath("documents.index")
    write_index(index, str(temporary_file_path))
    copy_file(temporary_file_path, target_path)


def are_embeddings_already_dumped(embeddings_path: UPath) -> bool:
    return embeddings_path.exists()


def load_or_create_embeddings(
    embeddings_path: UPath, embedder: EmbedderBase, input_to_embed: list[EmbeddingJobExample], is_query: bool = False
) -> list[Embedding]:
    if are_embeddings_already_dumped(embeddings_path):
        logger.info(f"Loading {'queries' if is_query else 'documents'} embeddings from {embeddings_path}")
        return [Embedding(**embedding_dict) for embedding_dict in json.load(UPath.open(embeddings_path))]
    else:
        logger.info(f"Embedding {'queries' if is_query else 'documents'}...")
        embeddings = embedder.embed(input_to_embed, is_query=is_query)
        save_embeddings(embeddings, embeddings_path)
        return embeddings


def get_embeddings(query_embeddings_path: UPath, document_embeddings_path: UPath) -> QueryAndDocumentEmbeddings:
    embedder = EmbeddingModelFactory.get(schema_linking_exp_configuration.embedding_based_algorithm_configuration)
    query_embeddings = load_or_create_embeddings(
        query_embeddings_path, embedder, processor_output.processed_queries, is_query=True
    )
    document_embeddings = load_or_create_embeddings(
        document_embeddings_path, embedder, processor_output.processed_documents
    )
    return QueryAndDocumentEmbeddings(query_embeddings=query_embeddings, document_embeddings=document_embeddings)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--output-dir",
        type=UPath,
        help="Path where predictions will be saved",
        required=True,
    )
    argument_parser.add_argument(
        "--path-to-experiment-config-yaml",
        type=UPath,
        help="Path to schema linking experiment configuration YAML file",
        required=True,
    )
    argument_parser.add_argument(
        "--compute-metrics",
        action=argparse.BooleanOptionalAction,
        help="If set, schema linking metrics will be computed",
    )
    argument_parser.add_argument(
        "--ground-truth-path",
        type=UPath,
        help="Ground truth data file",
        default=UPath("resources/schema_links/ground_truth_schema_links_bird_dev.jsonl"),
    )
    args = argument_parser.parse_args()

    schema_linking_exp_configuration_dto: SchemaLinkingExperimentConfigurationDto = (
        load_configuration_to_target_dataclass(
            args.path_to_experiment_config_yaml,
            target_dataclass=SchemaLinkingExperimentConfigurationDto,
        )
    )
    schema_linking_exp_configuration = schema_linking_exp_configuration_dto.to_domain()

    save_configuration_to_yaml(args.output_dir.joinpath("config.yaml"), asdict(schema_linking_exp_configuration))

    schema_linker = SchemaLinkerFactory().get(schema_linking_exp_configuration)
    data_processor = SchemaLinkingDataProcessorFactory().get(schema_linking_exp_configuration)
    processor_output = data_processor.process()

    if schema_linking_exp_configuration.schema_linking_algorithm_name == SupportedAlgorithms.NEAREST_NEIGHBOUR_SEARCH:
        embedding_job_config = schema_linking_exp_configuration.embedding_based_algorithm_configuration
        embedding_job_config.embedding_model_configuration.staging_output_dir = args.output_dir.joinpath("staging_dir")

        query_embeddings_path = args.output_dir.joinpath("query_embeddings.json")
        document_embeddings_path = args.output_dir.joinpath("document_embeddings.json")

        embedding_output = get_embeddings(
            query_embeddings_path=query_embeddings_path, document_embeddings_path=document_embeddings_path
        )

        search_index = (
            IndexBuilder(embedding_size=embedding_job_config.embedding_model_configuration.embedding_size)
            .add_embeddings(embedding_output.document_embeddings)
            .build()
        )

        save_index(search_index, args.output_dir.joinpath("documents.index"))

        predictions = schema_linker.forward(
            embedded_queries=embedding_output.query_embeddings,
            raw_queries=processor_output.processed_queries,
            all_documents=processor_output.raw_documents,
            search_index=search_index,
            max_neighbour_count=embedding_job_config.query_configuration.max_neighbour_count,
        )
    else:
        predictions = schema_linker.forward(processor_output)

    save_objects_to_jsonl_file(
        objs=[prediction.dict() for prediction in predictions],
        file_path=args.output_dir.joinpath("predictions.jsonl"),
    )

    if args.compute_metrics:
        ground_truths = read_jsonl(
            args.ground_truth_path,
            class_schema=SchemaLinkingDataSample,
        )

        schema_linking_metrics = SchemaLinkingMetrics(
            predictions=predictions,
            ground_truths=ground_truths,
        )

        schema_linking_metrics.compute_metrics()
