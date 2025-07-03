from text2sql.modules.schema_linking.configuration import EmbeddingsJobConfiguration
from text2sql.modules.schema_linking.domain.enums import AvailableEmbeddingModels
from text2sql.modules.schema_linking.schema_linkers.embedding_based.embedders.embbeder_base import (
    DummyEmbedder,
    EmbedderBase,
)
from text2sql.modules.schema_linking.schema_linkers.embedding_based.embedders.embedder_arctic_embed import (
    EmbedderHuggingFaceTransformers,
)
from text2sql.modules.schema_linking.schema_linkers.embedding_based.embedders.embedder_openai import EmbedderOpenAI


class EmbeddingModelFactory:
    @staticmethod
    def get(config: EmbeddingsJobConfiguration) -> EmbedderBase:
        model_name = AvailableEmbeddingModels(config.embedding_model_configuration.model_name)

        match model_name:
            case AvailableEmbeddingModels.DUMMY_MODEL:
                return DummyEmbedder(config)
            case AvailableEmbeddingModels.TEXT_EMBEDDING_ADA_002 | AvailableEmbeddingModels.TEXT_EMBEDDING_3_LARGE:
                return EmbedderOpenAI(config)
            case (
                AvailableEmbeddingModels.SNOWFLAKE_ARCTIC_EMBED_M
                | AvailableEmbeddingModels.SNOWFLAKE_ARCTIC_EMBED_M_LONG
                | AvailableEmbeddingModels.SNOWFLAKE_ARCTIC_EMBED_L
            ):
                return EmbedderHuggingFaceTransformers(config)
            case _:
                raise ValueError(f"Unknown embedder model for the name: {model_name}")
