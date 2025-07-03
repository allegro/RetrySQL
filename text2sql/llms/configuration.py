from dataclasses import dataclass

from transformers.tokenization_utils import PaddingStrategy
from upath import UPath

from text2sql.commons.io_utils import translate_gcs_dir_to_local
from text2sql.commons.logging_utils import get_logger
from text2sql.llm_training.domain.enums import SupportedTrainingModes, PretrainingDataTypes

logger = get_logger(__name__)

@dataclass
class ModelConfiguration:
    model_name: str
    weights_path: UPath | None
    tokenizer_path: UPath | None

    pre_trained_model_mode: SupportedTrainingModes
    pretraining_data_type: PretrainingDataTypes

    max_model_context_length: int
    padding_strategy: PaddingStrategy
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.weights_path and self.weights_path.protocol == "gs":
             self.weights_path = translate_gcs_dir_to_local(self.weights_path)
        if self.tokenizer_path and self.tokenizer_path.protocol == "gs":
            self.tokenizer_path = translate_gcs_dir_to_local(self.tokenizer_path)

@dataclass
class ModelConfigurationDto:
    model_name: str
    weights_path: str | None
    tokenizer_path: str | None

    pre_trained_model_mode: str
    pretraining_data_type: str

    max_model_context_length: int
    padding_strategy: PaddingStrategy
    verbose: bool = False

    def to_domain(self) -> ModelConfiguration:
        return ModelConfiguration(
            model_name=self.model_name,
            weights_path=UPath(self.weights_path) if self.weights_path is not None else None,
            tokenizer_path=UPath(self.tokenizer_path) if self.tokenizer_path is not None else None,
            max_model_context_length=self.max_model_context_length,
            padding_strategy=self.padding_strategy,
            verbose=self.verbose,
            pre_trained_model_mode=SupportedTrainingModes(self.pre_trained_model_mode),
            pretraining_data_type=PretrainingDataTypes(self.pretraining_data_type)
        )

# Arguments that might influence the generation process
# source: https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
@dataclass
class GenerationConfiguration:
    max_new_tokens: int
    batch_size: int

    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float

    do_sample: bool
    num_beams: int

@dataclass
class PostProcessingConfiguration:
    trim_output_from_input_sequence: bool
    add_select_statement_to_the_generated_sql: bool
    normalize_generated_sql: bool
    split_output_at_question: bool


@dataclass
class OpenSourceLlmConfiguration:
    model_configuration: ModelConfiguration
    generation_configuration: GenerationConfiguration
    post_processing_configuration: PostProcessingConfiguration
    output_path: UPath | None = None

    def validate_configuration(self) -> None:
        generation_configuration = self.generation_configuration

        assert generation_configuration.batch_size == 1, "Only batch_size = 1 is supported for the moment"

        decoding_strategy = (
            "Greedy Search"
            if not generation_configuration.do_sample
            else "Multinomial Sampling"
        )
        if generation_configuration.num_beams > 1:
            decoding_strategy = (
                "Beam Search" if not generation_configuration.do_sample else "Beam Search Multinomial Sampling"
            )

        logger.info(
            f"Decoding strategy: {decoding_strategy} with {generation_configuration.num_beams} beam(s) "
            f"and temperature: {generation_configuration.temperature}"
        )


@dataclass
class OpenSourceLlmConfigurationDto:
    model_configuration: ModelConfigurationDto
    generation_configuration: GenerationConfiguration
    post_processing_configuration: PostProcessingConfiguration
    output_path: UPath | None = None

    def to_domain(self) -> OpenSourceLlmConfiguration:
        return OpenSourceLlmConfiguration(
            model_configuration=self.model_configuration.to_domain(),
            generation_configuration=self.generation_configuration,
            post_processing_configuration=self.post_processing_configuration,
            output_path=self.output_path
        )
