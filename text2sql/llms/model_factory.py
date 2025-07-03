from dataclasses import asdict

from allms.domain.configuration import AzureOpenAIConfiguration, VertexAIConfiguration
from allms.models import AbstractModel, AzureOpenAIModel, VertexAIGeminiModel
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from transformers import AutoModelForCausalLM, AutoTokenizer
from upath import UPath
import torch

from text2sql.commons.io_utils import read_yaml, load_configuration_to_target_dataclass
from text2sql.commons.logging_utils import get_logger, mask_sensitive_data
from text2sql.llms.domain.enums import SupportedLlmProviders, SupportedLlms
from text2sql.llms.domain.models import LlmParams, OpenSourceLlm
from text2sql.llms.configuration import OpenSourceLlmConfigurationDto
from text2sql.settings import CONFIG_DIR

logger = get_logger(__name__)


class LlmModelFactory:
    @staticmethod
    def _get_cloud_based_llm(
            name: SupportedLlms,
            params: LlmParams,
            endpoint_config_path: UPath = CONFIG_DIR.joinpath("llm_endpoints.yaml"),
    ) -> AbstractModel:

        endpoint_config = read_yaml(endpoint_config_path)[name]
        endpoint_provider = endpoint_config.pop("provider")

        logger.info(f"Creating model instance `{name}` with params: {params.dict()}")

        match endpoint_provider:
            case SupportedLlmProviders.azure:
                endpoint = AzureOpenAIConfiguration(**endpoint_config)

                masked_config = mask_sensitive_data(asdict(endpoint), ["api_key", "azure_ad_token"])
                logger.info(f"Azure endpoint configuration: {masked_config}")

                return AzureOpenAIModel(config=endpoint, **params.dict())

            case SupportedLlmProviders.vertex_ai:
                endpoint = VertexAIConfiguration(**endpoint_config)
                # set safety settings for Gemini model not to block any response content
                endpoint.gemini_safety_settings = {
                    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                }
                logger.info(f"Vertex-AI endpoint configuration: {asdict(endpoint)}")

                return VertexAIGeminiModel(config=endpoint, **params.dict())

            case _:
                raise ValueError(
                    "Wrong endpoint provider: '%s'. Provide one of %s"
                    % (endpoint_provider, SupportedLlmProviders.get_values()),
                )

    @staticmethod
    def _get_open_source_llm(open_source_model_config_path: UPath, is_lora_model: bool = False) -> OpenSourceLlm:
        open_source_model_config = load_configuration_to_target_dataclass(
            path_to_config_yaml_file=open_source_model_config_path,
            target_dataclass=OpenSourceLlmConfigurationDto,
        )
        open_source_model_config = open_source_model_config.to_domain()

        if open_source_model_config.model_configuration.weights_path is None:
            logger.info(f"Loading model from HuggingFace model hub: {open_source_model_config.model_configuration.model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                open_source_model_config.model_configuration.model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                open_source_model_config.model_configuration.model_name,
                trust_remote_code=True
            )
        else:
            logger.info(f"Loading model from local path: {open_source_model_config.model_configuration.weights_path}")
            logger.info(f"Loading model from local path: {open_source_model_config.model_configuration.tokenizer_path}")
            model = AutoModelForCausalLM.from_pretrained(
                open_source_model_config.model_configuration.model_name if is_lora_model else
                open_source_model_config.model_configuration.weights_path,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )

            if is_lora_model:
                logger.info("Loading adapter weights")
                model.load_adapter(open_source_model_config.model_configuration.weights_path)
                model.enable_adapters()

            tokenizer = AutoTokenizer.from_pretrained(
                open_source_model_config.model_configuration.tokenizer_path,
                trust_remote_code=True
            )

        return OpenSourceLlm(llm=model, tokenizer=tokenizer)

    @staticmethod
    def create(
        name: SupportedLlms,
        params: LlmParams | None = None,
        endpoint_config_path: UPath | None = CONFIG_DIR.joinpath("llm_endpoints.yaml"),
        open_source_model_config_path: UPath | None = None,
        is_lora_model: bool = False,
    ) -> AbstractModel | OpenSourceLlm:

        match name:
            case SupportedLlms.opencoder_1_5b | SupportedLlms.opencoder_8b:
                return LlmModelFactory._get_open_source_llm(open_source_model_config_path, is_lora_model)
            case (SupportedLlms.gpt_4_32k | SupportedLlms.gpt_4o | SupportedLlms.gpt_4o_mini |
                  SupportedLlms.gemini_pro | SupportedLlms.gemini_flash
            ):
                return LlmModelFactory._get_cloud_based_llm(name, params, endpoint_config_path)
            case _:
                raise ValueError(
                    "Unsupported model name: '%s'. Provide one of %s" % (name, SupportedLlms.get_values())
                )
