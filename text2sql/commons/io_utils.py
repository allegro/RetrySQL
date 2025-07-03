import json
import shutil
import subprocess
from typing import Any, Type, TypeVar
from upath import UPath

import gcsfs
import marshmallow_dataclass
import numpy as np
import yaml
from allms.domain.response import ResponseData
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from text2sql.commons.list_enum import ListConvertableEnum
from text2sql.commons.logging_utils import get_logger
from text2sql.modules.schema_linking.domain.model import Embedding

logger = get_logger(__name__)

TBaseModel = TypeVar("TBaseModel", bound=BaseModel)


def read_yaml(filepath: UPath) -> dict:
    logger.info(f"Reading YAML file from {filepath}")

    with filepath.open(mode="r") as f:
        yaml_data = yaml.safe_load(f)

    return yaml_data


def save_configuration_to_yaml(filepath: UPath, config_dict: dict) -> None:
    def _serialize_data_for_yaml(data: Any) -> dict | str:
        if isinstance(data, dict):
            return {k: _serialize_data_for_yaml(v) for k, v in data.items()}
        elif isinstance(data, UPath):
            return str(data)
        elif isinstance(data, ListConvertableEnum):
            return data.name
        else:
            return data

    logger.info(f"Writing configuration file to {filepath}")

    serializable_data = _serialize_data_for_yaml(config_dict)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open(mode="w") as f:
        yaml.safe_dump(serializable_data, f, default_flow_style=False, sort_keys=False)


def load_configuration_to_target_dataclass(
    path_to_config_yaml_file: UPath, target_dataclass: Type[TBaseModel]
) -> TBaseModel:
    config_yml = read_yaml(path_to_config_yaml_file)
    return marshmallow_dataclass.class_schema(target_dataclass)().load(config_yml)


def read_jsonl(path: UPath, class_schema: Type[TBaseModel]) -> list[TBaseModel]:
    logger.info(f"Reading JSONL file from {path}")

    data_samples = []
    with path.open("r", encoding="utf-8") as json_file:
        for line in json_file:
            line = line.strip()
            try:
                json_data = json.loads(line)
            except json.JSONDecodeError as error:
                logger.error(f"Error parsing line: {error}")
                raise ValueError(f"JSON reading error with object: {line}") from error

            read_obj = parse_json_obj(json_data, class_schema)
            data_samples.append(read_obj)

    return data_samples


def read_json(path: UPath) -> Any:
    logger.info(f"Reading JSON file from {path}")

    with path.open(mode="r") as f:
        json_data = json.load(f)

    return json_data


def parse_json_obj(json_obj: dict[str, Any], class_schema: Type[TBaseModel]) -> TBaseModel:
    try:
        parsed_obj = class_schema.parse_obj(json_obj) if class_schema else json_obj
    except ValidationError as parsing_error:
        logger.error(parsing_error)
        raise ValueError(f"Error parsing json data: {json_obj}") from parsing_error

    return parsed_obj


def save_objects_to_jsonl_file(objs: Any, file_path: UPath) -> None:
    logger.info(f"Saving objects to: {file_path}")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open(mode="w") as output_file:
        for current_object in objs:
            json.dump(current_object, output_file, ensure_ascii=False)
            output_file.write("\n")


def save_predictions(
    predictions: list[ResponseData],
    output_dir: UPath,
    output_file_name: str,
) -> None:
    def _prepare_prediction(prediction: ResponseData) -> dict[str, str]:
        return ResponseData(
            response=prediction.response,
            input_data=prediction.input_data,
            number_of_generated_tokens=prediction.number_of_generated_tokens,
            number_of_prompt_tokens=prediction.number_of_prompt_tokens,
            error=str(prediction.error),
        ).dict()

    predictions_dict = list(map(lambda prediction: _prepare_prediction(prediction), predictions))

    output_dir.mkdir(exist_ok=True, parents=True)
    save_objects_to_jsonl_file(predictions_dict, output_dir.joinpath(output_file_name))


def save_embeddings(embeddings: list[Embedding], filepath: UPath) -> None:
    embeddings_json = [
        {
            "embedding": (
                embedding.embedding.tolist() if isinstance(embedding.embedding, np.ndarray) else embedding.embedding
            ),
            "example_id": embedding.example_id,
        }
        for embedding in embeddings
    ]
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w") as json_file:
        json.dump(embeddings_json, json_file)


def copy_dir(source_dir: UPath, target_dir: UPath) -> None:
    logger.info(f"Copying dir {source_dir} to {target_dir}")

    target_dir.mkdir(exist_ok=True, parents=True)

    for source_file in source_dir.iterdir():
        target_item = target_dir.joinpath(source_file.name)

        if source_file.is_dir():
            copy_dir(source_file, target_item)
        else:
            file_size = source_file.stat().st_size
            with source_file.open(mode="rb") as source, target_item.open(mode="wb") as target:
                with tqdm(
                    total=file_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Copying {source_file.name}",
                ) as progress_bar:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                        progress_bar.update(len(chunk))


def copy_file(source_file: UPath, target_path: UPath) -> None:
    logger.info(f"Copying file from {source_file} to {target_path}")

    with source_file.open(mode="rb") as source, target_path.open(mode="wb") as target:
        content = source.read()
        target.write(content)


def translate_gcs_dir_to_local(path: UPath) -> UPath:
    if isinstance(path.fs, gcsfs.core.GCSFileSystem) and path.is_dir():
        local_path = UPath("temp/translated").joinpath(path.name)

        if local_path.exists() and local_path.is_dir():
            logger.info(f"Directory {local_path} already exists. Returning the existing directory.")
            return local_path

        logger.info(f"Directory {local_path} does not exist. Copying from {path}.")
        local_path.mkdir(exist_ok=True, parents=True)

        copy_dir(path, local_path)

        return local_path

    return path


def translate_gcs_file_to_local(path: UPath) -> UPath:
    if isinstance(path.fs, gcsfs.core.GCSFileSystem) and path.is_file():
        local_path = UPath("temp/translated").joinpath(path.name)
        local_path.parent.mkdir(exist_ok=True, parents=True)

        copy_file(path, local_path)

        return local_path

    return path


def copy_local_dir_to_gcp_dir(local_src_dir: UPath, gcp_dst_dir: UPath) -> None:
    gcp_dst_dir = f"{gcp_dst_dir}/"
    try:
        subprocess.run(
            ["gsutil", "-q", "-m", "cp", "-r", str(local_src_dir), gcp_dst_dir], check=True
        )
        logger.info(f"Uploaded {local_src_dir} to {gcp_dst_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Error while uploading {local_src_dir}: error code: {e.returncode}; "
            f"output: {e.output}"
        )


def remove_local_dir(local_dir_path: UPath) -> None:
    logger.info(f"Removing local directory: {local_dir_path}")
    shutil.rmtree(local_dir_path)
