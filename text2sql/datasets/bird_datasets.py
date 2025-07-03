from typing import Type, TypeVar

from pydantic import BaseModel
from upath import UPath

from text2sql.commons.io_utils import read_json, read_jsonl
from text2sql.commons.logging_utils import get_logger
from text2sql.datasets.base_dataset import BaseDataset
from text2sql.datasets.domain.model import (
    BirdDevDataSample,
    BirdTrainDataSample,
    SchemaLinkingDataSample,
    Text2SqlDataSample,
)
from text2sql.settings import DATASETS_DIR

logger = get_logger(__name__)

TypeText2SqlModel = TypeVar("TypeText2SqlModel", bound=Text2SqlDataSample)


class BirdDataset(BaseDataset):

    def load_dev(
        self,
        db_root_dirpath: UPath = DATASETS_DIR.joinpath("BIRD_dev", "dev_databases"),
        samples_path: UPath = DATASETS_DIR.joinpath("BIRD_dev", "dev.json"),
        model_cls: Type[TypeText2SqlModel] = BirdDevDataSample,
    ) -> list[TypeText2SqlModel]:
        dev_data = list(
            map(
                lambda sample: model_cls(
                    database_path=db_root_dirpath.joinpath(sample["db_id"], f"{sample['db_id']}.sqlite"),
                    question_id=sample["question_id"],
                    database_name=sample["db_id"],
                    question=sample["question"],
                    sql_query=sample["SQL"],
                    knowledge=sample["evidence"],
                    difficulty=sample["difficulty"],
                ),
                read_json(path=samples_path),
            )
        )
        logger.info(f"Loaded {len(dev_data)} dev data samples")
        return dev_data

    def load_train(
        self,
        db_root_dirpath: UPath = DATASETS_DIR.joinpath("BIRD_train", "train_databases"),
        samples_path: UPath = DATASETS_DIR.joinpath("BIRD_train", "train.json"),
        model_cls: Type[TypeText2SqlModel] = BirdTrainDataSample,
    ) -> list[TypeText2SqlModel]:
        train_data = read_json(path=samples_path)
        train_data = [
            model_cls(
                database_path=db_root_dirpath.joinpath(sample["db_id"], f"{sample['db_id']}.sqlite"),
                question_id=idx,  # train data have no question_id in BIRD
                database_name=sample["db_id"],
                question=sample["question"],
                sql_query=sample["SQL"],
                knowledge=sample["evidence"],
            )
            for idx, sample in enumerate(train_data)
        ]
        logger.info(f"Loaded {len(train_data)} train data samples")
        return train_data

    def load_test(self) -> list[BaseModel]:
        pass


class BirdSchemaLinkDataset(BaseDataset):

    def load_dev(
        self,
        samples_path: UPath = DATASETS_DIR.joinpath("BIRD_dev", "schema_linking_dataset.jsonl"),
        model_cls: Type[SchemaLinkingDataSample] = SchemaLinkingDataSample,
    ) -> list[SchemaLinkingDataSample]:
        dev_data = read_jsonl(samples_path, class_schema=model_cls)
        logger.info(f"Loaded {len(dev_data)} dev data samples")

        return dev_data

    def load_train(
        self,
        samples_path: UPath = DATASETS_DIR.joinpath("BIRD_train", "schema_linking_dataset.jsonl"),
        model_cls: Type[SchemaLinkingDataSample] = SchemaLinkingDataSample,
    ) -> list[SchemaLinkingDataSample]:
        train_data = read_jsonl(samples_path, class_schema=model_cls)
        logger.info(f"Loaded {len(train_data)} train data samples")

        return train_data

    def load_test(self) -> list[BaseModel]:
        pass
