import logging
import sys
from typing import Any


def setup_logger() -> None:
    log_format = "[%(levelname)s] %(asctime)s %(filename)s (%(lineno)d)\t- %(message)s"
    log_dateformat = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        format=log_format,
        datefmt=log_dateformat,
        stream=sys.stdout,
        level=logging.INFO,
    )


def get_logger(name: str | None = None) -> logging.Logger:
    setup_logger()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


def mask_sensitive_data(config: dict[str, Any], keys_to_mask: list[str]) -> dict[str, Any]:
    masked_config = config.copy()
    for key in keys_to_mask:
        if key in masked_config:
            masked_config[key] = "***"
    return masked_config
