#!/bin/bash

if [[ "$#" -lt 2 ]]; then
    echo "usage: $0 PRETRAINED_MODEL_NAME_OR_PATH PRETRAINING_DATA_PATH"
    echo "example usage: $0 infly/OpenCoder-1.5B-Base resources/datasets/retry_data/pretrain_data.jsonl"
    exit 1
fi

PRETRAINED_MODEL_NAME_OR_PATH=${1}
PRETRAINING_DATA_PATH=${2}

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True accelerate launch --config-file configs/llm_training/accelerate/default_config.yaml \
  -m text2sql.llm_training.main \
  --pretrained-model-name-or-path "${PRETRAINED_MODEL_NAME_OR_PATH}" \
  --llm-training-data-path "${PRETRAINING_DATA_PATH}" \
  --training-mode PRETRAINING