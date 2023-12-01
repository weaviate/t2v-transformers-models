#!/usr/bin/env bash

set -eou pipefail

local_repo=${LOCAL_REPO?Variable LOCAL_REPO is required}
model_name=${MODEL_NAME?Variable MODEL_NAME is required}
onnx_runtime=${ONNX_RUNTIME?Variable ONNX_RUNTIME is required}

docker build \
  --build-arg "MODEL_NAME=$model_name" \
  --build-arg "ONNX_RUNTIME=$onnx_runtime" \
  -t "$local_repo" .
