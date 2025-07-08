#!/usr/bin/env bash

set -eou pipefail

local_repo=${LOCAL_REPO?Variable LOCAL_REPO is required}
model_name=${MODEL_NAME?Variable MODEL_NAME is required}
onnx_runtime=${ONNX_RUNTIME?Variable ONNX_RUNTIME is required}
trust_remote_code=${TRUST_REMOTE_CODE:-false}
use_sentence_transformers_vectorizer=${USE_SENTENCE_TRANSFORMERS_VECTORIZER:-false}
use_query_passage_prefixes=${USE_QUERY_PASSAGE_PREFIXES:-false}
use_query_prompt=${USE_QUERY_PROMPT:-false}

docker build \
  --build-arg "MODEL_NAME=$model_name" \
  --build-arg "ONNX_RUNTIME=$onnx_runtime" \
  --build-arg "TRUST_REMOTE_CODE=$trust_remote_code" \
  --build-arg "USE_SENTENCE_TRANSFORMERS_VECTORIZER=$use_sentence_transformers_vectorizer" \
  --build-arg "USE_QUERY_PASSAGE_PREFIXES=$use_query_passage_prefixes" \
  --build-arg "USE_QUERY_PROMPT=$use_query_prompt" \
  -t "$local_repo" .
