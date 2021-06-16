#!/usr/bin/env bash

set -e

local_repo=${LOCAL_REPO?Variable LOCAL_REPO is required}
model_name=${MODEL_NAME?Variable MODEL_NAME is required}
dimensions=${DIMENSIONS?Variable DIMENSIONS is required}

docker build --build-arg "MODEL_NAME=$model_name" --build-arg "DIMENSIONS=$dimensions" -t "$local_repo" .
