#!/usr/bin/env bash

set -eou pipefail

remote_repo=${REMOTE_REPO?Variable REMOTE_REPO is required}
model_name=${MODEL_NAME?Variable MODEL_NAME is required}
docker_username=${DOCKER_USERNAME?Variable DOCKER_USERNAME is required}
docker_password=${DOCKER_PASSWORD?Variable DOCKER_PASSWORD is required}
onnx_runtime=${ONNX_RUNTIME?Variable ONNX_RUNTIME is required}
trust_remote_code=${TRUST_REMOTE_CODE:-false}
use_sentence_transformers_vectorizer=${USE_SENTENCE_TRANSFORMERS_VECTORIZER:-false}
original_model_name=$model_name
git_tag=$GITHUB_REF_NAME

function main() {
  init
  echo "git ref type is $GITHUB_REF_TYPE"
  echo "git ref name is $GITHUB_REF_NAME"
  echo "git tag is $git_tag"
  echo "onnx_runtime is $onnx_runtime"
  echo "trust_remote_code is $trust_remote_code"
  push_tag
}

function init() {
  if [ ! -z "$MODEL_TAG_NAME" ]; then
    # a model tag name was specified to overwrite the model name. This is the
    # case, for example, when the original model name contains characters we
    # can't use in the docker tag
    model_name="$MODEL_TAG_NAME"
  fi

  docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
  docker buildx create --use
  echo "$docker_password" | docker login -u "$docker_username" --password-stdin
}

function push_tag() {
  if [ ! -z "$git_tag" ] && [ "$GITHUB_REF_TYPE" == "tag" ]; then
    model_name_part=$model_name
    if [ "$onnx_runtime" == "true" ]; then
      model_name_part="$model_name-onnx"
    fi
    tag_git="$remote_repo:$model_name_part-$git_tag"
    tag_latest="$remote_repo:$model_name_part-latest"
    tag="$remote_repo:$model_name_part"

    echo "Tag & Push $tag, $tag_latest, $tag_git"
    docker buildx build --platform=linux/arm64,linux/amd64 \
      --build-arg "MODEL_NAME=$original_model_name" \
      --build-arg "ONNX_RUNTIME=$onnx_runtime" \
      --build-arg "TRUST_REMOTE_CODE=$trust_remote_code" \
      --build-arg "USE_SENTENCE_TRANSFORMERS_VECTORIZER=$use_sentence_transformers_vectorizer" \
      --push \
      --tag "$tag_git" \
      --tag "$tag_latest" \
      --tag "$tag" \
      .
  fi
}

main "${@}"
