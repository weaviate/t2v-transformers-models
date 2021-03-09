#!/usr/local/env bash

set -e pipefail

# Docker push rules
# If not on master
# - nothing is pushed
# If on master and not PR
# - any commit is pushed as :<model>-<7-digit-hash> 
# If on tag (e.g. 1.0.0)
# - any commit is pushed as :<model>-<semver>
# - any commit is pushed as :<model>-latest
# - any commit is pushed as :<model>
git_hash=
pr=
local_repo=${LOCAL_REPO?Variable LOCAL_REPO is required}
remote_repo=${REMOTE_REPO?Variable REMOTE_REPO is required}
model_name=${MODEL_NAME?Variable MODEL_NAME is required}
docker_username=${DOCKER_USERNAME?Variable DOCKER_USERNAME is required}
docker_password=${DOCKER_PASSWORD?Variable DOCKER_PASSWORD is required}

function main() {
  init
  push_master
  push_tag
}

function init() {
  git_hash="$(git rev-parse HEAD | head -c 7)"
  pr=false
  if [ ! -z "$GIT_PULL_REQUEST" ]; then
    pr="$GIT_PULL_REQUEST"
  fi

  echo "$docker_password" | docker login -u "$docker_username" --password-stdin
}

# Note that some CI systems, such as travis, will not provide the branch, but
# the tag on a tag-push. So this method will not be called on a tag-run.
function push_master() {
  if [ "$GIT_BRANCH" == "master" ] && [ "$pr" == "false" ]; then
    # The ones that are always pushed

    tag="$remote_repo:$model_name-$git_hash"
    docker tag "$local_repo" "$tag" && docker push "$tag"
  fi
}

function push_tag() {
  if [ ! -z "$GIT_TAG" ]; then
    tag="$remote_repo:$model_name-$GIT_TAG"
    echo "Tag & Push $tag"
    docker tag "$LOCAL_TAG" "$tag" && docker push "$tag"

    tag="$remote_repo:$model_name-latest"
    echo "Tag & Push $tag"
    docker tag "$LOCAL_TAG" "$tag" && docker push "$tag"

    tag="$remote_repo:$model_name"
    echo "Tag & Push $tag"
    docker tag "$LOCAL_TAG" "$tag" && docker push "$tag"
  fi
}

main "${@}"
