#!/usr/bin/env bash

set -e

function build() {
  docker build -f custom.Dockerfile -t "custom-base" .
}

git_hash=
pr=
remote_repo=${REMOTE_REPO?Variable REMOTE_REPO is required}
docker_username=${DOCKER_USERNAME?Variable DOCKER_USERNAME is required}
docker_password=${DOCKER_PASSWORD?Variable DOCKER_PASSWORD is required}

function main() {
  init
  echo "git branch is $GIT_BRANCH"
  echo "git tag is $GIT_TAG"
  echo "pr is $pr"
  build
  push_main
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
function push_main() {
  if [ "$GIT_BRANCH" == "main" ] && [ "$pr" == "false" ]; then
    # The ones that are always pushed

    tag="$remote_repo:custom-$git_hash"
    docker tag "custom-base" "$tag" && docker push "$tag"
  fi
}

function push_tag() {
  if [ ! -z "$GIT_TAG" ]; then
    tag="$remote_repo:custom-$GIT_TAG"
    echo "Tag & Push $tag"
    docker tag "custom-base" "$tag" && docker push "$tag"

    tag="$remote_repo:custom-latest"
    echo "Tag & Push $tag"
    docker tag "custom-base" "$tag" && docker push "$tag"

    tag="$remote_repo:custom"
    echo "Tag & Push $tag"
    docker tag "custom-base" "$tag" && docker push "$tag"
  fi
}

main "${@}"
