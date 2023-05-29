#!/usr/bin/env bash

set -eou pipefail

git_hash=
remote_repo=${REMOTE_REPO?Variable REMOTE_REPO is required}
docker_username=${DOCKER_USERNAME?Variable DOCKER_USERNAME is required}
docker_password=${DOCKER_PASSWORD?Variable DOCKER_PASSWORD is required}
git_tag=${GITHUB_REF##*/}

function main() {
  init
  echo "git ref type is $GITHUB_REF_TYPE"
  echo "git ref name is $GITHUB_REF_NAME"
  echo "git tag is $git_tag"
  push_tag
}

function init() {
  git_hash="$(git rev-parse HEAD | head -c 7)"

  docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
  docker buildx create --use

  echo "$docker_password" | docker login -u "$docker_username" --password-stdin
}

function push_tag() {
  if [ ! -z "$git_tag" ] && [ "$GITHUB_REF_TYPE" == "tag" ]; then
    tag_git="$remote_repo:custom-$git_tag"
    tag_latest="$remote_repo:custom-latest"
    tag="$remote_repo:custom"

    echo "Tag & Push $tag, $tag_latest, $tag_git"
    docker buildx build --platform=linux/arm64,linux/amd64 -f custom.Dockerfile \
      --tag "$tag" \
      --tag "$tag_latest" \
      --tag "$tag_git" \
      --push \
      .
  fi
}

main "${@}"
