#!/usr/bin/env bash

set -eou pipefail

echo "Stop all docker containers"
docker ps -q | xargs --no-run-if-empty docker stop
echo "Delete all docker containers"
docker ps -a -q | xargs --no-run-if-empty docker rm
echo "Delete all docker images"
docker images -q | xargs --no-run-if-empty docker rmi
echo "Success"
