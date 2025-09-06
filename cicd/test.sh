#!/usr/bin/env bash

set -eou pipefail

local_repo=${LOCAL_REPO?Variable LOCAL_REPO is required}

pip3 install -r requirements-test.txt

echo "Running tests with authorization on"

container_id=$(docker run -d -it -e AUTHENTICATION_ALLOWED_TOKENS='token1,token2' -p "8000:8080" "$local_repo")

python3 smoke_auth_test.py

docker stop $container_id

echo "Running tests with enabled cache"

container_id=$(docker run -d -it -e ENABLE_CACHE='1' -p "8000:8080" "$local_repo")

python3 smoke_validate_cache_test.py

docker stop $container_id

echo "Running tests without authorization"

container_id=$(docker run -d -it -p "8000:8080" "$local_repo")

python3 smoke_test.py

echo "All tests passed"
