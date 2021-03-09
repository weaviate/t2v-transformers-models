#!/usr/bin/env bash

set -e

local_repo=${LOCAL_REPO?Variable LOCAL_REPO is required}

pip3 install -r requirements-test.txt

docker run -d -it -p "8000:8080" "$local_repo"

python3 smoke_test.py
