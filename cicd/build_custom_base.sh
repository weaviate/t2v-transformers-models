#!/usr/bin/env bash

set -eou pipefail

docker build -f custom.Dockerfile -t "custom-base" .
