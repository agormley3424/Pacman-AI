#!/bin/bash

readonly THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

trap exit SIGINT

python -m flake8 "${THIS_DIR}/pacai" --config="${THIS_DIR}/.ci/flake8.cfg"
