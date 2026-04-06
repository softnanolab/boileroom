#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
uv run python "${SCRIPT_DIR}/check_model_server_health.py" "$@"
