#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run import smoke-tests inside each published model image. The script spins up
each container, imports the expected modules, and fails fast if any import
raises an exception.

Options:
  --tag=<value>        Tag suffix to verify (default: dev)
  --registry=<value>   Docker registry namespace (default: docker.io/jakublala)
  --pull               docker pull images before running checks
  -h, --help           Show this help and exit

Examples:
  $(basename "$0") --tag=dev
  $(basename "$0") --tag=latest --pull
EOF
}

REQUESTED_TAG="dev"
DOCKER_REGISTRY="docker.io/jakublala"
DO_PULL=false

for arg in "$@"; do
  case "$arg" in
    --tag=*) REQUESTED_TAG="${arg#*=}" ;;
    --registry=*) DOCKER_REGISTRY="${arg#*=}" ;;
    --pull) DO_PULL=true ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $arg" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required but was not found on PATH." >&2
  exit 1
fi

# Get the boileroom root directory (parent of parent of scripts/images)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BOILEROOM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

if [ ! -d "${BOILEROOM_ROOT}/boileroom" ]; then
  echo "Error: Could not find boileroom directory at ${BOILEROOM_ROOT}/boileroom" >&2
  echo "Please run this script from the boileroom repository." >&2
  exit 1
fi

declare -A IMAGE_NAMES=(
  ["esm"]="boileroom-esm"
  ["chai"]="boileroom-chai1"
  ["boltz"]="boileroom-boltz"
)

# Check core.py files - these contain the actual model logic and dependencies
declare -A MODULE_IMPORTS=(
  ["esm"]="boileroom.models.esm.core"
  ["chai"]="boileroom.models.chai.core"
  ["boltz"]="boileroom.models.boltz.core"
)

MODEL_ORDER=("esm" "chai" "boltz")

for model in "${MODEL_ORDER[@]}"; do
  image="${DOCKER_REGISTRY}/${IMAGE_NAMES[${model}]}:${REQUESTED_TAG}"
  echo "Checking imports for ${model} (${image})"

  if [ "${DO_PULL}" = true ]; then
    docker pull "${image}"
  fi

  read -r -a module_args <<<"${MODULE_IMPORTS[${model}]}"
  if [ "${#module_args[@]}" -eq 0 ]; then
    echo "No modules configured for ${model}; skipping." >&2
    continue
  fi

  # Mount boileroom source and set PYTHONPATH so imports work
  # Pass modules as environment variable to avoid argument parsing issues with heredoc
  MODULES_STR="${module_args[*]}"
  docker run --rm \
    -v "${BOILEROOM_ROOT}:${BOILEROOM_ROOT}:ro" \
    -e PYTHONPATH="${BOILEROOM_ROOT}" \
    -e CHECK_MODULES="${MODULES_STR}" \
    "${image}" micromamba run -n base python <<'PY'
import importlib
import os
import sys

modules_str = os.environ.get("CHECK_MODULES", "")
if not modules_str:
    raise SystemExit("No modules provided to import")

modules = modules_str.split()
if not modules:
    raise SystemExit("No modules provided to import")

for module in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        raise SystemExit(f"FAILED: {module} - {exc}") from exc
    else:
        print(f"OK: {module}")
PY
done

echo "All module imports succeeded for tag: ${REQUESTED_TAG}"

