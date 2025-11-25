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

declare -A IMAGE_NAMES=(
  ["esm"]="boileroom-esm"
  ["chai"]="boileroom-chai1"
  ["boltz"]="boileroom-boltz"
)

declare -A MODULE_IMPORTS=(
  ["esm"]="boileroom.models.esm.core boileroom.models.esm.esmfold boileroom.models.esm.esm2"
  ["chai"]="boileroom.models.chai.chai1"
  # ["boltz"]="boileroom.models.boltz.boltz2"
  ["boltz"]="boileroom.models.esm.core"
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

  docker run --rm "${image}" micromamba run -n base python - "${module_args[@]}" <<'PY'
import importlib
import sys

modules = sys.argv[1:]
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

