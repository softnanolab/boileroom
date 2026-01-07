#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Run import smoke-tests inside each published model image.

Options:
  --tag=<value>        Tag suffix to verify (default: dev)
  --registry=<value>   Docker registry namespace (default: docker.io/jakublala)
  --pull               docker pull images before running checks
  -h, --help           Show this help and exit
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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BOILEROOM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

if [ ! -d "${BOILEROOM_ROOT}/boileroom" ]; then
  echo "Error: Could not find boileroom directory at ${BOILEROOM_ROOT}/boileroom" >&2
  exit 1
fi

declare -A IMAGE_NAMES=(
  ["esm"]="boileroom-esm"
  ["chai"]="boileroom-chai1"
  ["boltz"]="boileroom-boltz"
)

declare -A MODEL_DIRS=(
  ["esm"]="esm"
  ["chai"]="chai"
  ["boltz"]="boltz"
)

MODEL_ORDER=("esm" "chai" "boltz")

for model in "${MODEL_ORDER[@]}"; do
  image="${DOCKER_REGISTRY}/${IMAGE_NAMES[${model}]}:${REQUESTED_TAG}"
  echo "Checking imports for ${model} (${image})"

  [ "${DO_PULL}" = true ] && docker pull "${image}"

  model_dir="${MODEL_DIRS[${model}]}"
  env_yml="${BOILEROOM_ROOT}/boileroom/models/${model_dir}/environment.yml"
  core_file="${BOILEROOM_ROOT}/boileroom/models/${model_dir}/core.py"

  if [ ! -f "${env_yml}" ]; then
    echo "Error: environment.yml not found: ${env_yml}" >&2
    exit 1
  fi

  docker run --rm \
    -v "${BOILEROOM_ROOT}:${BOILEROOM_ROOT}:ro" \
    -e PYTHONPATH="${BOILEROOM_ROOT}" \
    "${image}" micromamba run -n base python -c "
import ast
import importlib
import re
import sys
from pathlib import Path

env_yml = Path('${env_yml}')
core_file = Path('${core_file}')

# Parse environment.yml to extract pip dependencies (simple regex-based parser)
deps = []
in_pip_section = False
with open(env_yml) as f:
    for line in f:
        stripped = line.strip()
        if stripped == '- pip:' or stripped == 'pip:':
            in_pip_section = True
            continue
        if in_pip_section:
            # Stop if we hit an empty line or a non-indented line (new top-level section)
            if not stripped or (not line.startswith(' ') and not line.startswith('\t')):
                in_pip_section = False
                if not stripped:
                    continue
                else:
                    break
            if stripped.startswith('#'):
                continue
            # Extract package name (remove version specifiers and leading dash)
            pkg_name = re.split(r'[>=<!=]', stripped.lstrip('- '))[0].strip()
            # Map package names to import names
            import_map = {
                'pytorch-lightning': 'pytorch_lightning',
                'torch-tensorrt': None,
                'hf-transfer': None,
                'hf_transfer': None,
            }
            import_name = import_map.get(pkg_name, pkg_name.replace('-', '_'))
            if import_name:
                deps.append(import_name)

# Also add numpy (always present in conda deps)
deps.append('numpy')

# Check core file syntax
if not core_file.exists():
    print(f'FAILED: Core file not found: {core_file}', file=sys.stderr)
    sys.exit(1)

try:
    ast.parse(core_file.read_text(), filename=str(core_file))
    print(f'OK: {core_file.name} (syntax valid)')
except SyntaxError as e:
    print(f'FAILED: {core_file.name} has syntax errors: {e}', file=sys.stderr)
    sys.exit(1)

# Check dependencies
for dep in deps:
    try:
        importlib.import_module(dep)
        print(f'OK: {dep}')
    except Exception as exc:
        print(f'FAILED: {dep} - {exc}', file=sys.stderr)
        sys.exit(1)
" || {
      echo "FAILED: Import check for ${model}" >&2
      exit 1
    }
done

echo "All module imports succeeded for tag: ${REQUESTED_TAG}"
