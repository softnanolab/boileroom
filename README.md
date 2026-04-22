# boileroom: protein prediction models across Modal and Apptainer

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/boileroom.svg)](https://pypi.org/project/boileroom/)
[![GitHub last commit](https://img.shields.io/github/last-commit/jakublala/boileroom.svg)](https://github.com/jakublala/boileroom/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/jakublala/boileroom.svg)](https://github.com/jakublala/boileroom/issues)

`boileroom` is a Python package that provides a unified interface to protein prediction models across Modal's serverless GPUs and Apptainer-based local or HPC execution.

> 🚨🚨🚨 **v0.3.0** introduced major changes, including new models and inference backends. If you're upgrading from v0.2, please see the [Migration Guide](docs/migration_v0.2_to_v0.3.md) for details on breaking changes and how to update your code. 🚨🚨🚨

> ⚠️ **Note:** This package is currently in active development. The API and features may change between versions. We recommend pinning your version in production environments.

## Features

- 🚀 Modal and Apptainer execution backends
- 🔄 Unified API across different models and runtimes
- 🎯 Production-ready with GPU acceleration
- 📦 Easy installation and deployment

## Installation

1. Install the package using pip:

```bash
pip install boileroom
```

2. If you plan to use Modal, set up Modal credentials:

```bash
modal token new
```

For local containerized execution instead, install Apptainer and use `backend="apptainer"`.

## Quick Start

```python
from boileroom import ESMFold

# Use Modal by default; pass backend="apptainer" for local containerized execution
model = ESMFold()

# Predict structure for a protein sequence
sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"

result = model.fold(sequence, options={"include_fields": ["plddt"]})

# Access prediction results
atom_array = result.atom_array[0]
coordinates = atom_array.coord
confidence = result.plddt[0]  # Requested explicitly via include_fields above
```

## Available Models

| Model      | Status | Description                                    | Reference                                              |
|------------|--------|------------------------------------------------|--------------------------------------------------------|
| ESMFold    | ✅      | Fast protein structure prediction   | [Facebook (now Meta)](https://github.com/facebookresearch/esm)     |
| ESM-2    | ✅      | MSA-free embedding model   | [Facebook (now Meta)](https://github.com/facebookresearch/esm)     |
| Chai-1    | ✅      | Protein design and structure prediction model | [Chai Discovery](https://github.com/chaidiscovery/chai-lab) |
| Boltz-2   | ✅      | Diffusion-based protein structure prediction | [Boltz / MIT](https://github.com/jwohlwend/boltz) |

## Development

1. Clone the repository:

```bash
git clone https://github.com/jakublala/boileroom
cd boileroom
```

2. Install development dependencies using `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync
```

3. Run tests:

```bash
uv run pytest
```

For Modal integration tests, run the model families in parallel shards:

```bash
uv run pytest -v -n 4 --dist loadgroup -m integration
```

This starts four pytest workers and keeps each model family on its own worker, so Boltz, Chai, ESM2, and ESMFold use separate Modal apps without registering unrelated GPU functions in the same app.

To run the same integration tests in series, omit xdist:

```bash
uv run pytest -v -m integration
```

or only one test that's more verbose and shows print statements:

```bash
uv run python -m pytest tests/test_basic.py::test_esmfold_batch -v -s
```

To specify a GPU type for Modal backend tests (defaults to T4 if not specified):

```bash
uv run pytest --gpu A100-40GB
```

To run Modal integration tests against a specific Docker Hub namespace, image tag, and GPU type:

```bash
uv run pytest -v -n 4 --dist loadgroup -m integration \
  --docker-user phauglin \
  --image-tag cuda12.6-my-test-tag \
  --gpu A10
```

Available GPU options include `T4`, `A100-40GB`, `A100-80GB`, and other Modal-supported GPU types.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use `boileroom` in your research, please cite:

```bibtex
@software{boileroom2025,
  author = {Lála, Jakub},
  title = {boileroom: serverless protein prediction models},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/softnanolab/boileroom}
}
```

## Acknowledgments

- [Modal Labs](https://modal.com/) for the serverless infrastructure
- The teams behind ESMFold, AlphaFold, and other protein prediction models
