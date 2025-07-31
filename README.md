# boileroom: serverless protein prediction models

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/boileroom.svg)](https://pypi.org/project/boileroom/)
[![GitHub last commit](https://img.shields.io/github/last-commit/jakublala/boileroom.svg)](https://github.com/jakublala/boileroom/commits/main)
[![GitHub issues](https://img.shields.io/github/issues/jakublala/boileroom.svg)](https://github.com/jakublala/boileroom/issues)

`boileroom` is a Python package that provides a unified interface to various protein prediction models, running them efficiently on Modal's serverless infrastructure.

## Features

- 🚀 Serverless execution of protein models
- 🔄 Unified API across different models
- 🎯 Production-ready with GPU acceleration
- 📦 Easy installation and deployment

## Installation

1. Install the package using pip:

```bash
pip install boileroom
```

2. Set up Modal credentials (if you haven't already):

```bash
modal token new
```

## Quick Start

```python
from boileroom import app
from boileroom.esmfold import ESMFold

# Initialize the model
model = ESMFold()

# Predict structure for a protein sequence
sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"
with app.run():
    result = model.fold.remote([sequence])

# Access prediction results
coordinates = result.positions
confidence = result.plddt
```

## Available Models

| Model      | Status | Description                                    | Reference                                              |
|------------|--------|------------------------------------------------|--------------------------------------------------------|
| ESMFold    | ✅      | Fast protein structure prediction   | [Facebook (now Meta)](https://github.com/facebookresearch/esm)     |
| ESM-2    | ✅      | MSA-free embedding model   | [Facebook (now Meta)](https://github.com/facebookresearch/esm)     |

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

or only one test that's more verbose and shows print statements:

```bash
uv run python -m pytest tests/test_basic.py::test_esmfold_batch -v -s
```

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
  url = {https://github.com/jakublala/boileroom}
}
```

## Acknowledgments

- [Modal Labs](https://modal.com/) for the serverless infrastructure
- The teams behind ESMFold, AlphaFold, and other protein prediction models
