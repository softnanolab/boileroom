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
from boileroom import ESMC, ESMFold

# Use Modal by default; pass backend="apptainer" for local containerized execution
model = ESMFold()

# Predict structure for a protein sequence
sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"

result = model.fold(sequence, options={"include_fields": ["plddt"]})

# Access prediction results
atom_array = result.atom_array[0]
coordinates = atom_array.coord
confidence = result.plddt[0]  # Requested explicitly via include_fields above

# ESM-C / ESM3 are embedding-only. Colon-separated chains are supported.
embedder = ESMC(config={"model_name": "esmc_300m"})
embedding_result = embedder.embed("ACD:EF")
embedding_result.embeddings.shape  # (1, 5, features), residue-only
embedding_result.chain_index       # [[0, 0, 0, 1, 1]]
```

ESM-C and ESM3 use the MIT-licensed 2026 Chan Zuckerberg Biohub `esm` fork (weights `biohub/esmc-*-2024-12` and `biohub/esm3-sm-open-v1`). They share the Biohub `esmfold2` runtime image — the same `esm` package backs all three. Set `MODEL_DIR` to control the shared model-weight cache.

Confidence metrics returned by structure wrappers use a consistent public shape: `plddt` entries are unit-scale
per-residue arrays on `[0, 1]`, and scalar scores such as `ptm` and `iptm` are returned as shape-`(1,)` arrays.
In `0.3.1`, this replaces ESMFold's old padded pLDDT batch array and moves Boltz `ptm`/`iptm` from nested
`confidence` dictionaries to top-level fields.

Protenix, OpenDDE, RF3 and AlphaFold2-Multimer keep their model runners loaded between `fold()` calls. Create one model instance and reuse it for successive jobs, just like the other wrappers. Protenix, OpenDDE and AlphaFold2-Multimer fetch MSAs from the ColabFold MMseqs2 server by default; none needs local genetic databases. RF3 runs no MSA search at all and folds each chain from its sequence unless you pass alignments. Callers can supply their own alignment with `options={"msa": ...}` (Protenix, OpenDDE, RF3, ESMFold2 and AlphaFold2-Multimer) and mmCIF structure templates with `options={"templates": {...}}` (Protenix and OpenDDE only). See [docs/models.md](docs/models.md) for examples, configuration and Modal GPU defaults.

ESMFold2, Protenix, OpenDDE and RF3 also accept `config={"optimization": "vanilla" | "exact"}` (default `"vanilla"`), which runs the Anthropic kit kernels on A100 or H100/H200 GPUs for a per-fold speedup; RF3 also accepts `"fast"` and `"big"` (a lower-memory mode). See [docs/optimization.md](docs/optimization.md).

## Available Models

| Model      | Status | Description                                    | Reference                                              |
|------------|--------|------------------------------------------------|--------------------------------------------------------|
| ESMFold    | ✅      | Fast protein structure prediction   | [Facebook (now Meta)](https://github.com/facebookresearch/esm)     |
| ESMFold2   | ✅      | MIT-licensed all-atom structure prediction model | [Chan Zuckerberg Biohub](https://github.com/Biohub/esm) |
| ESM-2    | ✅      | MSA-free embedding model   | [Facebook (now Meta)](https://github.com/facebookresearch/esm)     |
| ESM-C    | ✅      | MIT-licensed embedding-only model | [Chan Zuckerberg Biohub](https://github.com/Biohub/esm) |
| ESM3     | ✅      | MIT-licensed multimodal model, embedding-only in Boileroom | [Chan Zuckerberg Biohub](https://github.com/Biohub/esm) |
| Chai-1    | ✅      | Protein design and structure prediction model | [Chai Discovery](https://github.com/chaidiscovery/chai-lab) |
| Boltz-2   | ✅      | Diffusion-based protein structure prediction | [Boltz / MIT](https://github.com/jwohlwend/boltz) |
| Protenix  | 🍊      | AlphaFold3-style biomolecular structure prediction (Protenix v2) | [ByteDance](https://github.com/bytedance/Protenix) |
| OpenDDE   | 🍊      | AF3-style biomolecular structure prediction (OpenDDE v1, Protenix-compatible interface) | [Aureka Research](https://github.com/aurekaresearch/OpenDDE) |
| RF3       | 🍊      | RoseTTAFold 3 protein structure and complex prediction (weights license unconfirmed, see [docs/models.md](docs/models.md#rf3)) | [RosettaCommons](https://github.com/RosettaCommons/foundry) |
| AlphaFold2-Multimer | 🍊 | Protein complex prediction via ColabFold (`alphafold2_multimer_v3`) | [Google DeepMind](https://github.com/google-deepmind/alphafold) / [ColabFold](https://github.com/sokrypton/ColabFold) |

> **Licensing:** all bundled model weights are MIT-licensed except **Chai-1**, whose weights are released under the non-commercial Chai Discovery Community License. Review Chai Discovery's terms before using Chai-1 outside research. The **RF3** code is BSD-3-Clause, but its weights' license has not been confirmed from a primary source, so they are downloaded on first use instead of being baked into the images; check it before redistributing.

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

This starts four pytest workers and keeps each model family on its own worker, so each model family (Boltz, Chai, ESM2, ESMFold, ESMFold2, Protenix, OpenDDE, RF3, AlphaFold2-Multimer) uses its own Modal app without registering unrelated GPU functions in the same app.

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
  --docker-user <your-dockerhub-user> \
  --image-tag cuda12.6-my-test-tag \
  --gpu A10
```

The same `--image-tag` option works for the Apptainer backend. For Apptainer you can also pass the tag inline as
`--backend apptainer:<tag>` (the inline suffix wins over `--image-tag`).

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
- The teams behind ESMFold, AlphaFold, ColabFold, Protenix, and other protein prediction models
