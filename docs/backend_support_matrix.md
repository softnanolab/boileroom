Backend Support Matrix
======================

The table below tracks which execution backends are currently available for each model. A green mark denotes
available support, while a red cross indicates that the combination is not yet implemented.

### Structure Algorithms

| Model      | 🟢 Modal | 🐧 Apptainer (FastAPI) | 🐳 Docker (FastAPI) | 💻 Local (uv) | 💻 Local (Docker/Apptainer) | 🐍 Conda (micromamba/mamba/conda) |
|------------|:--------:|:---------------------:|:------------------:|:-------------:|:--------------------------:|:--------------------------------:|
| Boltz-2    | 🍊       | ❌                    | ❌                 | ❌            | ❌                         | 🍊                                |
| Chai-1     | 🍊       | ❌                    | ❌                 | ❌            | ❌                         | 🍊                                |
| ESMFold    | ✅       | ❌                    | ❌                 | ✅            | ❌                         | ✅                                |

### Embedding Algorithms

| Model      | 🟢 Modal | 🐧 Apptainer (FastAPI) | 🐳 Docker (FastAPI) | 💻 Local (uv) | 💻 Local (Docker/Apptainer) | 🐍 Conda (micromamba/mamba/conda) |
|------------|:--------:|:---------------------:|:------------------:|:-------------:|:--------------------------:|:--------------------------------:|
| ESM-2      | ✅       | ❌                    | ❌                 | ✅            | ❌                         | ✅                                |
