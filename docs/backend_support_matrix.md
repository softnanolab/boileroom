Backend Support Matrix
======================

The table below tracks which execution backends are currently available for each model.

Legend:
- ✅ Full support
- 🍊 Experimental/partial support
- ❌ Not implemented

### Structure Algorithms

| Model      | 🟢 Modal | 🐧 Apptainer |
|------------|:--------:|:------------:|
| Boltz-2    | 🍊       | 🍊           |
| Chai-1     | 🍊       | 🍊           |
| ESMFold    | ✅       | 🍊           |

### Embedding Algorithms

| Model      | 🟢 Modal | 🐧 Apptainer |
|------------|:--------:|:------------:|
| ESM-2      | ✅       | 🍊           |

## Backend Descriptions

### Modal (Default)
Modal is the production default backend. Models run on Modal's serverless GPU infrastructure with automatic scaling and caching.

### Apptainer
Apptainer backend runs models in containerized environments locally. Requires Apptainer installed on the system and pre-built Docker images from DockerHub.

Usage:
```python
from boileroom import Boltz2

# Use Modal backend (default)
model = Boltz2(backend="modal")

# Use Apptainer backend
model = Boltz2(backend="apptainer")
```
