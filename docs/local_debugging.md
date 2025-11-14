Conda Backend (Recommended for Dependency Isolation)
====================================================
The `CondaBackend` runs models in separate conda environments via HTTP microservice, ensuring complete dependency independence between Boiler Room and model-specific environments. This is the recommended approach for local debugging as it avoids dependency conflicts.

**Supported Tools:**
- **micromamba** (recommended): Fastest and most lightweight
- **mamba**: Fast, drop-in replacement for conda
- **conda**: Standard conda package manager

**Auto-Detection:**
By default, the conda backend automatically detects available tools in priority order: micromamba > mamba > conda. You can also explicitly specify a tool.

**Installation:**
Install one of the supported tools. We recommend micromamba for lean and fast performance:
- Micromamba: https://mamba.readthedocs.io/en/latest/installation.html
- Mamba: https://mamba.readthedocs.io/en/latest/installation.html
- Conda: https://docs.conda.io/en/latest/miniconda.html

**Example Usage:**
```python
from boileroom import ESM2

# Auto-detect available tool (micromamba > mamba > conda)
# Will suggest faster alternatives if available
model = ESM2(backend="conda", device="cuda:0")

# Explicitly use mamba
model = ESM2(backend="mamba", device="cuda:0")

# Explicitly use micromamba (fastest)
model = ESM2(backend="micromamba", device="cuda:0")

# The backend will automatically:
# 1. Create conda environment from environment.yml if missing
# 2. Start HTTP server in the conda environment
# 3. Handle all model operations via HTTP
result = model.embed(sequences)
```

Sometimes, the local (conda/apptainer) environments might require some specific commands that need to be run. Either to install, or to run before setting up the server. For that, we will create a `.install` and `.run` files in each model's folder, that will be executed as effective bash scripts through the backend.


Backend Requirements
--------------------

| Backend | Extra | Additional Step |
|---------|-------|-----------------|
| Boltz-2 | boltz | Install `numpy<2.0` |
| Chai-1  | chai  | Install `numpy<2.0` |

Example Usage
-------------

```python
from boileroom import Chai1
model = Chai1(backend="local")
result = model.fold([sequence])
```

```
boltz = [
    "boltz==2.1.1",
]
chai = [
    "chai_lab==0.6.1",
    "hf_transfer==0.1.8",
]
```

Chai requires some rdkit, that is not supported on some older CentOS, hence need to do:
For this, we should make Docker/Apptainer images -- then one can debug in the Local environment.
```
micromamba install python=3.12 rdkit
pip install chai_lab==0.6.1 hf_transfer==0.1.8
```

This is a bit tricky, and I would have to implement Apptainer images to some extent - that is possible but not a priority.
