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
| AlphaFold2-Multimer | 🍊 | 🍊 |
| Boltz-2    | 🍊       | 🍊           |
| Chai-1     | 🍊       | 🍊           |
| ESMFold2   | 🍊       | 🍊           |
| ESMFold    | ✅       | 🍊           |
| OpenDDE    | 🍊       | 🍊           |
| Protenix   | 🍊       | 🍊           |
| RF3        | 🍊       | 🍊           |

### Embedding Algorithms

| Model      | 🟢 Modal | 🐧 Apptainer |
|------------|:--------:|:------------:|
| ESM-2      | ✅       | 🍊           |
| ESM-C      | ✅       | 🍊           |
| ESM3       | ✅       | 🍊           |

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

# Or pin an explicit published image tag
model = Boltz2(backend="apptainer:cuda11.8-0.3.0")
```

`backend="apptainer"` resolves through `BOILEROOM_IMAGE_TAG` when set, otherwise through the installed boileroom package version on the default `12.6` image line. An explicit suffix such as `backend="apptainer:cuda11.8-0.3.0"` wins over the shared env override. The kit images behind `optimization="exact"` / `"fast"` are the exception: they are pulled by the digest pinned in `KIT_IMAGE_DIGESTS`, a suffix or `BOILEROOM_KIT_IMAGE_TAG` names a kit tag instead, and `BOILEROOM_IMAGE_TAG` does not apply (see [docker_images.md](docker_images.md#kit-images-optimizationexact-and-fast)).

### Generation Algorithms

| Model | 🟢 Modal | 🐧 Apptainer |
|-------|:--------:|:------------:|
| ProFam | 🍊 | 🍊 |

ProFam generates amino-acid sequences from one family prompt; it does not return
structures or embeddings. Both backend implementations share the same core and
signed output contract, but container/GPU validation must be reported separately
from offline transport tests. See [ProFam usage and limitations](models.md#profam).
