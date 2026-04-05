## Boileroom Docker images

### What exists today
- **base**: `boileroom/boileroom/images/Dockerfile` → CUDA-enabled micromamba base with Python 3.12. Tag: `docker.io/jakublala/boileroom-base`.
- **boltz**: `boileroom/boileroom/models/boltz/Dockerfile` → copies `environment.yml` and installs the Boltz env. Tag: `docker.io/jakublala/boileroom-boltz`.
- **chai1**: `boileroom/boileroom/models/chai/Dockerfile` → copies `environment.yml`, installs chai-specific deps, sets HF env vars. Tag: `docker.io/jakublala/boileroom-chai1`.
- **esm**: `boileroom/boileroom/models/esm/Dockerfile` → copies `environment.yml` shared by esm2/esmfold. Tag: `docker.io/jakublala/boileroom-esm`.

Dockerfiles are the canonical image definition for all runtimes. Docker/Apptainer images are built from these Dockerfiles, and Modal pulls the corresponding published model image from Docker Hub instead of maintaining a separate handwritten dependency stack.

### Tag scheme
- Canonical published tags are CUDA-qualified, for example `cuda12.6-latest`, `cuda12.6-0.3.0`, and `cuda11.8-latest`.
- The default CUDA line is `12.6`. That line also gets unqualified aliases such as `latest` and `0.3.0`.
- Runtime shorthands such as `backend="apptainer"` or `backend="apptainer:0.3.0"` resolve through those default aliases.

### 🚀 Quick start (dev)
Use the Python helper to build all images (base + models) with a single global worker limit.

```bash
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --tag=latest --platform=linux/amd64 --max-workers=4

# Optional flags
uv run python scripts/images/build_model_images.py --no-cache ...
uv run python scripts/images/build_model_images.py --all-cuda --tag=0.3.0 --push ...
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --tag=pr-39 --push ...
```

Single-platform non-push builds auto-load into the local Docker daemon. Multi-platform builds should generally be paired with `--push`.

You can still use the Bash wrapper if you prefer:
```bash
chmod +x scripts/images/build_model_images.sh
scripts/images/build_model_images.sh --cuda-version=12.6 --tag=latest --platform=linux/amd64
```

### ✅ Import smoke tests
Run the lightweight smoke script to ensure each image can import its expected modules:

```bash
uv run python scripts/images/check_model_imports.py --tag=latest
uv run python scripts/images/check_model_imports.py --all-cuda --tag=latest --pull
uv run python scripts/images/check_model_imports.py --all-cuda --tag=0.3.0 --pull
```

The GitHub Actions workflow (`.github/workflows/build-docker-images.yml`) runs the same checks after building both canonical CUDA-qualified tags and the default `12.6` aliases.

### 🛠️ Manual local builds
- Build base:
```bash
docker build \
  --platform linux/amd64 \
  -t docker.io/jakublala/boileroom-base:dev \
  -f boileroom/boileroom/images/Dockerfile \
  boileroom/boileroom/images
```

- Build boltz (using the dev base tag):
```bash
docker build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=docker.io/jakublala/boileroom-base:dev \
  -t docker.io/jakublala/boileroom-boltz:dev \
  -f boileroom/boileroom/models/boltz/Dockerfile \
  boileroom/boileroom/models/boltz
```

- Build chai1:
```bash
docker build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=docker.io/jakublala/boileroom-base:dev \
  -t docker.io/jakublala/boileroom-chai1:dev \
  -f boileroom/boileroom/models/chai/Dockerfile \
  boileroom/boileroom/models/chai
```

- Build esm:
```bash
docker build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=docker.io/jakublala/boileroom-base:dev \
  -t docker.io/jakublala/boileroom-esm:dev \
  -f boileroom/boileroom/models/esm/Dockerfile \
  boileroom/boileroom/models/esm
```

### ☁️ Push local tags to Docker Hub
Use the helper script with `--push` to push all images after building. Authenticate first:
```bash
docker login
uv run python scripts/images/build_model_images.py --all-cuda --tag=0.3.0 --push
```
This publishes:
- canonical tags such as `cuda11.8-0.3.0` and `cuda12.6-0.3.0`
- default aliases such as `0.3.0` for the `12.6` line

### 📦 CI publishing (production)
GitHub Actions at `.github/workflows/build-docker-images.yml` now drives the release pipeline:
- Triggers automatically on pushes to `main` and can also be run manually via **Run workflow**.
- Runs the same Python helper twice with Docker Buildx: once for `latest`, once for the current `project.version` from `pyproject.toml`.
- Each invocation uses `--all-cuda --push`, so the base and per-model images land on Docker Hub with canonical CUDA-qualified tags and default `12.6` aliases.
- Future merges inherit the cache layers thanks to BuildKit, keeping CI times reasonable.

### 🧱 Convert Docker images to Apptainer (SIF)
If your cluster uses Apptainer/Singularity for job execution, you can convert the Docker images to a `.sif` image in two common ways:

1) Directly from the registry (simplest):
```bash
# Default CUDA aliases
apptainer pull base.sif  docker://docker.io/jakublala/boileroom-base:latest
apptainer pull chai1.sif docker://docker.io/jakublala/boileroom-chai1:latest

# Explicit CUDA-qualified tags
apptainer pull chai1-cu118.sif docker://docker.io/jakublala/boileroom-chai1:cuda11.8-latest
apptainer pull chai1-cu126.sif docker://docker.io/jakublala/boileroom-chai1:cuda12.6-latest

# If the repository is private, authenticate first to Docker Hub:
# This will prompt for your Docker Hub credentials if needed.
apptainer remote login docker://docker.io
apptainer pull base.sif  docker://docker.io/jakublala/boileroom-base:latest
apptainer pull chai1.sif docker://docker.io/jakublala/boileroom-chai1:latest
```

2) From a local Docker image (no registry pull on the cluster) (🚨 **THIS HAS NOT BEEN TESTED, AND IS NOT RECOMMENDED** 🚨):
```bash
# On a build machine (e.g., your workstation):
docker pull docker.io/jakublala/boileroom-chai1:cuda12.6-latest
docker save --format oci-archive -o chai1-oci.tar docker.io/jakublala/boileroom-chai1:cuda12.6-latest

# Transfer chai1-oci.tar to the cluster, then:
apptainer build chai1.sif oci-archive://chai1-oci.tar
```

Either approach yields an Apptainer image `chai1.sif` that you can run with:
```bash
apptainer exec chai1.sif python -c "import torch; print(torch.cuda.is_available())"
```

### 📂 Configure model storage location (MODEL_DIR)
Set `MODEL_DIR` at runtime to the host-mounted path that should store model weights. Model-specific directories are automatically derived under `MODEL_DIR` (e.g., `MODEL_DIR/chai` for Chai, `MODEL_DIR/boltz` for Boltz). The runtime automatically sets `CHAI_DOWNLOADS_DIR=$MODEL_DIR/chai` when `MODEL_DIR` is defined.

Docker example:
```bash
# Store models on host at /data/models and expose to the container
docker run --rm \
  -e MODEL_DIR=/data/models \
  -v /data/models:/data/models \
  docker.io/jakublala/boileroom-chai1:latest python -c "import os; print(os.getenv('MODEL_DIR'))"
```

Apptainer examples:
```bash
# Option 1: keep default /mnt/models by binding a host dir there
apptainer exec -B /scratch/weights:/mnt/models chai1.sif python -c "import os; print(os.getenv('MODEL_DIR'))"

# Option 2: redirect MODEL_DIR anywhere and bind the same path
apptainer exec --env MODEL_DIR=/scratch/weights -B /scratch/weights:/scratch/weights \
  chai1.sif python -c "import os; print(os.getenv('MODEL_DIR'))"
```

### 🧩 Add your own image
1) Create a `Dockerfile` under `boileroom/boileroom/models/<your_image>/Dockerfile` that starts FROM the dev base locally:
```Dockerfile
ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:latest
FROM ${BASE_IMAGE}
```
2) Build it locally (adjust path and tag):
```bash
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --tag=latest --platform=linux/amd64
```
3) Optionally wire it into the helper scripts after `base` in the correct order.
4) For CI, extend `.github/workflows/build-docker-images.yml` or the helper script to include your image so it is built and tagged alongside the others.

### 💡 Tips
- Keep network-heavy `pip install` steps in as few layers as possible to improve caching.
- Use `--platform linux/amd64` locally to match CI.
- Prefer canonical CUDA-qualified tags when you need exact reproducibility.
