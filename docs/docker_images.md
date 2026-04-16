## Boileroom Docker images

### What exists today
- **base**: `boileroom/images/Dockerfile` → CUDA-enabled micromamba base with Python 3.12. Tag: `docker.io/jakublala/boileroom-base`.
- **boltz**: `boileroom/models/boltz/Dockerfile` → copies `environment.yml` and installs the Boltz env. Tag: `docker.io/jakublala/boileroom-boltz`.
- **chai1**: `boileroom/models/chai/Dockerfile` → copies `environment.yml`, installs chai-specific deps, sets HF env vars. Tag: `docker.io/jakublala/boileroom-chai1`.
- **esm**: `boileroom/models/esm/Dockerfile` → copies `environment.yml` shared by esm2/esmfold. Tag: `docker.io/jakublala/boileroom-esm`.

Dockerfiles are the canonical image definition for all runtimes. Docker/Apptainer images are built from these Dockerfiles, and Modal pulls the corresponding published model image from Docker Hub instead of maintaining a separate handwritten dependency stack.

### Tag scheme
- Canonical published tags are CUDA-qualified, for example `cuda12.6-0.3.0`, `cuda11.8-0.3.0`, or `cuda12.6-sha-abc1234`.
- The default CUDA line is `12.6`. That line also gets an unqualified alias for the exact package version or temporary validation tag, for example `0.3.0` or `sha-abc1234`.
- `latest` is not published.
- Runtime shorthands such as `backend="apptainer"` resolve to the installed boileroom package version on the default `12.6` CUDA line.

### 🚀 Quick start
Use the Python helper to build all images (base + models) with a single global worker limit.

```bash
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --platform=linux/amd64 --max-workers=4

# Optional flags
uv run python scripts/images/build_model_images.py --no-cache ...
uv run python scripts/images/build_model_images.py --all-cuda --tag=0.3.0 --push ...
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --tag=sha-$(git rev-parse --short HEAD) --push ...
```

Single-platform non-push builds auto-load into the local Docker daemon. Multi-platform builds should generally be paired with `--push`.

### 🔖 Tag policy
- Docker Hub is kept clean for users. The long-lived public tags are version tags such as `0.3.0` and the corresponding CUDA-qualified tags such as `cuda12.6-0.3.0` and `cuda11.8-0.3.0`.
- Short-lived validation tags such as `sha-<shortsha>` are fine when you need to test a branch through Docker Hub or Modal before promoting a version tag.
- Validation tags should be deleted once the validation pass is complete.

For example, a temporary validation push on the default CUDA line:
```bash
TAG=sha-$(git rev-parse --short HEAD)
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --tag="$TAG" --platform=linux/amd64 --push
```

Modal can then be pointed at that published validation tag with:
```bash
export BOILEROOM_MODAL_IMAGE_TAG="$TAG"
```

Because `12.6` is the default CUDA line, publishing `--tag="$TAG"` also creates the explicit `cuda12.6-$TAG` tag alongside the unqualified alias.

You can still use the Bash wrapper if you prefer:
```bash
chmod +x scripts/images/build_model_images.sh
scripts/images/build_model_images.sh --cuda-version=12.6 --platform=linux/amd64
```

### ✅ Import smoke tests
Run the lightweight smoke script to ensure each image can import its expected modules:

```bash
uv run python scripts/images/check_model_imports.py
uv run python scripts/images/check_model_imports.py --all-cuda --tag=0.3.0 --pull
```

The GitHub Actions workflow (`.github/workflows/build-docker-images.yml`) runs the same checks after building canonical CUDA-qualified validation tags and the matching unqualified validation alias on the default `12.6` line.

### 🛠️ Manual local builds
- Build base:
```bash
docker build \
  --platform linux/amd64 \
  -t docker.io/jakublala/boileroom-base:local \
  -f boileroom/images/Dockerfile \
  boileroom/images
```

- Build boltz (using the local base tag):
```bash
docker build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=docker.io/jakublala/boileroom-base:local \
  -t docker.io/jakublala/boileroom-boltz:local \
  -f boileroom/models/boltz/Dockerfile \
  boileroom/models/boltz
```

- Build chai1:
```bash
docker build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=docker.io/jakublala/boileroom-base:local \
  -t docker.io/jakublala/boileroom-chai1:local \
  -f boileroom/models/chai/Dockerfile \
  boileroom/models/chai
```

- Build esm:
```bash
docker build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=docker.io/jakublala/boileroom-base:local \
  -t docker.io/jakublala/boileroom-esm:local \
  -f boileroom/models/esm/Dockerfile \
  boileroom/models/esm
```

### ☁️ Push local tags to Docker Hub
Use the helper script with `--push` to push all images after building. Authenticate first:
```bash
docker login
uv run python scripts/images/build_model_images.py --all-cuda --tag=0.3.0 --push
```
This publishes:
- canonical tags such as `cuda11.8-0.3.0` and `cuda12.6-0.3.0`
- the unqualified version alias such as `0.3.0` for the `12.6` line

### 📦 CI publishing (production)
GitHub Actions at `.github/workflows/build-docker-images.yml` now drives the release pipeline:
- Triggers automatically on pushes to `main` and can also be run manually via **Run workflow** from `main`.
- Builds a temporary `sha-<commit>` validation tag, verifies that exact pushed artifact, and only then promotes it to an automatically derived `0.3.x` version from `scripts/ci/derive_version.py`.
- The `0.3.x` patch component is the number of commits after the configured main-line baseline, so it increases with every new commit on `main`.
- Each successful run publishes canonical CUDA-qualified tags and the unqualified version alias for the default `12.6` line.
- The official release path currently publishes `linux/amd64` only. If you want to experiment with additional architectures, pass an explicit multi-platform `--platform` value and validate it separately before treating it as supported.
- Future merges inherit dependency cache layers through BuildKit registry caches, keeping CI times reasonable even on fresh GitHub-hosted runners.
- PyPI is not published by this workflow. Python package publication happens later from the GitHub release workflow, which injects the `0.3.x` release tag into `pyproject.toml` before building.

### 🧱 Convert Docker images to Apptainer (SIF)
If your cluster uses Apptainer/Singularity for job execution, you can convert the Docker images to a `.sif` image in two common ways:

1) Directly from the registry (simplest):
```bash
# Version-matched aliases on the default CUDA line
apptainer pull base.sif  docker://docker.io/jakublala/boileroom-base:0.3.0
apptainer pull chai1.sif docker://docker.io/jakublala/boileroom-chai1:0.3.0

# Explicit CUDA-qualified tags
apptainer pull chai1-cu118.sif docker://docker.io/jakublala/boileroom-chai1:cuda11.8-0.3.0
apptainer pull chai1-cu126.sif docker://docker.io/jakublala/boileroom-chai1:cuda12.6-0.3.0

# If the repository is private, authenticate first to Docker Hub:
# This will prompt for your Docker Hub credentials if needed.
apptainer remote login docker://docker.io
apptainer pull base.sif  docker://docker.io/jakublala/boileroom-base:0.3.0
apptainer pull chai1.sif docker://docker.io/jakublala/boileroom-chai1:0.3.0
```

2) From a local Docker image (no registry pull on the cluster) (🚨 **THIS HAS NOT BEEN TESTED, AND IS NOT RECOMMENDED** 🚨):
```bash
# On a build machine (e.g., your workstation):
docker pull docker.io/jakublala/boileroom-chai1:cuda12.6-0.3.0
docker save --format oci-archive -o chai1-oci.tar docker.io/jakublala/boileroom-chai1:cuda12.6-0.3.0

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
  docker.io/jakublala/boileroom-chai1:0.3.0 python -c "import os; print(os.getenv('MODEL_DIR'))"
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
1) Create a `Dockerfile` under `boileroom/models/<your_image>/Dockerfile` that starts FROM the local base image:
```Dockerfile
ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:local
FROM ${BASE_IMAGE}
```
2) Build it locally (adjust path and tag):
```bash
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --platform=linux/amd64
```
3) Optionally wire it into the helper scripts after `base` in the correct order.
4) For CI, extend `.github/workflows/build-docker-images.yml` or the helper script to include your image so it is built and tagged alongside the others.

### 💡 Tips
- Keep network-heavy `pip install` steps in as few layers as possible to improve caching.
- Use `--platform linux/amd64` locally to match the official release workflow.
- Prefer canonical CUDA-qualified tags when you need exact reproducibility.
