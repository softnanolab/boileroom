## Boileroom Docker images

### What exists today
- **base**: `boileroom/boileroom/images/Dockerfile` → CUDA-enabled micromamba, system deps, Python libs (e.g., `biotite`). Tag: `docker.io/jakublala/boileroom-base`.
- **chai1**: `boileroom/boileroom/models/chai/Dockerfile` → builds on `base`, installs `torch==2.5.1+cu118`, `hf_transfer`, `chai_lab`. Tag: `docker.io/jakublala/boileroom-chai1`.

> 🚨 Warning
> Modal and Docker/Apptainer currently use different images. This will be unified in a future pass. Until then, avoid mixing tags across runners and verify which image your runtime expects.

### 🚀 Quick start (dev)
Use the dev helper script to build all images for development in order (`base` → `chai1`).

```bash
scripts/images/build_all_docker.sh --platform=linux/amd64

# Optional: disable cache
scripts/images/build_all_docker.sh --no-cache
```

If needed, make it executable once:
```bash
chmod +x scripts/images/build_all_docker.sh
```

### 🐧 Podman (HPC/cluster-friendly alternative)
On clusters where Docker isn't available (common on HPC systems), you can use Podman. A mirrored helper script exists and behaves the same as the Docker one.

```bash
chmod +x scripts/images/build_all_podman.sh
scripts/images/build_all_podman.sh --platform=linux/amd64

# Optional: disable cache
scripts/images/build_all_podman.sh --no-cache
```

Manual Podman builds:
```bash
# base
podman build \
  --platform linux/amd64 \
  -t docker.io/jakublala/boileroom-base:dev \
  -f boileroom/boileroom/images/Dockerfile \
  boileroom/boileroom/images

# chai1 (using the dev base tag)
podman build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=docker.io/jakublala/boileroom-base:dev \
  -t docker.io/jakublala/boileroom-chai1:dev \
  -f boileroom/boileroom/models/chai/Dockerfile \
  boileroom/boileroom/models/chai
```

> Note
> Podman runs rootless by default, which is often required on multi-user clusters. It produces the same images compatible with GHCR and downstream tools.

### 🛠️ Manual local builds
- Build base:
```bash
docker build \
  --platform linux/amd64 \
  -t docker.io/jakublala/boileroom-base:dev \
  -f boileroom/boileroom/images/Dockerfile \
  boileroom/boileroom/images
```

- Build chai1 (using the dev base tag):
```bash
docker build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=docker.io/jakublala/boileroom-base:dev \
  -t docker.io/jakublala/boileroom-chai1:dev \
  -f boileroom/boileroom/models/chai/Dockerfile \
  boileroom/boileroom/models/chai
```

### ☁️ Push local :dev tags to GHCR
Use the helper script to push locally built `:dev` images to GHCR:
```bash
# Authenticate first
docker login
# or: podman login docker.io

# Push all available :dev images
boileroom/scripts/images/push_all_dev_to_dockerhub.sh

# Use Podman instead of Docker
boileroom/scripts/images/push_all_dev_to_dockerhub.sh --tool=podman
```

### 📦 CI publishing (production)
GitHub Actions at `.github/workflows/build-docker-images.yml`:
- Builds `base` first and uses its digest output.
- Builds `chai1` pinned to `base@sha256:...` for reproducibility.
- Uses BuildKit cache for faster incremental builds.

### 🧱 Convert Podman images to Apptainer (SIF)
If your cluster uses Apptainer/Singularity for job execution, you can convert the Podman/Docker images to a `.sif` image in two common ways:

1) Directly from the registry (simplest):
```bash
# Public images (CI publishes :latest)
apptainer pull base.sif  docker://docker.io/jakublala/boileroom-base:latest
apptainer pull chai1.sif docker://docker.io/jakublala/boileroom-chai1:latest

# If the repository is private, authenticate first to Docker Hub:
# This will prompt for your Docker Hub credentials if needed.
apptainer remote login docker://docker.io
apptainer pull base.sif  docker://docker.io/jakublala/boileroom-base:latest
apptainer pull chai1.sif docker://docker.io/jakublala/boileroom-chai1:latest
```

2) From a local Podman image (no registry pull on the cluster) (🚨 **THIS HAS NOT BEEN TESTED, AND IS NOT RECOMMENDED** 🚨):
```bash
# On a build machine (e.g., your workstation):
podman pull docker.io/jakublala/boileroom-chai1:dev
podman save --format oci-archive -o chai1-oci.tar docker.io/jakublala/boileroom-chai1:dev

# Transfer chai1-oci.tar to the cluster, then:
apptainer build chai1.sif oci-archive://chai1-oci.tar
```

Either approach yields an Apptainer image `chai1.sif` that you can run with:
```bash
apptainer exec chai1.sif python -c "import torch; print(torch.cuda.is_available())"
```

### 📂 Configure model storage location (MODEL_DIR)
By default models are stored under `/mnt/models` inside the image. You can redirect this via the `MODEL_DIR` environment variable at runtime. The code will also set `CHAI_DOWNLOADS_DIR=$MODEL_DIR/chai1` automatically.

Docker/Podman examples:
```bash
# Store models on host at /data/models and expose to the container
docker run --rm \
  -e MODEL_DIR=/data/models \
  -v /data/models:/data/models \
  docker.io/jakublala/boileroom-chai1:dev python -c "import os; print(os.getenv('MODEL_DIR'))"

podman run --rm \
  -e MODEL_DIR=/data/models \
  -v /data/models:/data/models \
  docker.io/jakublala/boileroom-chai1:dev python -c "import os; print(os.getenv('MODEL_DIR'))"
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
ARG BASE_IMAGE=docker.io/jakublala/boileroom-base:dev
FROM ${BASE_IMAGE}
```
2) Build it locally (adjust path and tag):
```bash
docker build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=docker.io/jakublala/boileroom-base:dev \
  -t docker.io/jakublala/<your_image>:dev \
  -f boileroom/boileroom/models/<your_image>/Dockerfile \
  boileroom/boileroom/models/<your_image>
```
3) Optionally wire it into the dev helper script (`build_all_docker.sh`) after `base` in the correct order.
4) For CI, extend `.github/workflows/build-docker-images.yml` to build and push your image, pinning to `base@sha256:...` similarly to `chai1`.

### 💡 Tips
- Keep network-heavy `pip install` steps in as few layers as possible to improve caching.
- Use `--platform linux/amd64` locally to match CI.
- Prefer digest pinning in CI; use dev tags locally.
