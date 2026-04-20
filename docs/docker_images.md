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
uv run python scripts/images/build_model_images.py --verbose ...
uv run python scripts/images/build_model_images.py --all-cuda --tag=0.3.0 --push ...
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --tag=0.3.0 --push --local-base ...
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --tag=sha-$(git rev-parse --short HEAD) --push ...
```

Images publish to `docker.io/jakublala` by default. If `--tag` is omitted, image helpers use the current boileroom package version from `pyproject.toml`. Pass `--docker-user` and `--tag` to build or publish a specific tag under another Docker Hub namespace:

```bash
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --tag=0.3.0 --docker-user=my-dockerhub-user --push
```

The image build, smoke check, and promotion helpers all accept the same `--docker-user` flag.

Modal pytest uses `docker.io/jakublala` plus the current package version by default. To run against manually published images, pass the same user and tag:

```bash
uv run pytest --backend=modal --docker-user=my-dockerhub-user --image-tag=0.3.0
```

Single-platform non-push builds auto-load into the local Docker daemon. Multi-platform builds should generally be paired with `--push`.
Pushed buildx builds import and export stable per-image registry caches such as `boileroom-chai1:buildcache-cuda12.6`, so GitHub Actions runners can reuse dependency layers across validation tags and releases. Pass `--no-cache` to bypass those caches.
Pass `--verbose` to stream Docker build output and plain BuildKit progress while still writing per-image log files.
In CI, the release workflow splits CUDA lines across separate GitHub-hosted runners and passes `--max-workers` within each CUDA job. That keeps the base image dependency order intact while letting model image builds and Docker Hub transfers overlap.
For single-platform publishing, pass `--local-base` to build and tag images with `buildx --load` before pushing. This keeps dependent model builds from waiting on Docker Hub to receive and then re-serve the base image. Model builds also receive the loaded base tag as a named `docker-image://` build context so their `FROM` instruction resolves locally while preserving BuildKit registry cache import/export.

### ARM64 smoke workflow
The `.github/workflows/arm64-image-smoke.yml` workflow runs on pull requests to `main` and on manual dispatch. It uses an `ubuntu-24.04-arm` runner, builds the image set for `linux/arm64` with the `arm64-ci` tag, and then runs the import and server-health smoke checks. It is informational and does not push images.

On `main`, ARM64 image smoke is folded into the Docker publishing workflow instead of running as a second separate workflow. That keeps the branch smoke path fast and local while making release promotion wait for the same ARM64 smoke coverage.

To reproduce the same path locally on an ARM64 machine, run:

```bash
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --tag=arm64-ci --platform=linux/arm64 --max-workers=3
uv run python scripts/images/check_model_imports.py --cuda-version=12.6 --tag=arm64-ci
uv run python scripts/images/check_model_server_health.py --cuda-version=12.6 --tag=arm64-ci
```

The build helper also supports `--skip-existing` and `--force-rebuild` for registry-aware rebuilds.

### 🔖 Tag policy
- Docker Hub is kept clean for users. The long-lived public tags are version tags such as `0.3.0` and the corresponding CUDA-qualified tags such as `cuda12.6-0.3.0` and `cuda11.8-0.3.0`.
- Short-lived validation tags such as `sha-<shortsha>` are fine when you need to test a branch through Docker Hub or Modal before promoting a version tag.
- Validation tags should be deleted once the validation pass is complete.

For example, a temporary validation push on the default CUDA line:
```bash
TAG=sha-$(git rev-parse --short HEAD)
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --tag="$TAG" --platform=linux/amd64 --push
```

Because `12.6` is the default CUDA line, publishing `--tag="$TAG"` also creates the explicit `cuda12.6-$TAG` tag alongside the unqualified alias.

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
- Manual runs can also be dispatched from a non-`main` branch with `promote` left disabled. That validation-only path builds and pushes temporary `sha-<commit>` validation images, runs the local AMD64 and ARM64 smoke checks, and skips public version-tag promotion.
- Builds a temporary `sha-<commit>` validation tag, verifies that exact pushed artifact, and only then promotes it to an automatically derived `0.3.x` version from `scripts/ci/derive_version.py`.
- Builds each CUDA line in its own job, with model images parallelized behind the matching locally available base image by `--local-base` and `--max-workers`.
- Verifies canonical CUDA-qualified tags from the same runner-local images after each CUDA build. The default-CUDA alias is checked locally in the `12.6` job.
- Runs the ARM64 smoke build and checks in the same publishing workflow on `main`; the standalone ARM64 workflow is reserved for pull requests and manual runs.
- The `0.3.x` patch component is the number of commits after the configured main-line baseline, so it increases with every new commit on `main`.
- Each successful run publishes canonical CUDA-qualified tags and the unqualified version alias for the default `12.6` line.
- The official release path currently publishes `linux/amd64` only. If you want to experiment with additional architectures, pass an explicit multi-platform `--platform` value and validate it separately before treating it as supported.
- Future merges inherit dependency cache layers through BuildKit registry caches, keeping CI times reasonable even on fresh GitHub-hosted runners.
- PyPI is not published by this workflow. Python package publication happens later from the GitHub release workflow, which injects the `0.3.x` release tag into `pyproject.toml` before building.

To test the publishing workflow before merging:
1. Push your branch.
2. Open **Build and Push Boileroom Images** in GitHub Actions.
3. Choose **Run workflow**, select your branch, leave `promote` disabled, and optionally set `docker_repository` to a temporary namespace such as `docker.io/my-dockerhub-user`.
4. After the run, delete the temporary `sha-<commit>` validation tags if you no longer need them.

The same branch validation run can be triggered with GitHub CLI:
```bash
gh secret set DOCKERHUB_TEST_TOKEN

gh workflow run build-docker-images.yml \
  --ref "$(git branch --show-current)" \
  -f promote=false \
  -f docker_repository=docker.io/my-dockerhub-user \
  -f dockerhub_username=my-dockerhub-user \
  -f dockerhub_token_secret=DOCKERHUB_TEST_TOKEN
```

The `gh secret set` command reads the token from your terminal and stores it as a repository secret. Avoid passing Docker Hub tokens through workflow inputs because inputs are visible in run metadata.

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
