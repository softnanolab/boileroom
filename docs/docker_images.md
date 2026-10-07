## Boileroom Docker images

### What exists today
- **base**: `boileroom/images/Dockerfile` → Python 3.12 slim base with shared OS build/runtime tools. Tag: `docker.io/jakublala/boileroom-base`.
- **alphafold**: `boileroom/models/alphafold/Dockerfile` → installs ColabFold (`colabfold_batch`) plus `jax[cuda12]`; MSAs come from the ColabFold MMseqs2 server, so no local genetic databases, HMMER, HH-suite, or Kalign are installed. Tag: `docker.io/jakublala/boileroom-alphafold2-multimer`. Platform: `linux/amd64`.
- **boltz**: `boileroom/models/boltz/Dockerfile` → installs Boltz runtime dependencies from `requirements.txt`. Tag: `docker.io/jakublala/boileroom-boltz`.
- **chai1**: `boileroom/models/chai/Dockerfile` → installs Chai runtime dependencies from `requirements.txt`, sets HF env vars. Tag: `docker.io/jakublala/boileroom-chai1`.
- **esm**: `boileroom/models/esm/Dockerfile` → installs ESM runtime dependencies from `requirements.txt` shared by esm2/esmfold. Tag: `docker.io/jakublala/boileroom-esm`.
- **opendde**: `boileroom/models/opendde/Dockerfile` → OpenDDE 1.1.1 and the Anthropic kit stack (Python 3.11, torch 2.7.1+cu126, cuequivariance 0.10.0) in `/opt/opendde`, CUDA 12.6 `nvcc` for the Triton JIT, GCC 13 libstdc++ for `exact`, HMMER/Kalign. No `LAYERNORM_TYPE` is set image-wide: the worker sets it per mode (torch LayerNorm for `vanilla`, upstream's fused `fast_layernorm` for `exact` and `fast`, JIT-built at first use with the venv's `ninja`; see [optimization.md](optimization.md#layernorm-per-mode)). Python 3.11.5 there is python-build-standalone; the boileroom server runs on the base image's Python 3.12. Tag: `docker.io/jakublala/boileroom-opendde`. Platform: `linux/amd64`.
- **protenix**: `boileroom/models/protenix/Dockerfile` → installs Protenix plus HMMER/Kalign CLI dependencies. Tag: `docker.io/jakublala/boileroom-protenix`. Platform: `linux/amd64`.
- **esmfold2**: `boileroom/models/esmfold2/Dockerfile` → installs the MIT-licensed 2026 Chan Zuckerberg Biohub `esm` package (`esm==3.4.1.post1`, torch 2.11, CUDA 12.6 only) from `requirements.txt`. Tag: `docker.io/jakublala/boileroom-esmfold2`. **Shared by ESMFold2, ESM-C, and ESM3** — all three use the same Biohub `esm` package, so ESM-C/ESM3 run on this image instead of a separate one.

- **esmfold2-kit** and **protenix-kit** (opt-in, built by hand rather than by CI): `boileroom/models/esmfold2/kit/Dockerfile` and `boileroom/models/protenix/kit/Dockerfile` → the stack of the Anthropic optimization kit (torch 2.13+cu130, kit commit `f4f62fa`) behind `optimization="exact"` and `"fast"`. Names: `boileroom-esmfold2-kit` and `boileroom-protenix-kit`. Used only when a kit mode is requested; `optimization="vanilla"` (the default) keeps using the stock images above. Registry pulls name them by the digest pinned in `KIT_IMAGE_DIGESTS`, not by the release tag, and `BOILEROOM_IMAGE_TAG` does not apply to them. They are not built or smoke-tested by `build_model_images.py` or CI; a full release tags their pinned digests with `X.Y.Z` (`promote_image_tags.py --kit-images-only`) and the tag cleanup keeps every digest ever pinned: see [Kit images](#kit-images-optimizationexact-and-fast).

Dockerfiles are the canonical image definition for all runtimes. Docker/Apptainer images are built from these Dockerfiles, and Modal pulls the corresponding published model image from Docker Hub instead of maintaining a separate handwritten dependency stack. CUDA variants select the PyTorch wheel index; the runtime images rely on PyTorch/NVIDIA wheels for user-space CUDA libraries and on Docker/Apptainer GPU integration for host driver libraries.

### Tag scheme
- Canonical published tags are CUDA-qualified, for example `cuda12.6-0.3.0`, `cuda11.8-0.3.0`, `cuda12.6-0.3.1-alpha.1`, or `cuda12.6-sha-abc1234`.
- The default CUDA line is `12.6`. That line also gets an unqualified alias for the exact package version, alpha prerelease, or temporary validation tag, for example `0.3.0`, `0.3.1-alpha.1`, or `sha-abc1234`.
- `latest` is not published.
- Runtime shorthands such as `backend="apptainer"` resolve through `BOILEROOM_IMAGE_TAG` when set, otherwise through the installed boileroom package version on the default `12.6` CUDA line. This applies to the stock images only; the kit images follow their own rules (see [Kit images](#kit-images-optimizationexact-and-fast)).

### Using prebuilt images when your checkout is ahead of the last release
The default tag is the `version` in `pyproject.toml` (or the installed package version). Between releases that stable tag, for example `0.4.3`, is **not published**: `main` only publishes alpha tags such as `0.4.3-alpha.8`. A default lookup then fails to find the image, and the only way forward looks like building it locally (30–45 minutes). You do not need to build anything. Point the runtime at an existing published tag instead.

Find a published tag on [Docker Hub](https://hub.docker.com/r/jakublala/boileroom-esmfold2/tags) (swap in the model's image name) and pick the newest `X.Y.Z-alpha.N` or the latest stable release. Then use any one of:

```bash
export BOILEROOM_IMAGE_TAG=0.4.3-alpha.8          # Modal and Apptainer
uv run pytest --image-tag 0.4.3-alpha.8 ...       # pytest, both backends
```

```python
ESMFold2(backend="apptainer:0.4.3-alpha.8")        # inline tag, wins over the env var
```

The tag must contain the model dependencies you need. Older tags contain older dependency stacks (for example, the `esm 3.4.1.post1` port of ESMFold2 is not in `0.4.1`), so prefer the newest alpha when you are on `main`. The first pull is large (the `esmfold2` image is about 4.3 GB compressed) but is cached afterwards.

### 🚀 Quick start
Use the Python helper to build all images (base + models) with a single global worker limit.

```bash
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --platform=linux/amd64 --max-workers=1

# Optional flags
uv run python scripts/images/build_model_images.py --no-cache ...
uv run python scripts/images/build_model_images.py --verbose ...
uv run python scripts/images/build_model_images.py --all-cuda --tag=0.3.0 --push ...
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --tag=0.3.0 --push --local-base ...
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --tag=sha-$(git rev-parse --short HEAD) --push ...
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --model=esmfold2 ...
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --base-mode=only --push ...
```

Images publish to `docker.io/jakublala` by default. If `--tag` is omitted, image helpers use the current boileroom package version from `pyproject.toml`. Pass `--docker-user` and `--tag` to build or publish a specific tag under another Docker Hub namespace:

```bash
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --tag=0.3.0 --docker-user=my-dockerhub-user --push
```

The image build, smoke check, and promotion helpers all accept the same `--docker-user` flag.

Pytest uses `docker.io/jakublala` plus the current package version by default. To run against manually published images, pass `--image-tag` and `--docker-user`:

```bash
uv run pytest --backend=modal --docker-user=my-dockerhub-user --image-tag=0.3.0
```

For the Modal integration suite, use grouped xdist scheduling so each model family runs in its own worker and Modal app:

```bash
uv run pytest -v -n 4 --dist loadgroup -m integration \
  --docker-user=my-dockerhub-user \
  --image-tag=0.3.0 \
  --gpu=A10
```

For serial integration execution against the same image, omit xdist:

```bash
uv run pytest -v -m integration --docker-user=my-dockerhub-user --image-tag=0.3.0 --gpu=A10
```

`--image-tag` is honored by both the Modal and Apptainer test backends. The Apptainer backend additionally accepts an inline tag via `--backend apptainer:<tag>`, which wins over `--image-tag`.

For lower-level runtime configuration outside pytest, `BOILEROOM_IMAGE_TAG` is the shared image-tag override used by both Modal image lookup and Apptainer's default image tag. An explicit Apptainer suffix such as `backend="apptainer:<tag>"` wins over the env override. Prefer pytest's `--image-tag` option for test runs so the selected image is explicit in the test command and report header. Neither `--image-tag` nor `BOILEROOM_IMAGE_TAG` selects a kit image; use `--kit-image-tag` or `BOILEROOM_KIT_IMAGE_TAG` for those.

Single-platform non-push builds auto-load into the local Docker daemon. Multi-platform builds should generally be paired with `--push`.
Pushed buildx builds import and export stable per-image registry caches such as `boileroom-chai1:buildcache-cuda12.6`, so GitHub Actions runners can reuse dependency layers across validation tags and releases. Pass `--no-cache` to bypass those caches.
Model Dockerfiles also mount a BuildKit uv cache scoped to the active CUDA line, for example `boileroom-uv-cu12.6`, so repeated builds can reuse downloaded wheels even when a full dependency-install layer has to run again.
Pass `--verbose` to stream Docker build output and plain BuildKit progress while still writing per-image log files.
In CI, the release workflow first publishes one AMD64 base image per CUDA line, then builds each model/CUDA pair on a fresh runner with `--max-workers=1`. AMD64 base and model jobs push directly from BuildKit under the run's candidate tag (`sha-<12 hex>`, or the full commit SHA on a release), model jobs prune the build cache and pull only their own image for smoke checks, and only after every smoke check passed does `promote_image_tags.promote_one` point the final alpha or stable tags at the verified manifest (the base gets its final tags once it is built; it is smoked through the models). A validation-only run's candidate is its final tag, so it skips that retag. This isolates disk usage so one large model cannot exhaust a runner used by the others, and a model that fails its smoke checks never gets a public version tag.
ARM64 validation also uses one fresh runner per model. A dedicated ARM64 runner builds the base once, exports it as a one-day run artifact, and each model runner loads that exact base locally; the published base tag remains AMD64-only.
For single-platform publishing, pass `--local-base` to build and tag images with `buildx --load` before pushing. This keeps dependent model builds from waiting on Docker Hub to receive and then re-serve the base image. Model builds also receive the loaded base tag as a named `docker-image://` build context so their `FROM` instruction resolves locally while preserving BuildKit registry cache import/export.

### ARM64 smoke workflow
The `.github/workflows/arm64-image-smoke.yml` workflow runs on pull requests to `main` and on manual dispatch. It builds one ARM64 base artifact, then uses one `ubuntu-24.04-arm` runner per model to build `linux/arm64` images with the `arm64-ci` tag and run the import and server-health smoke checks. It is informational and does not push images.

The workflow does not install the full project dependency set on the host runner. Host-side image scripts run with `uv run --no-project --with pyyaml`, while heavy model dependencies such as PyTorch and SciPy are validated inside the Docker images themselves.

Image configs can restrict supported platforms. AlphaFold2-Multimer, Protenix and OpenDDE currently advertise `linux/amd64` only, so ARM64 smoke builds and checks skip those images while still validating the ARM64-compatible model images.

On `main`, ARM64 image smoke is folded into the Docker publishing workflow instead of running as a second separate workflow. That keeps the branch smoke path fast and local while making release promotion wait for the same ARM64 smoke coverage.

To reproduce the same path locally on an ARM64 machine, run:

```bash
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --model=esmfold2 --tag=arm64-ci --platform=linux/arm64 --max-workers=1
uv run python scripts/images/check_model_imports.py --cuda-version=12.6 --model=esmfold2 --tag=arm64-ci
uv run python scripts/images/check_model_server_health.py --cuda-version=12.6 --model=esmfold2 --tag=arm64-ci
```

The build helper also supports `--skip-existing` and `--force-rebuild` for registry-aware rebuilds.

To build or check a subset of images, pass `--only <model>` (repeatable) to `build_model_images.py`, `check_model_imports.py`, and `check_model_server_health.py`. Selectors are model family keys such as `alphafold`, `protenix`, or `boltz`. The shared base image is always built, because every model image starts from it. If `--only` matches nothing on the requested platform (for example `--only alphafold --platform=linux/arm64`), the checks exit cleanly without doing anything:

```bash
uv run python scripts/images/build_model_images.py --cuda-version=12.6 --tag=sha-$(git rev-parse --short HEAD) --push --only alphafold
```

### 🔖 Tag policy
- Docker Hub is kept clean for users. The long-lived public tags are stable version tags such as `0.3.0`, alpha prerelease tags such as `0.3.1-alpha.1`, and the corresponding CUDA-qualified tags such as `cuda12.6-0.3.0` and `cuda11.8-0.3.0`.
- Short-lived validation tags such as `sha-<shortsha>` are fine when you need to test a branch through Docker Hub or Modal before promoting a version tag.
- Validation tags should be deleted once the validation pass is complete.
- The `.github/workflows/cleanup-dockerhub-tags.yml` workflow enforces retention weekly by keeping the latest 3 alpha versions, deleting `sha-*` tags older than 7 days, and preserving stable and `buildcache-*` tags.
- In the kit repositories it also keeps every tag that points at any digest listed in `KIT_IMAGE_DIGEST_HISTORY` (every digest `KIT_IMAGE_DIGESTS` has ever pinned; append-only), so the image an installed release pulls is never left untagged. A kit repository in which no tag points at its current pin is not pruned at all, and the command then exits non-zero naming it; tag the digest again (`promote_image_tags.py --kit-images-only`) or update `KIT_IMAGE_DIGESTS`. An earlier pin that no tag holds is only reported as a warning.
- A full GitHub release runs `promote_image_tags.py --kit-images-only` (job `publish-kit-image-tags`), which gives each kit image's pinned digest the stable `X.Y.Z` tag. A manual promotion does the same after the stock images unless `--skip-kit-images` is passed. Kit tags carry no CUDA qualifier, since the kit stack is CUDA 13.0, and the kit images are never rebuilt per release. The script refuses to run while `BOILEROOM_KIT_IMAGE_TAG` is set, refuses before pushing anything if a kit target tag already names another digest (`--force-kit-tags` moves it), and checks after each push that every target tag serves the source digest.

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

> ESM-C and ESM3 do not have their own image — they run on the `esmfold2` image above (same Biohub `esm` package).

### Kit images (`optimization="exact"` and `"fast"`)
The kit modes need a different stack from the stock images (see [optimization.md](optimization.md)), so ESMFold2 and Protenix have a second, opt-in image each, defined next to their stock Dockerfile in a `kit/` directory. They are not built by `scripts/images/build_model_images.py` and not built or smoke-tested by CI (a full release only tags their pinned digests): the ESMFold2 image compiles flash-attn, TransformerEngine and xformers from source (about 30 minutes on a 48-64 core builder, hours on a GitHub-hosted runner). Each Dockerfile runs a CPU-only import check of its stack at build time, so a kit image whose kernels fail to import does not build.

Where a runtime gets the kit image:

- **Modal** builds the image from the Dockerfile in your installed boileroom the first time a kit mode is used, and caches it. This is the default (`BOILEROOM_KIT_IMAGE_SOURCE=build`). `BOILEROOM_IMAGE_REF` then reads `build:<Dockerfile path>@<hash>`, where the hash covers the Dockerfile, its build context and (for ESMFold2) the build arguments and the compile and finish steps, so it changes whenever the built image would.
- **Modal with `BOILEROOM_KIT_IMAGE_SOURCE=registry`**, and **Apptainer always**, pull the published image from `BOILEROOM_DOCKER_REPOSITORY` (default `docker.io/jakublala`) by the digest pinned in `KIT_IMAGE_DIGESTS` (`boileroom/images/metadata.py`; the digests are listed in [optimization.md](optimization.md#kit-images)).
- To pull a tag instead of the pinned digest, set `BOILEROOM_KIT_IMAGE_TAG=<tag>` (pytest: `--kit-image-tag <tag>`); with Apptainer, `backend="apptainer:<tag>"` names the kit tag too and wins over the env var. `BOILEROOM_IMAGE_TAG` and `--image-tag` never apply to kit images. Pytest refuses `--kit-image-tag` while Modal builds the image from source, since the tag would be ignored.

To use your own build, build and push it, then point the runtime at it:

```bash
docker build -f boileroom/models/protenix/kit/Dockerfile boileroom/models/protenix/kit -t <repository>/boileroom-protenix-kit:<tag>
docker build -f boileroom/models/esmfold2/kit/Dockerfile boileroom/models/esmfold2/kit -t <repository>/boileroom-esmfold2-kit:<tag>
docker push <repository>/boileroom-protenix-kit:<tag>
docker push <repository>/boileroom-esmfold2-kit:<tag>

export BOILEROOM_KIT_IMAGE_SOURCE=registry BOILEROOM_DOCKER_REPOSITORY=<repository> BOILEROOM_KIT_IMAGE_TAG=<tag>
```

With `backend="apptainer"` the kit image is pulled as `docker://<repository>/boileroom-<family>-kit@sha256:<digest>` (or `:<tag>`); the interpreter is `/usr/local/bin/python3.11` in the Protenix kit image and `python3.12` in the ESMFold2 one.

The Dockerfiles pin the kit commit; `KIT_COMMIT` in `boileroom/images/metadata.py` must match it (`tests/contracts/test_kit_images.py` checks this). After rebuilding and pushing a kit image, update its digest in `KIT_IMAGE_DIGESTS` and append it to `KIT_IMAGE_DIGEST_HISTORY` (never remove the old entry); until then installations keep pulling the old one.

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
GitHub Actions at `.github/workflows/build-docker-images.yml` now drives the image publishing pipeline:
- Triggers automatically on non-documentation pushes to `main`, on published GitHub releases, and can also be run manually via **Run workflow** from `main`.
- Manual runs can also be dispatched from a non-`main` branch with `promote` left disabled. That validation-only path builds and pushes temporary `sha-<commit>` validation images, runs the AMD64 and ARM64 smoke checks, and skips public version-tag publishing. The `models` dispatch input (space-separated family keys, e.g. `alphafold protenix`) limits a validation-only run to those images; promoted runs always build everything.
- Pushes to `main` build and validate an automatically derived alpha prerelease tag from `scripts/ci/derive_version.py`, such as `0.4.3-alpha.1`. Full GitHub releases build and validate the stable release tag.
- Publishes one AMD64 base image per CUDA line, then builds every supported model/CUDA pair in a separate matrix job with `--max-workers=1`.
- Prunes BuildKit state before verification and pulls only the selected model image. The default-CUDA alias is checked in that model's `12.6` job.
- Builds the ARM64 base once per run and shares it as a short-lived artifact across isolated ARM64 model jobs.
- Runs the ARM64 smoke build and checks in the same publishing workflow on `main`; the standalone ARM64 workflow is reserved for pull requests and manual runs.
- The alpha suffix counts commits since the latest reachable stable release tag, for example `0.3.1-alpha.1`, `0.3.1-alpha.2`, and so on. Before the first stable release tag, the count falls back to the configured CI baseline.
- README- and docs-only commits reuse the alpha tag of the latest image-changing commit on the first-parent history, matching the publishing workflow's path filter. Scheduled tests resolve that tag from their checked-out commit. The next image build still counts intervening documentation commits, preserving existing tag numbering.
- Each successful run publishes canonical CUDA-qualified tags and the unqualified version alias for the default `12.6` line.
- The official release path currently publishes `linux/amd64` only. If you want to experiment with additional architectures, pass an explicit multi-platform `--platform` value and validate it separately before treating it as supported.
- Future merges inherit dependency cache layers through BuildKit registry caches, keeping CI times reasonable even on fresh GitHub-hosted runners.
- Published full GitHub releases from `vX.Y.Z` tags publish the stable `X.Y.Z` Docker tag.
- GitHub releases marked as pre-releases do not publish stable Docker or PyPI artifacts.
- PyPI is not published by this workflow. Python package publication happens from the separate GitHub release workflow, which injects the stable release tag into `pyproject.toml` before building.

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

#### How `backend="apptainer"` caches images
You do not need to pull by hand for `backend="apptainer"`; the backend pulls on first use and reuses the file afterwards.

- The cache directory is the `cache_dir` argument, else `MODEL_DIR`, else `~/.cache/boileroom`. Images go to `<cache>/images/<repository with / replaced by ->_<tag>.sif`, for example `jakublala-boileroom-chai1_0.3.0.sif`, or `..._sha256-<hex>.sif` for a digest reference such as a kit image. A registry other than `docker.io` is prefixed to the name.
- A tag is resolved to its digest through the registry first and that digest is pulled, then recorded beside the image as `<name>.sif.digest`. Inside the container `BOILEROOM_IMAGE_REF` is `<repository>/<image>:<tag>@sha256:<hex>` when the digest was recorded, and the reference as given otherwise; it reaches `metadata.runtime["image_ref"]`.
- A cached `.sif` is reused as is, even if the tag has since moved. Delete the `.sif` (and its `.digest`) to pull again.
- The server must answer its health check within `BOILEROOM_APPTAINER_STARTUP_TIMEOUT` seconds (default 1800, which covers a first kit run downloading its weights); the value must be a positive, finite number. If the server exits with code 3 during startup, the requested optimization mode was refused and `OptimizationUnavailableError` is raised; any other early exit raises `RuntimeError` with the tail of the log.

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
