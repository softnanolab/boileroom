## Boileroom Docker images

### Structure
- **base**: Defined in `boileroom/boileroom/images/Dockerfile`. Contains CUDA-enabled micromamba base, system deps, and Python (pip upgraded) with core libs like `biotite`. Tagged as `ghcr.io/boileroom/base`.
- **chai1**: Defined in `boileroom/boileroom/models/chai/Dockerfile`. Builds on `base` and installs `torch==2.5.1+cu118`, `hf_transfer`, and `chai_lab`. Tagged as `ghcr.io/boileroom/chai1`.

Images publish to GHCR via the GitHub Actions workflow at `.github/workflows/build-chai-images.yml`. The workflow:
- Builds `base` first and records its content digest.
- Builds `chai1` pinned to the exact `base@sha256:...` digest for reproducibility.
- Uses BuildKit cache (`type=gha`) for faster incremental builds.

### Local development builds
You can build locally with Docker.

#### Build base locally
```bash
docker build \
  --platform linux/amd64 \
  -t ghcr.io/boileroom/base:dev \
  -f boileroom/images/Dockerfile \
  boileroom/images
```

Optionally push (requires `ghcr.io` login):
```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u <github-username> --password-stdin
docker push ghcr.io/boileroom/base:dev
```

#### Build chai1 locally (using base tag or digest)
Using a tag:
```bash
docker build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=ghcr.io/boileroom/base:dev \
  -t ghcr.io/boileroom/chai1:dev \
  -f boileroom/models/chai/Dockerfile \
  boileroom/models/chai
```

Using an exact digest for reproducibility:
```bash
# After pushing base, get the digest
BASE_DIGEST=$(docker buildx imagetools inspect ghcr.io/boileroom/base:dev | awk '/Digest:/ {print $2; exit}')

docker build \
  --platform linux/amd64 \
  --build-arg BASE_IMAGE=ghcr.io/boileroom/base@${BASE_DIGEST} \
  -t ghcr.io/boileroom/chai1:dev \
  -f boileroom/boileroom/models/chai/Dockerfile \
  boileroom/boileroom/models/chai
```

### Notes and tips
- `.dockerignore` files keep the build context minimal (only `Dockerfile`), which speeds up remote builds and improves cache hits.
- Prefer pinning to image digests (`@sha256:...`) in CI for deterministic builds.
- Keep network-heavy `pip install` commands grouped in as few layers as practical to improve caching.
- Use `--platform linux/amd64` locally to match CI environment.

