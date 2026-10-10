# SAE features (`SAE`)

The `SAE` model maps an amino-acid sequence to **sparse-autoencoder (SAE)
features** of the ESM-C representation space. It reproduces the sparse-coding
analysis from *Language Modeling Materializes a World Model of Protein Biology*
(Biohub, 2026): a sparse autoencoder is trained per ESM-C transformer layer and
decomposes each residue representation into a high-dimensional, sparse,
interpretable feature basis (`k = 64` active features per residue, codebook of
`2**14 = 16384` features).

A protein is summarized by **max-pooling** each feature across its residues into a
single per-protein feature vector (`pooled_features`).

## Two feature sources

Select the backend with `feature_source`:

- **`"forge"` (default)** — call Biohub's hosted `ESMCForgeInferenceClient` with a
  `SAEConfig` and read back SAE activations. This is the **only** way to use the
  **ESMC-6B layer-60** SAE featured in the paper (the default), which is too large
  to run locally. Requires a Biohub API token (`forge_token` config or the
  `ESM_API_KEY` env var).
- **`"local"`** — run ESM-C locally / on Modal to get per-layer hidden states, then
  apply a local `SparseAutoencoder` loaded from the Biohub per-layer Hugging Face
  weights. Works for the **300M / 600M** SAEs on a single GPU, no token needed.

## Usage

```python
from boileroom import SAE

# Default: ESMC-6B / layer 60 via Forge (needs ESM_API_KEY or config forge_token)
model = SAE(config={"forge_token": "..."})
result = model.get_features("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")
result.pooled_features.shape   # (1, 16384)

# Local backend with the 600M SAE (no token, needs a GPU)
local = SAE(config={
    "feature_source": "local",
    "esmc_model_name": "esmc_600m",
    "sae_repo_id": "biohub/ESMC-600M-sae-k64-codebook16384",
    "sae_layer": 27,
})
```

Request dense per-residue activations with `options={"include_per_residue": True}`.

## Configuration

| Key | Default | Notes |
| --- | --- | --- |
| `feature_source` | `forge` | `forge` or `local`. |
| `num_features` | `16384` | Codebook / feature-space size. |
| `k` | `64` | Active features per residue (TopK). |
| `sae_layer` | `60` (forge) / per-model (local) | Layer the SAE was trained on, in Biohub's numbering (see [Layer numbering](#layer-numbering)). `60` matches the default Forge/ESMC-6B SAE. For the `local` backend it defaults per model to a layer at the same relative depth (`27` for `esmc_600m`, `22` for `esmc_300m`); override to target another layer. An out-of-range layer raises at inference. |
| `normalize_features` | `True` (forge) / `False` (local) | Set at initialization (not per-call). Scales each feature as `activation / max * idf`, where `max` is the feature's maximum activation and `idf = log(N / f)` its inverse document frequency over UniRef90. Forge applies it server-side. Locally it needs those statistics in the checkpoint; the released all-layer checkpoints only carry placeholders (all ones), so `normalize_features=True` raises at load instead of returning raw activations. |
| `include_per_residue` | `False` | Also return dense per-residue activations. |
| **Forge** `forge_model` | `esmc-6b-2024-12` | Forge ESM-C model id. |
| **Forge** `forge_sae_model` | `esmc-6b-2024-12-sae-layer60-k64-codebook16384` | Forge SAE model id. |
| **Forge** `forge_url` | `https://biohub.ai` | Forge base URL. |
| **Forge** `forge_token` | `None` | API token; falls back to `ESM_API_KEY`. |
| **Local** `esmc_model_name` | `esmc_600m` | ESM-C variant (`esmc_300m` / `esmc_600m`); selects the local `sae_repo_id` / `sae_layer` defaults. |
| **Local** `sae_repo_id` | per-model (`biohub/ESMC-600M-sae-k64-codebook16384` for `esmc_600m`) | HF repo with per-layer weights; defaults to the repo matching `esmc_model_name` unless set. Confirm ids against the [Biohub SAE collection](https://huggingface.co/collections/biohub/esmc-saes-for-hidden-states-all-layers). |
| **Local** `activation` | `topk` | `topk` (what the released SAEs are trained with) or `relu` (experimental; does not reproduce Biohub's features). |

## Weights (local backend)

The local `SparseAutoencoder` reproduces Biohub's reference implementation,
`esm.models.esmc.sae.EsmcSaeLayer` (`esm==3.4.1.post1`):

```text
z        = (x - mean(x)) / (std(x) + 1e-5)    # per-residue z-score
features = topk_k(relu((z - b_dec) @ W_enc))   # no encoder bias
recon    = features @ W_dec + b_dec            # reconstructs z
```

Each released layer checkpoint (`layer_{N}.safetensors`) holds exactly these
float32 tensors (shown for the 600M SAE; the 300M SAE has `d_model = 960`):

| Tensor | Shape | Role |
| --- | --- | --- |
| `W_enc` | `(1152, 16384)` | Encoder. |
| `W_dec` | `(16384, 1152)` | Decoder. |
| `b_dec` | `(1152,)` | Decoder bias, subtracted before encoding. |
| `idf` | `(16384,)` | Normalization statistic; all ones (placeholder) in the all-layer releases. |
| `max` | `(16384,)` | Normalization statistic; all ones (placeholder) in the all-layer releases. |

`SparseAutoencoder.from_pretrained(repo_id, layer=...)` reads the repo's
`config.json` (`d_model`, `codebook_dim`, `k`, `available_layers`) and downloads
one `layer_{layer}.safetensors`. Loading is strict: a missing or extra tensor
(e.g. an encoder bias `b_enc`), a wrong shape, a layer the repo does not ship, an
SAE trained on residual updates, or a `d_model` / `num_features` / `k` that
disagrees with the repo config raises instead of being renamed, transposed,
defaulted or ignored. Local checkpoints round-trip through
`SparseAutoencoder.save` / `.load`.

### Layer numbering

Biohub numbers SAE layers by ESM-C's native hidden-state stack: layer `N` is the
input of transformer block `N` (the output of `N` blocks), so a model with `B`
blocks has SAE layers `0..B` (`0..36` for 600M, `0..30` for 300M), and layer `B`
is the final, layer-normed embedding. The local backend therefore feeds the SAE
for layer `N` the output of block `N - 1`, and the final embeddings for layer
`B`. Layer `0` (the token embeddings) is not exposed by ESM-C and raises.

Biohub's hosted SAEs use layer 27 for 600M (boileroom's local default too) and
layer 23 for 300M; boileroom's local 300M default is layer 22.

### Reference tests

`tests/data/sae` holds features that Biohub's own pipeline (`EsmcModel` with the
SAE attached, float32 on the CPU) computes for the local default layers, written
by `scripts/testing/sae_biohub_reference.py`. `tests/sae/test_sae_reference.py`
checks the released checkpoints against them, both on the CPU from the stored
layer states (exact match) and end to end on a backend (`-m integration`).
On the GPU, ESM-C runs in bfloat16, so a few features near each residue's top-k
cut differ from the float32 reference.

The `sae_module` submodule (autoencoder math) and `forge` submodule (API client)
are free of Modal / heavy `esm` imports at module scope, so the core is
unit-testable with fakes — no GPU, token, or network.
