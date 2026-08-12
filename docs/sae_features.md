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
| `sae_layer` | `60` (forge) / per-model (local) | Transformer layer the SAE was trained on. `60` matches the default Forge/ESMC-6B SAE. For the `local` backend it defaults per model to a layer at the same relative depth (`27` for `esmc_600m`, `22` for `esmc_300m`); override to target another layer. An out-of-range layer raises at inference. |
| `normalize_features` | `True` | Set at initialization (not per-call). Forge: TF-IDF normalization. Local: L2-normalize per-residue activations before pooling. |
| `include_per_residue` | `False` | Also return dense per-residue activations. |
| **Forge** `forge_model` | `esmc-6b-2024-12` | Forge ESM-C model id. |
| **Forge** `forge_sae_model` | `esmc-6b-2024-12-sae-layer60-k64-codebook16384` | Forge SAE model id. |
| **Forge** `forge_url` | `https://biohub.ai` | Forge base URL. |
| **Forge** `forge_token` | `None` | API token; falls back to `ESM_API_KEY`. |
| **Local** `esmc_model_name` | `esmc_600m` | ESM-C variant (`esmc_300m` / `esmc_600m`); selects the local `sae_repo_id` / `sae_layer` defaults. |
| **Local** `sae_repo_id` | per-model (`biohub/ESMC-600M-sae-k64-codebook16384` for `esmc_600m`) | HF repo with per-layer weights; defaults to the repo matching `esmc_model_name` unless set. Confirm ids against the [Biohub SAE collection](https://huggingface.co/collections/biohub/esmc-saes-for-hidden-states-all-layers). |
| **Local** `activation` | `topk` | `topk` or `relu`. |

## Weights (local backend)

`SparseAutoencoder.from_pretrained(repo_id, layer=..., num_features=..., d_model=...)`
downloads one `layer_{layer}.safetensors` file and remaps common parameter names
(`W_enc`/`b_enc`/`W_dec`/`b_pre` and encoder/decoder aliases), transposing weight
matrices when a checkpoint stores them in the opposite orientation. Local
checkpoints round-trip through `SparseAutoencoder.save` / `.load`, which the test
suite uses to exercise the model without network access.

The `sae_module` submodule (autoencoder math) and `forge` submodule (API client)
are free of Modal / heavy `esm` imports at module scope, so the core is
unit-testable with fakes — no GPU, token, or network.
