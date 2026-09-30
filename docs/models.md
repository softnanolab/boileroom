# Model Examples


### Boltz
- Triton is not supported with arm-based architecture on the current version of Python/Torch. To run, for instance, on GH200 GPUs, one has to set `no_kernels` to True in the `config` during model instantiation.

Example Usage:
```python
os.environ['MODEL_DIR'] = "/.model_cache" # somewhere with a lot of storage
model = Boltz2(backend='apptainer', device="cuda:0", config={"no_kernels": True})
result = model.fold(
    sequence=['MLKNVHVLVLGAGDVGSVVVRLLEK'],
    options={
        "include_fields": ["plddt", "pae"]
    }
    )

result.atom_array
result.plddt
result.pae

```

Structure-wrapper confidence metrics use a shared output contract. `plddt` is returned as one unit-scale
per-residue array on `[0, 1]` per sample, while scalar scores such as `ptm` and `iptm` are shape-`(1,)` arrays.
In `0.3.1`, this replaces ESMFold's old padded pLDDT batch array and moves Boltz `ptm`/`iptm` from nested
`confidence` dictionaries to top-level fields.

### Protenix
`Protenix` loads the official Python inference runner (`protenix==2.0.0`, default checkpoint `protenix-v2`) once per backend instance and reuses its weights across calls. A single
`fold()` call accepts one sequence entry; use `:` to join multiple protein chains. On Modal it defaults to an
`A100-40GB` GPU.

Keep one model context open for all of your jobs. The first Modal call starts the GPU worker and loads
the checkpoint; subsequent calls reuse that worker's runner. The Apptainer service loads it at startup.
No separate job-submission API is needed: use the usual `fold()` method for each new input.

Example usage:
```python
from boileroom import Protenix

query = "MLKNVHVLVLGAGDVGSVVVRLLEK:MLKNVHVLVLGAGDVGSVVVRLLEK"
with Protenix(
    backend="modal",
    config={
        "model_name": "protenix-v2",
        "use_msa": True,
        "sample": 1,
        "cycle": 10,
        "step": 200,
    },
) as model:
    # Supply a different query for each job, or repeat one with another seed.
    results = [
        model.fold(
            query,
            options={"seeds": str(seed), "include_fields": ["ptm", "iptm", "pae", "token_chain_ids", "cif"]},
        )
        for seed in (101, 102, 103)
    ]

result = results[0]
result.atom_array
result.iptm             # shape-(1,) arrays, one per ranked sample
result.pae              # (tokens, tokens) PAE matrix per sample
result.token_chain_ids  # per-token chain labels for interface scoring (e.g. ipSAE)
```

Outputs are collected from every seed/sample directory in rank order. Besides `ptm`/`iptm`/`plddt`, Protenix
exposes the raw `confidence` summary, `pae`, `token_chain_ids`/`token_res_ids`, `atom_plddt`, `seeds`,
`sample_ranks`, and `pdb`/`cif`.

MSA handling:
- **Server (default):** `use_msa=True` sends the search to the ColabFold MMseqs2 server at `msa_server_url`
  (default `https://api.colabfold.com`, the same as AlphaFold2-Multimer and Boltz). Protenix's own default
  server can queue jobs for a long time, so Boileroom points the worker at the configured server via
  `MMSEQS_SERVICE_HOST_URL`.
- **Provided MSA:** pass `config={"unpaired_msa": [a3m_chain_a, None, ...]}` with one A3M string (or `None` for
  query-only) per chain; no server call is made.
- **Single sequence:** set `use_msa=False`.

Sampling is controlled by `seeds` (comma-separated), `sample` (diffusion samples per seed, default 5), `cycle`
(recycles, default 10) and `step` (diffusion steps, default 200). Lower values trade accuracy for speed. Template
and RNA-MSA searches require the external tools and databases expected by Protenix. The runtime image installs
`hmmer` and `kalign`, but database paths still need to be available inside the container when those features are
enabled.

Protenix keeps its runner in a persistent worker, retaining a hard `timeout_seconds` limit (3500 seconds by default, `None` disables it). A timeout or inference failure discards the worker; the next call reloads cleanly. Use one model context for repeated predictions. Seeds, cycle/step/sample counts, dtype, MSA inputs and output selection remain per-call options; each request gets fresh output paths and inference settings.

The prerelease CLI setting `protenix_command` has been removed. `model_name`, `device`, `msa_server_url`, `use_template`, `trimul_kernel`, `triatt_kernel`, `enable_cache`, `enable_fusion`, and `enable_tf32` are now initialization-only settings. Create a new instance to change them.

### OpenDDE
`OpenDDE` wraps the AF3-style [OpenDDE](https://github.com/aurekaresearch/OpenDDE) runner (`opendde==1.1.1`, single released model `opendde_v1`). Its runner API, JSON input and output files match Protenix 2.0, so it shares Protenix's interface end to end: one sequence entry per call with `:` joining chains, `unpaired_msa` A3M passthrough, the ColabFold MMseqs2 server via `msa_server_url`, the persistent worker with `timeout_seconds`, and the same `OpenDDEOutput` fields (`atom_array`, `confidence`, `plddt`, `ptm`, `iptm`, `pae`, `token_chain_ids`, `token_res_ids`, `atom_plddt`, `seeds`, `sample_ranks`, `pdb`, `cif`). Differences: `dtype` is `bf16` (default) or `fp32`, `trimul_kernel`/`triatt_kernel` default to `auto`, and the runner lives in its own Python 3.11 virtualenv (`opendde_python`, initialization-only). Weights download to `$MODEL_DIR/opendde` on first use.

```python
from boileroom import OpenDDE

with OpenDDE(backend="modal", config={"optimization": "fast"}) as model:
    result = model.fold("SEQ_A:SEQ_B", options={"include_fields": ["pae", "token_chain_ids", "ptm", "iptm"]})
```

`optimization="exact" | "fast"` uses the Anthropic OpenDDE kit (A100/H100/H200; see [optimization.md](optimization.md)). Fused LayerNorm (`LAYERNORM_TYPE=fast_layernorm`) is used in every mode and JIT-compiles once into `$MODEL_DIR/opendde/jit`.

### AlphaFold2-Multimer
`AlphaFold2Multimer` keeps ColabFold's Python model runners and parameters resident, using `alphafold2_multimer_v3`
by default. Repeated `fold()` calls reuse the same runners and JAX compilation cache. MSAs are
fetched from the public ColabFold MMseqs2 server (`https://api.colabfold.com`), so **no local genetic databases
(the ~2.6 TB AlphaFold data tree) are required**. AlphaFold model parameters are cached under `data_dir`
(default `${MODEL_DIR}/alphafold`); ColabFold downloads them there on first use. The wrapper accepts one
top-level sequence entry and uses `:` to split chains. On Modal it defaults to an `A100-80GB` GPU.

ColabFold runs in its own Python 3.10 environment inside the image, pinned to a fixed commit, because it needs
Python < 3.12 and `pandas<2`. By default ColabFold turns on CUDA unified memory (`TF_FORCE_UNIFIED_MEMORY=1`,
memory fraction 4.0). This stalls parameter loading on Modal, so Boileroom runs its resident worker with
`TF_FORCE_UNIFIED_MEMORY=0` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`. Values set in the caller's environment still
take precedence.

Example usage:
```python
from boileroom import AlphaFold2Multimer

query = "MLKNVHVLVLGAGDVGSVVVRLLEK:MLKNVHVLVLGAGDVGSVVVRLLEK"
with AlphaFold2Multimer(
    backend="modal",
    config={
        "num_models": 5,
        "num_recycle": 3,
        "use_amber": False,
    },
) as model:
    results = [
        model.fold(
            query,
            options={"random_seed": seed, "include_fields": ["ranking", "plddt", "iptm", "pae", "cif"]},
        )
        for seed in (0, 1, 2)
    ]

result = results[0]
result.atom_array
result.ranking
result.plddt
result.iptm
```

MSA handling mirrors the other adapters:
- **Server (default):** alignments come from the ColabFold MMseqs2 server and are cached, content-addressed, under
  `${data_dir}/msa_cache` so repeat folds of the same complex and MSA settings skip the server.
  Cache identity includes `msa_server_url`; switching providers fetches a fresh alignment. Older entries
  without a provider in their key are ignored.
- **Provided MSA:** pass `options={"msa": MSAInput(path="complex.a3m")}` (or `MSAInput(sequences=[...])`) to supply a
  ColabFold-compatible complex a3m directly; the server is not queried.
- **Single sequence:** set `config={"use_msa_server": False}` to run without an alignment.

Set `use_templates=True` to enable ColabFold templates and `use_amber=True` (optionally `use_gpu_relax=True`) for
Amber relaxation of the ranked predictions.

Other config keys: `num_models` (1–5, default 5), `num_recycle` (default 3), `num_seeds`, `random_seed`,
`msa_mode`, `pair_mode` (default `unpaired_paired`), `rank_by` (default `multimer`), and `timeout_seconds` (no
limit by default). Outputs are `ranking`, `plddt` (unit scale), `ptm`, `iptm`, `pae`, and `pdb`/`cif`, listed in
ColabFold rank order.

The prerelease `colabfold_command` setting has been removed. `colabfold_python` selects the isolated interpreter
(default `/opt/colabfold/bin/python`). It, `device`, `data_dir`, `model_type`, `num_models`, `num_recycle`,
`use_templates` and `rank_by` are initialization-only settings because they determine the loaded runners.
The supported model families are `alphafold2_multimer_v1`, `v2` and `v3`. Seeds, MSA inputs, server options,
relaxation and output selection remain per-call options. A new sequence length or MSA shape may trigger JAX
compilation, but it does not reload model parameters.

Both adapters serialize requests inside each worker and keep request files separate. Closing the model
context releases its backend; a timeout or inference error discards the affected worker, and the next call
loads a fresh one. On Modal, reuse lasts for the lifetime of a warm GPU container: idle scale-down (currently
10 minutes), platform restarts or additional containers from autoscaling require another initial load.

This follows the upstream separation between [Protenix runner construction and inference](https://github.com/bytedance/Protenix/blob/main/runner/batch_inference.py)
and [ColabFold model loading](https://github.com/sokrypton/ColabFold/blob/efbf31c37cedb38cd09c69c1b991910a9866480e/colabfold/alphafold/models.py)
with [prediction on existing runners](https://github.com/sokrypton/ColabFold/blob/efbf31c37cedb38cd09c69c1b991910a9866480e/colabfold/batch.py).

### ESM-2
- A fresh `ESM2` instance starts on the backbone-only fast path and automatically switches to an internal masked-LM variant when `include_fields` requests `lm_logits` or `["*"]`; after that first upgrade, the instance keeps the MLM-capable model resident for later calls.
- Inline `<mask>` tokens are supported directly in sequence strings for both monomers and multimers, and returned logits stay aligned to the residue axis rather than raw tokenizer positions.
- `lm_logits` are full-vocabulary ESM logits over the tokenizer vocabulary.

Example usage:
```python
from boileroom import ESM2
from transformers import AutoTokenizer

model = ESM2(
    backend="apptainer",
    device="cuda:0",
    config={"model_name": "esm2_t6_8M_UR50D"},
)

result = model.embed(
    ["AC<mask>D"],
    options={"include_fields": ["lm_logits"]},
)

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
id_to_token = {token_id: token for token, token_id in tokenizer.get_vocab().items()}

result.embeddings.shape  # (1, 4, 320)
result.lm_logits.shape   # (1, 4, tokenizer.vocab_size)
# residue index 2 is the masked residue position in AC<mask>D

top_token_ids = result.lm_logits[0].argmax(axis=-1)
top_tokens = [id_to_token[token_id] for token_id in top_token_ids]
```


### ESM-C and ESM3
- `ESMC` and `ESM3` are embedding-only in this release. Boileroom does not expose ESM3 generation or folding APIs yet.
- Both wrappers use the MIT-licensed 2026 Chan Zuckerberg Biohub [`esm`](https://github.com/Biohub/esm) fork (the same package that backs ESMFold2) and support Modal and Apptainer through the shared `embed` contract. They run on the shared `esmfold2` Biohub runtime image rather than a separate one.
- Supported ESM-C model names are `esmc_300m` and `esmc_600m` (weights `biohub/esmc-300m-2024-12` / `biohub/esmc-600m-2024-12`). The cheap/default config is `esmc_300m`.
- Supported ESM3 model names are `esm3_sm_open_v1` plus aliases `esm3-open`, `esm3-sm-open-v1`, and `esm3-open-2024-03` (weights `biohub/esm3-sm-open-v1`).
- All ESM-C and ESM3 weights are MIT-licensed via the Biohub Hugging Face repositories.
- Outputs are residue-only arrays shaped `(batch, residues, features)`. BOS/EOS, chain-break, and other special-token rows are stripped. Batched outputs pad embeddings/logits with zeros and pad `chain_index` / `residue_index` with `-1`.
- Colon-separated input such as `"ACD:EF"` means multiple chains. Boileroom maps `:` to the SDK chain-break syntax internally and returns `chain_index=[0,0,0,1,1]`, `residue_index=[0,1,2,0,1]` for the residues.
- Set `MODEL_DIR` to control the model cache used by Modal/Apptainer runtimes.

Example usage:
```python
from boileroom import ESMC, ESM3

model = ESMC(config={"model_name": "esmc_300m"})
result = model.embed("ACD:EF")
result.embeddings.shape  # (1, 5, features)
result.chain_index       # [[0, 0, 0, 1, 1]]
result.residue_index     # [[0, 1, 2, 0, 1]]

esm3 = ESM3(config={"model_name": "esm3_sm_open_v1"})
esm3_result = esm3.embed(["ACD", "EF"])
```

Optional fields:
- `include_fields=["lm_logits"]` requests residue-aligned sequence logits when the SDK returns them.
- `include_fields=["hidden_states"]` is supported for ESM-C when provided by the SDK. ESM3 raises a clear `ValueError` for hidden-state requests.
- **ESM3-only track logits** — ESM3 is an all-to-all masked model, so it can predict its other tracks from sequence alone (structure input is optional, not required). Each is a per-residue logits array; decode to an estimate downstream via the SDK's per-track tokenizer. Requesting any of these on ESM-C raises a clear `ValueError`. The structure/folding track is not exposed.
    - `sasa_logits` — over the discretized SASA token vocabulary.
    - `secondary_structure_logits` — over the SS8 vocabulary.
    - `function_logits` — over the function-annotation vocabulary.
    - `residue_annotation_logits` — multi-hot residue-annotation logits.
- `include_fields=["*"]` means all supported optional fields for that model.

### ESMFold2
- ESMFold2 uses Biohub's ESMFold2 model family through the `esm` package (`esm>=3.4.1`, torch 2.11, CUDA 12.6 only), so it has its own runtime image instead of sharing the ESMFold/ESM-2 image.
- The default `biohub/ESMFold2` checkpoint is pinned to a Hugging Face revision (`revision=None` resolves to the pin). Pass `config={"revision": "<sha>"}` to load another snapshot; other `model_name` values load their latest snapshot unless `revision` is set.
- Checkpoint tensors use buffered Safetensors reads (`pread`) to avoid memory-mapped loading stalls on mounted model volumes. ESM still performs its strict checkpoint-key validation.
- The CCD dictionary always uses the canonical `biohub/ESMFold2` pin, including with alternate model checkpoints. It is cached under a revision-specific subdirectory of `ccd_cache_dir`; legacy unversioned `ccd.pkl` files are ignored.
- String inputs follow the existing BoilerRoom convention: `model.fold("AAA:BBB")` predicts one multichain complex, while `model.fold(["AAA", "BBB"])` predicts a batch of independent proteins.
- For all-atom complexes, pass lightweight input dataclasses from `boileroom.models.esmfold2.types` such as `ProteinInput`, `DNAInput`, `RNAInput`, `LigandInput`, and `StructurePredictionInput`.
- For explicit in-memory MSAs, use the shared `boileroom.inputs.MSAInput` abstraction; ESMFold2 also re-exports it from `boileroom.models.esmfold2` for compatibility. File-backed MSA paths are reserved for adapters such as Boltz-2 and are not consumed by ESMFold2 yet.

Example usage:
```python
from boileroom import ESMFold2
from boileroom.inputs import MSAInput
from boileroom.models.esmfold2.types import DNAInput, LigandInput, ProteinInput, StructurePredictionInput

model = ESMFold2(
    backend="modal",
    device="L4",
    config={"model_name": "biohub/ESMFold2-Fast"},
)

result = model.fold(
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    options={"include_fields": ["plddt", "ptm", "cif"], "num_sampling_steps": 50, "seed": 0},
)

result.atom_array[0]
result.plddt[0]
result.ptm[0]
result.cif[0]

complex_input = StructurePredictionInput(
    sequences=[
        ProteinInput(id="A", sequence="MIEIKDKQLTGLRFIDLFAGLGGFRLALESCGAECVYSNEWDKYAQEVYEMNFGEKPEG"),
        ProteinInput(id="M", sequence="ACD", msa=MSAInput(sequences=["ACD", "ACE"])),
        DNAInput(id="B", sequence="GATAGCGCTATC"),
        LigandInput(id="L", ccd=["SAH"]),
    ]
)
complex_result = model.fold(complex_input, options={"include_fields": ["cif", "iptm"]})
```
