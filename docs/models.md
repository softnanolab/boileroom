# Model Examples

## Configuration keys

Every model checks its `config` (at construction) and its per-call `options` against the keys it knows:

- A key must be in the model core's `DEFAULT_CONFIG` or be one of the shared keys `msa`, `templates` and
  `optimization`. Any other key (a typo such as `num_sample`, or an option the model does not have) raises
  `ValueError` naming the key and listing the allowed ones, at construction, per call and in `update_config`.
- The core's `STATIC_CONFIG_KEYS` (such as `device` or `model_name`) and `optimization` can only be set at
  construction; passing them per call raises `ValueError`.
- `msa` and `templates` given in `config=` apply to every call. A model that cannot use them refuses them, whether they
  come from `config=` or a call, rather than ignoring them.
- `optimization` is `"vanilla"` for every model without an optimization kit; `"exact"` and `"fast"` raise
  `OptimizationUnavailableError` there (see [optimization.md](optimization.md)).

On Modal these errors are raised from the first call, not while the container starts.

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
- **Provided MSA:** pass `options={"msa": [a3m_chain_a, None, ...]}` with one A3M string per chain, or `None` for a
  query-only chain (this also suppresses the automatic search for that chain). The list length must equal the number of
  chains and each A3M's first row must be the chain's sequence. When `msa` is set, no server call is made. A supplied
  MSA needs `use_msa=True`; with `use_msa=False` it is refused with `ValueError` rather than ignored.
- Each entry becomes that chain's *unpaired* MSA; there is no paired input. For a heteromer the cross-chain pairing a
  server search provides is therefore lost when `msa` is given. An all-`None` list means query-only for every chain.
- The `unpaired_msa` option has been removed; pass the same list as `msa`.
- Rows may hold only ASCII letters, `-` and `.`: whitespace inside a row, `*`, digits, `#` comment lines and non-ASCII
  text are refused, because upstream's reader would drop them or count them as insertions. `.` (insert-state gap) is
  removed and lowercase insertions are kept. A chain shorter than 5 residues accepts only its query row, since upstream
  ignores the MSA of such a chain.
- **Single sequence:** set `use_msa=False`.

Templates:
- Pass `options={"templates": {"name": mmcif_text, ...}}` with up to 4 mmCIF structures (name -> mmCIF text). The templates
  apply to one chain only, selected by `options["templates_chain"]` (a valid index into the `:`-separated chains, default
  `0`); the other protein chains get no templates.
- Caller templates need a checkpoint with a template embedder (`protenix-v2`, `protenix_base_default_v1.0.0`,
  `protenix_base_20250630_v1.0.0`; for OpenDDE `opendde_v1`). They replace the template search, so they cannot be combined
  with `use_template=True`; `use_template=True` on a checkpoint without a template embedder is refused too.
- Each mmCIF must be a full PDB-style file with `_atom_site`, `_entity_poly_seq` and `_struct_asym` loops, exactly one
  polymer chain and at least one peptide residue type in `_chem_comp` (a missing `_chem_comp` loop is filled in as
  L-peptide). It must share at least 10 aligned residues with the target chain and cover more than 10% of it, and no two
  templates may stage the same sequence. Upstream would drop such templates silently; boileroom refuses them with
  `ValueError`, and the worker fails the request if fewer templates were featurized than were staged.
- Boileroom stages each structure under a synthetic id, aligns it to the query chain and stamps a release date that
  passes Protenix's 2021-09-30 cutoff; nothing is searched or downloaded. A template of the target itself, which upstream
  would discard as a duplicate of the query, is kept by replacing its first aligned residue in the hit row with a letter
  that does not occur in the query.

Sampling is controlled by `seeds` (comma-separated), `sample` (diffusion samples per seed, default 5), `cycle`
(recycles, default 10) and `step` (diffusion steps, default 200). Lower values trade accuracy for speed. Template
and RNA-MSA searches require the external tools and databases expected by Protenix. The runtime image installs
`hmmer` and `kalign`, but database paths still need to be available inside the container when those features are
enabled.

Optimization: `config={"optimization": "vanilla" | "exact" | "fast"}` (default `"vanilla"`, initialization-only) switches
to the Anthropic kit kernels on A100 or H100/H200 GPUs only; other GPUs are refused by name. See
[optimization.md](optimization.md) for the requirements, kit image and measured speedups.

Protenix keeps its runner in a persistent worker, retaining a hard `timeout_seconds` limit (3500 seconds by default, `None` disables it). A timeout or inference failure discards the worker; the next call reloads cleanly. A refused optimization mode (`OptimizationUnavailableError`) is the exception: the refusal stands for that model instance and is raised again on every later call, without restarting the worker. Use one model context for repeated predictions. Seeds, cycle/step/sample counts, dtype, MSA inputs and output selection remain per-call options; each request gets fresh output paths and inference settings.

The prerelease CLI setting `protenix_command` has been removed. `model_name`, `device`, `msa_server_url`, `use_template`, `trimul_kernel`, `triatt_kernel`, `enable_cache`, `enable_fusion`, `enable_tf32`, and `optimization` are now initialization-only settings. Create a new instance to change them.

### OpenDDE
`OpenDDE` wraps the AF3-style [OpenDDE](https://github.com/aurekaresearch/OpenDDE) runner (`opendde==1.1.1`, single released model `opendde_v1`). Its runner API, JSON input and output files match Protenix 2.0, so it shares Protenix's interface end to end: one sequence entry per call with `:` joining chains, the ColabFold MMseqs2 server via `msa_server_url`, the persistent worker with `timeout_seconds`, and the same `OpenDDEOutput` fields (`atom_array`, `confidence`, `plddt`, `ptm`, `iptm`, `pae`, `token_chain_ids`, `token_res_ids`, `atom_plddt`, `seeds`, `sample_ranks`, `pdb`, `cif`). Differences: `dtype` is `bf16` (default) or `fp32`, `trimul_kernel`/`triatt_kernel` default to `cuequivariance` (an explicit request runs cuEquivariance or fails; the kernel that ran is recorded as `predict.kernel.resolved.*` in `metadata.runtime`), and the runner lives in its own Python 3.11.5 interpreter (python-build-standalone, under `/opt/opendde`; `opendde_python`, initialization-only). Weights download to `$MODEL_DIR/opendde` on first use. On Modal it defaults to an `A100-40GB` GPU.

- **Provided MSA:** `options={"msa": [a3m_chain_a, None, ...]}`, one A3M string or `None` (query-only) per chain, with the same rules as Protenix: unpaired only (heteromer pairing is lost), `use_msa=True` required, and `unpaired_msa` removed.
- **Templates:** `options={"templates": {"name": mmcif_text}}` (up to 4) applied to chain `options["templates_chain"]` (default `0`), with the same rules as Protenix; `opendde_v1` has a template embedder.
- **Optimization:** `config={"optimization": "vanilla" | "exact" | "fast"}` (default `"vanilla"`, initialization-only). `exact` and `fast` use the Anthropic OpenDDE kit on A100 or H100/H200 only (other GPUs are refused by name); see [optimization.md](optimization.md). Unlike Protenix and ESMFold2, OpenDDE uses the stock `boileroom-opendde` image for every mode, because its image already carries the kit stack.

```python
from boileroom import OpenDDE

with OpenDDE(backend="modal", config={"optimization": "fast"}) as model:
    result = model.fold("SEQ_A:SEQ_B", options={"include_fields": ["pae", "token_chain_ids", "ptm", "iptm"]})
```

LayerNorm depends on the mode (`OPENDDE_LAYERNORM` in `boileroom/models/opendde/core.py`): `vanilla` runs torch LayerNorm (`LAYERNORM_TYPE=torch`, upstream's default), and `exact` / `fast` run upstream's fused LayerNorm (`fast_layernorm`), which the worker JIT-builds at first use with the venv's `ninja` and CUDA 12.6 `nvcc`, caching it under `$MODEL_DIR/opendde/jit/<stack key>/torch_ext` (keyed by torch, CUDA and compute capability; see [optimization.md](optimization.md#jit-compile-caches)). The image sets no `LAYERNORM_TYPE`; the worker sets it per mode and refuses any other value. A kit mode whose fused LayerNorm does not load is refused. See [optimization.md](optimization.md#layernorm-per-mode).

On compute capability 7.x cards (V100, T4) upstream runs both triangle sites on its torch kernels in fp32, whatever `trimul_kernel`, `triatt_kernel` and `dtype` ask for. `vanilla` serves that stock behaviour and records it as `worker.kernel.cc7_fallback=true` in `metadata.runtime` (with `predict.kernel.resolved.*` showing `torch` and `fp32`); on every other card a requested kernel that resolves to something else is still refused. The kit modes run on A100 or H100/H200 only.

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
- **Provided MSA:** the server is not queried. `options["msa"]` takes one of three forms:
  - `MSAInput(sequences=[...])`: one row per alignment entry, `:`-joined with one segment per chain (counting repeated
    chains). Rows that cover several chains become ColabFold's paired MSA; a heteromer also gets each chain's own segments
    as unpaired rows. Every copy of a repeated chain must carry the same segment in each row.
  - `MSAInput(path="...")`: an a3m file, read on the caller's side. A single chain takes plain a3m; a complex needs
    ColabFold's complex a3m, whose first line is the `#<lengths>\t<cardinalities>` header over the unique chains. A
    headerless multi-chain file is refused, because ColabFold would fold it as one chain.
  - A list with one entry per chain: A3M text for that chain's unpaired MSA, or `None` for query only. This form carries
    no cross-chain pairing, and at least one entry must be text.
- Every form must start with the requested chain(s) exactly as the first row (no insertions) and give every row one
  aligned column per residue. Rows may hold only letters and `-`; `.` is refused, because ColabFold counts it as an
  aligned column. `MSAInput(..., remove_insertions=True)` drops lowercase insertions and `.` before these checks; in the
  list form delete `.` yourself. A3M text is parsed with the shared `boileroom.inputs.parse_a3m`.
- **Single sequence:** set `config={"use_msa_server": False}` to run without an alignment.

`options["templates"]` (caller-supplied mmCIF structures, accepted by Protenix and OpenDDE) is refused here with a `ValueError`.
Set `use_templates=True` to enable ColabFold's own server-side templates and `use_amber=True` (optionally `use_gpu_relax=True`) for
Amber relaxation of the ranked predictions. The `optimization` kit modes do not apply to AlphaFold2-Multimer.

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
- Both wrappers use the 2026 Chan Zuckerberg Biohub [`esm`](https://github.com/Biohub/esm) fork, whose code is MIT-licensed (the same package that backs ESMFold2), and support Modal and Apptainer through the shared `embed` contract. They run on the shared `esmfold2` Biohub runtime image rather than a separate one.
- Supported ESM-C model names are `esmc_300m` and `esmc_600m` (weights `biohub/esmc-300m-2024-12` / `biohub/esmc-600m-2024-12`). The cheap/default config is `esmc_300m`.
- Supported ESM3 model names are `esm3_sm_open_v1` plus aliases `esm3-open`, `esm3-sm-open-v1`, and `esm3-open-2024-03` (weights `biohub/esm3-sm-open-v1`).
- Weights come from the Biohub Hugging Face repositories; their terms are set there (the ESM-C model cards list
  `license: mit, other` together with Biohub's acceptable use policy). Check the model card of the repository you load.
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

**ESM3 inverse folding.** `ESM3.inverse_fold(sequence, backbone_coordinates, positions)` masks the residues at `positions` (indices over all residues, chain breaks excluded), conditions ESM3 on the given `(n_residues, 3, 3)` N/CA/C backbone coordinates, and returns an `ESM3InverseFoldingOutput` with `logits` of shape `(n_positions, 20)` over `amino_acids` (`"ACDEFGHIKLMNPQRSTVWY"`). All positions are scored in one forward pass, so pass a single position for the distribution of one residue given all others.

### ESMFold2
- ESMFold2 uses Biohub's ESMFold2 model family through the `esm` package (`esm>=3.4.1`, torch 2.11, CUDA 12.6 only), so it has its own runtime image instead of sharing the ESMFold/ESM-2 image.
- The default `biohub/ESMFold2` checkpoint is pinned to a Hugging Face revision (`revision=None` resolves to the pin). Pass `config={"revision": "<sha>"}` to load another snapshot; other `model_name` values load their latest snapshot unless `revision` is set.
- Checkpoint tensors use buffered Safetensors reads (`pread`) to avoid memory-mapped loading stalls on mounted model volumes. ESM still performs its strict checkpoint-key validation.
- The CCD dictionary always uses the canonical `biohub/ESMFold2` pin, including with alternate model checkpoints. It is cached under a revision-specific subdirectory of `ccd_cache_dir`; legacy unversioned `ccd.pkl` files are ignored.
- String inputs follow the existing BoilerRoom convention: `model.fold("AAA:BBB")` predicts one multichain complex, while `model.fold(["AAA", "BBB"])` predicts a batch of independent proteins.
- For all-atom complexes, pass lightweight input dataclasses from `boileroom.models.esmfold2.types` such as `ProteinInput`, `DNAInput`, `RNAInput`, `LigandInput`, and `StructurePredictionInput`.
- For explicit in-memory MSAs, use the shared `boileroom.inputs.MSAInput` abstraction; ESMFold2 also re-exports it from `boileroom.models.esmfold2` for compatibility. File-backed MSA paths are reserved for adapters such as Boltz-2 and are not consumed by ESMFold2 yet.
- `options={"msa": [a3m_text_or_None, ...]}` is a shortcut that attaches an A3M to protein entries: one A3M string or `None` per entry of the input's `sequences` (for `"AAA:BBB"` that is one entry per chain). It applies to exactly one input structure (not a batch), the first A3M row must equal the entry's sequence, every row must have the same aligned length, and an entry that already carries an `MSAInput` or is not a protein must be `None`.
- ESMFold2-Fast (`biohub/ESMFold2-Fast`) has no MSA encoder, so it refuses any MSA, inline on an entry or from `options["msa"]`, in every mode with a `ValueError` instead of silently folding single-sequence. Use `biohub/ESMFold2` for MSA folds.
- Covalent bonds are checked before folding: a bond naming a missing chain, a residue index out of range (0-based) or a negative atom index raises `ValueError`, as does combining bonds with `:` / `|` chainbreaks (give each chain its own entry). An atom index past the heavy atoms of an unmodified protein, DNA or RNA residue, or past the atoms of a CCD ligand minus its leaving atoms, also raises `ValueError` in every mode. The kit's older esm would otherwise drop that bond silently. Indices on modified residues, unknown nucleotides and SMILES ligands are left to esm.
- `options["templates"]` is not supported: ESMFold2 refuses it with a `ValueError` (only Protenix and OpenDDE take caller-supplied mmCIF templates).
- `config={"optimization": "vanilla" | "exact" | "fast"}` (default `"vanilla"`, initialization-only) runs the Anthropic kit kernels on A100 or H100/H200 only; L4, L40S and CPU are refused. Kit modes run on a separate kit image and Modal class (the default Modal GPU is `A100-80GB`); see [optimization.md](optimization.md).
- Kit modes: the kit loads its own pinned snapshots (about 27 GB) into `$MODEL_DIR/esmfold2/kit-hf` on the first call. They serve only `model_name="biohub/ESMFold2"` or `"biohub/ESMFold2-Fast"`, and setting `revision`, `cache_dir` or `ccd_cache_dir`, or any other `model_name`, raises `ValueError` at construction (the kit reads its own `ccd.pkl` from that directory).
- The kit arms one kernel variant per process: ESMFold2-Fast uses `fast`, and the full model uses `full_nomsa` unless `config={"kit_msa": True}` (initialization-only, a bool) selects `full_msa`. Both full variants fold with a user MSA; `kit_msa` selects the kernels, not whether an MSA is used.
- Kit modes fold with the kit's own sampler schedule (esm 3.3.0), so `noise_scale`, `step_scale` and `max_inference_sigma` are refused with `ValueError`, and `msa_max_depth` must be an integer (`None` would mean the checkpoint's depth under vanilla but every row under the kit). The kit prepares, runs and decodes in one call, so `preprocessing_time` and `postprocessing_time` are `0.0` and `inference_time` covers the whole fold.
- Reference check: `tests/esmfold2/test_esmfold2_integration.py` folds the sequences of PDB entries 1UBQ (ubiquitin) and 1BRS (barnase–barstar, chains A and D) with both checkpoints and compares structure, pLDDT, pTM, ipTM and the full PAE matrix (including its inter-chain blocks) against predictions made by the Biohub Platform's hosted ESMFold2 for the same sequences and sampler settings. The references live under `tests/data/esmfold2/` with a `manifest.json`; regenerate them with `ESM_API_KEY=... uv run --with "esm==3.4.1.post1" python scripts/testing/esmfold2_biohub_reference.py`. The Platform's `lm_mask_pct` and `lm_dropout` are pinned to the checkpoints' own values (0.0 and 0.25) so both sides run the same model settings. The Platform exposes no seed, so the vanilla comparison allows sampler noise (structure is compared tightly only over residues both predictions call confident); the `exact` and `fast` kit modes fold 1UBQ with both checkpoints and must land within 1.1 times vanilla's own seed-to-seed noise of the Biohub reference on four metrics (all-residue and confident-residue CA RMSD, mean and max PAE entry gap; the noise was measured on an A100-80GB over seeds 0-3 and is recorded in the test). The kit comparisons run only when `BOILEROOM_KIT_IMAGE_SOURCE` is set (see [optimization.md](optimization.md)).

Example usage:
```python
from boileroom import ESMFold2
from boileroom.inputs import MSAInput
from boileroom.models.esmfold2.types import DNAInput, LigandInput, ProteinInput, StructurePredictionInput

# Vanilla runs on any GPU. For the kit kernels use config={"optimization": "fast"} with an A100 or H100/H200; L4 is refused.
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

# MSAs need the full checkpoint: ESMFold2-Fast refuses them.
full_model = ESMFold2(backend="modal", device="L4", config={"model_name": "biohub/ESMFold2"})
complex_input = StructurePredictionInput(
    sequences=[
        ProteinInput(id="A", sequence="MIEIKDKQLTGLRFIDLFAGLGGFRLALESCGAECVYSNEWDKYAQEVYEMNFGEKPEG"),
        ProteinInput(id="M", sequence="ACD", msa=MSAInput(sequences=["ACD", "ACE"])),
        DNAInput(id="B", sequence="GATAGCGCTATC"),
        LigandInput(id="L", ccd=["SAH"]),
    ]
)
complex_result = full_model.fold(complex_input, options={"include_fields": ["cif", "iptm"]})
```
