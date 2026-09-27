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
`Protenix` wraps the official `protenix pred` CLI (`protenix==2.0.0`, default checkpoint `protenix-v2`). A single
`fold()` call accepts one sequence entry; use `:` to join multiple protein chains. On Modal it defaults to an
`A100-40GB` GPU.

Example usage:
```python
from boileroom import Protenix

model = Protenix(
    backend="modal",
    config={
        "model_name": "protenix-v2",
        "use_msa": True,
        "sample": 1,
        "cycle": 10,
        "step": 200,
    },
)

result = model.fold(
    "MLKNVHVLVLGAGDVGSVVVRLLEK:MLKNVHVLVLGAGDVGSVVVRLLEK",
    options={"include_fields": ["ptm", "iptm", "pae", "token_chain_ids", "cif"]},
)

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
  server can queue jobs for a long time, so Boileroom points the CLI at the configured server via
  `MMSEQS_SERVICE_HOST_URL`.
- **Provided MSA:** pass `config={"unpaired_msa": [a3m_chain_a, None, ...]}` with one A3M string (or `None` for
  query-only) per chain; no server call is made.
- **Single sequence:** set `use_msa=False`.

Sampling is controlled by `seeds` (comma-separated), `sample` (diffusion samples per seed, default 5), `cycle`
(recycles, default 10) and `step` (diffusion steps, default 200). Lower values trade accuracy for speed. Template
and RNA-MSA searches require the external tools and databases expected by Protenix. The runtime image installs
`hmmer` and `kalign`, but database paths still need to be available inside the container when those features are
enabled.

### AlphaFold2-Multimer
`AlphaFold2Multimer` drives ColabFold's `colabfold_batch` with `--model-type alphafold2_multimer_v3`. MSAs are
fetched from the public ColabFold MMseqs2 server (`https://api.colabfold.com`), so **no local genetic databases
(the ~2.6 TB AlphaFold data tree) are required**. AlphaFold model parameters are cached under `data_dir`
(default `${MODEL_DIR}/alphafold`); ColabFold downloads them there on first use. The wrapper accepts one
top-level sequence entry and uses `:` to split chains. On Modal it defaults to an `A100-80GB` GPU.

ColabFold runs in its own Python 3.10 environment inside the image, pinned to a fixed commit, because it needs
Python < 3.12 and `pandas<2`. By default ColabFold turns on CUDA unified memory (`TF_FORCE_UNIFIED_MEMORY=1`,
memory fraction 4.0). This stalls parameter loading on Modal, so Boileroom runs `colabfold_batch` with
`TF_FORCE_UNIFIED_MEMORY=0` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`. Values set in the caller's environment still
take precedence.

Example usage:
```python
from boileroom import AlphaFold2Multimer

model = AlphaFold2Multimer(
    backend="modal",
    config={
        "num_models": 5,
        "num_recycle": 3,
        "use_amber": False,
    },
)

result = model.fold(
    "MLKNVHVLVLGAGDVGSVVVRLLEK:MLKNVHVLVLGAGDVGSVVVRLLEK",
    options={"include_fields": ["ranking", "plddt", "iptm", "pae", "cif"]},
)

result.atom_array
result.ranking
result.plddt
result.iptm
```

MSA handling mirrors the other adapters:
- **Server (default):** alignments come from the ColabFold MMseqs2 server and are cached, content-addressed, under
  `${data_dir}/msa_cache` so repeat folds of the same complex skip the server.
- **Provided MSA:** pass `options={"msa": MSAInput(path="complex.a3m")}` (or `MSAInput(sequences=[...])`) to supply a
  ColabFold-compatible complex a3m directly; the server is not queried.
- **Single sequence:** set `config={"use_msa_server": False}` to run without an alignment.

Set `use_templates=True` to enable ColabFold templates and `use_amber=True` (optionally `use_gpu_relax=True`) for
Amber relaxation of the ranked predictions.

Other config keys: `num_models` (1–5, default 5), `num_recycle` (default 3), `num_seeds`, `random_seed`,
`msa_mode`, `pair_mode` (default `unpaired_paired`), `rank_by` (default `multimer`), and `timeout_seconds` (no
limit by default). Outputs are `ranking`, `plddt` (unit scale), `ptm`, `iptm`, `pae`, and `pdb`/`cif`, listed in
ColabFold rank order.

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
