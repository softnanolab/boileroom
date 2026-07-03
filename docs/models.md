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
`Protenix` wraps the official `protenix pred` CLI. A single `fold()` call accepts one sequence entry; use `:` to
join multiple protein chains.

Example usage:
```python
from boileroom import Protenix

model = Protenix(
    backend="apptainer",
    device="cuda:0",
    config={
        "model_name": "protenix_base_default_v1.0.0",
        "use_msa": True,
        "use_template": False,
        "sample": 1,
        "step": 200,
    },
)

result = model.fold(
    "MLKNVHVLVLGAGDVGSVVVRLLEK:MLKNVHVLVLGAGDVGSVVVRLLEK",
    options={"include_fields": ["confidence", "cif"]},
)

result.atom_array
result.confidence
result.cif
```

Set `use_msa=False` for single-sequence inference. Template and RNA-MSA searches require the external tools and
databases expected by Protenix; the runtime image installs `hmmer` and `kalign`, but database paths still need to
be available inside the container when those features are enabled.

### AlphaFold2-Multimer
`AlphaFold2Multimer` wraps the official DeepMind AlphaFold runner with `--model_preset=multimer`. It requires the
AlphaFold databases and model parameters in `data_dir`; by default this is `${MODEL_DIR}/alphafold` inside the
runtime. The wrapper accepts one top-level sequence entry and uses `:` to split chains into a multi-record FASTA.

Example usage:
```python
from boileroom import AlphaFold2Multimer

model = AlphaFold2Multimer(
    backend="apptainer",
    device="cuda:0",
    config={
        "max_template_date": "2022-01-01",
        "db_preset": "full_dbs",
        "models_to_relax": "none",
    },
)

result = model.fold(
    "MLKNVHVLVLGAGDVGSVVVRLLEK:MLKNVHVLVLGAGDVGSVVVRLLEK",
    options={"include_fields": ["ranking", "plddt", "iptm", "pdb"]},
)

result.atom_array
result.ranking
result.plddt
result.iptm
```

AlphaFold2-Multimer does not download the genetic databases automatically. Use `data_dir` for a bind-mounted
database tree containing the official AlphaFold data layout, including UniProt and PDB seqres for multimer runs.
With Apptainer, the default `data_dir` is `${MODEL_DIR}/alphafold` inside the container, so place the databases
under `$MODEL_DIR/alphafold` on the host or pass another container-visible path.

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
