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
- `include_fields=["sasa_logits"]` is supported for **ESM3 only**: per-residue logits over the discretized SASA token vocabulary, predicted from sequence (structure input is optional, not required). Decode to a SASA estimate via the SDK's SASA tokenizer bins. Requesting it on ESM-C raises a clear `ValueError`.
- `include_fields=["*"]` means all supported optional fields for that model.

### ESMFold2
- ESMFold2 uses Biohub's ESMFold2 model family through the `esm` package and Biohub's Transformers fork, so it has its own runtime image instead of sharing the ESMFold/ESM-2 image.
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
