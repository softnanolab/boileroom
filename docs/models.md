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
