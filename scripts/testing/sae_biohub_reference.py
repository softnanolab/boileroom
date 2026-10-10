"""Generate reference SAE features from Biohub's own ESM-C + SAE pipeline for the SAE tests.

The SAE tests check boileroom's local feature path against the features that Biohub's reference
implementation computes for the same sequences, layer and checkpoint. This script produces those references by
running the native ``esm`` pipeline on the CPU in float32: ``EsmcModel`` with the per-layer ``EsmcSaeLayer`` attached
through ``add_sae_models``, so the SAE reads exactly the hidden state Biohub trained it on. For every ESM-C model in
:data:`MODELS` it writes, under ``tests/data/sae``:

- ``<model>.layer<N>.npz``: for every sequence, ``<name>.indices`` / ``<name>.values`` hold each residue's ``k``
  active features (top ``k`` of the dense activations; BOS/EOS rows dropped), and for the sequences in
  :data:`STATE_SEQUENCES` ``<name>.states`` holds the layer-``N`` hidden states the SAE consumed
- ``manifest.json``: the sequences, pinned Hub revisions, SAE architecture and package versions behind every file

The script needs the Biohub ``esm`` SDK, which requires a newer ``torch`` than boileroom pins, so run it outside the
project environment::

    uv run --isolated --no-project --with "esm==3.4.1.post1" python scripts/testing/sae_biohub_reference.py

It downloads the ESM-C weights and one SAE layer per model (about 3 GB in total) to the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "tests" / "data" / "sae"
#: boileroom ``esmc_model_name`` -> native ESM-C repo, SAE repo and SAE layer (boileroom's local defaults).
MODELS: dict[str, dict[str, str | int]] = {
    "esmc_600m": {"esmc_repo": "biohub/ESMC-600M", "sae_repo": "biohub/ESMC-600M-sae-k64-codebook16384", "layer": 27},
    "esmc_300m": {"esmc_repo": "biohub/ESMC-300M", "sae_repo": "biohub/ESMC-300M-sae-k64-codebook16384", "layer": 22},
}
SEQUENCES: dict[str, str] = {
    "ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "short": "MLKNVHVLVLGAGDVGSVVVRLLEK",
}
#: Sequences whose layer states are stored too, so the SAE can be checked without running ESM-C.
STATE_SEQUENCES = ("ubiquitin",)


def reference_for_model(model_name: str, spec: dict[str, str | int]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Run the native ESM-C + SAE pipeline for every sequence and return the npz arrays and manifest entry."""
    import esm
    import torch
    from esm.models.esmc import EsmcTokenizer
    from esm.models.esmc.model import EsmcModel
    from esm.models.esmc.sae import EsmcSaeModel
    from huggingface_hub import HfApi

    layer = int(spec["layer"])
    esmc_repo, sae_repo = str(spec["esmc_repo"]), str(spec["sae_repo"])
    api = HfApi()
    esmc_revision = api.model_info(esmc_repo).sha
    sae_revision = api.model_info(sae_repo).sha

    model = EsmcModel.from_pretrained(esmc_repo, revision=esmc_revision, device="cpu", dtype=torch.float32).eval()
    sae = EsmcSaeModel.from_pretrained(
        sae_repo, revision=sae_revision, allow_patterns=["config.json", f"layer_{layer}.safetensors"]
    )
    sae.initialize_layers([layer])
    sae_layer = sae.layers[str(layer)]
    model.add_sae_models([sae_layer])
    tokenizer = EsmcTokenizer()
    k = int(sae.config.k)

    arrays: dict[str, np.ndarray] = {}
    min_topk_gap = float("inf")
    for name, sequence in SEQUENCES.items():
        inputs = tokenizer(sequence, return_tensors="pt")
        with torch.inference_mode():
            output = model(**inputs, output_hidden_states=True)
            dense = output.sae_outputs[f"layer{layer}"].to_dense().reshape(-1, sae.config.codebook_dim)[1:-1]
            states = output.hidden_states[layer].reshape(-1, sae.config.d_model)[1:-1]
            # The SAE is per-residue, so re-encoding the residue rows alone must reproduce the pipeline's features.
            rerun = sae_layer(states).feature_magnitudes
            pre = torch.relu((sae_layer._zscore_normalize_representation(states) - sae_layer.b_dec) @ sae_layer.W_enc)
        if dense.shape[0] != len(sequence):
            raise RuntimeError(f"{model_name}/{name}: {dense.shape[0]} residue rows for {len(sequence)} residues.")
        if not torch.equal(rerun, dense):
            raise RuntimeError(f"{model_name}/{name}: re-encoding the layer-{layer} states does not match sae_outputs.")
        top = torch.topk(dense, k=k, dim=-1)
        if int((dense != 0).sum(-1).max()) > k:
            raise RuntimeError(f"{model_name}/{name}: more than k={k} active features in a residue.")
        ranked = torch.topk(pre, k=k + 1, dim=-1).values
        min_topk_gap = min(min_topk_gap, float((ranked[:, k - 1] - ranked[:, k]).min()))
        arrays[f"{name}.indices"] = top.indices.numpy().astype(np.int32)
        arrays[f"{name}.values"] = top.values.numpy().astype(np.float32)
        if name in STATE_SEQUENCES:
            arrays[f"{name}.states"] = states.numpy().astype(np.float32)
        print(f"{model_name}/{name}: {dense.shape[0]} residues, {int((dense != 0).any(0).sum())} distinct features")

    entry: dict[str, object] = {
        "esmc_model_name": model_name,
        "esmc_repo": esmc_repo,
        "esmc_revision": esmc_revision,
        "num_blocks": int(model.config.num_hidden_layers),
        "sae_repo": sae_repo,
        "sae_revision": sae_revision,
        "layer": layer,
        "d_model": int(sae.config.d_model),
        "num_features": int(sae.config.codebook_dim),
        "k": k,
        "features": f"{model_name}.layer{layer}.npz",
        "state_sequences": list(STATE_SEQUENCES),
        # Smallest margin between the k-th and (k+1)-th ReLU-ed pre-activation over all residues: how much numerical
        # drift the top-k selection tolerates before a feature swaps in or out.
        "min_topk_gap": min_topk_gap,
        "esm_version": esm.__version__,
        "torch_version": torch.__version__,
        "dtype": "float32",
        "device": "cpu",
        "generated_at": dt.date.today().isoformat(),
    }
    return arrays, entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", action="append", choices=sorted(MODELS), help="ESM-C model(s) to generate (default: all)."
    )
    args = parser.parse_args(argv)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "manifest.json"
    manifest: dict[str, object] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {"references": {}}
    )
    manifest["generated_by"] = "scripts/testing/sae_biohub_reference.py"
    manifest["sequences"] = SEQUENCES
    references = manifest.setdefault("references", {})
    assert isinstance(references, dict)
    for model_name in args.model or sorted(MODELS):
        arrays, entry = reference_for_model(model_name, MODELS[model_name])
        np.savez_compressed(DATA_DIR / str(entry["features"]), **arrays)
        references[model_name] = entry
        print(f"wrote {DATA_DIR / str(entry['features'])} (min top-k gap {entry['min_topk_gap']:.3g})")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
