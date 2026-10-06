"""Generate ESMFold2 reference predictions from the Biohub Platform for the integration tests.

The integration test ``tests/esmfold2/test_esmfold2_integration.py::test_esmfold2_matches_biohub_reference``
folds the sequence of a PDB entry with boileroom and compares the result against the prediction that
Biohub's own hosted ESMFold2 made for the same sequence and sampler settings. This script produces
those references: it reads each entry's sequence from the RCSB FASTA service, folds it through the
Biohub Platform API and writes, under ``tests/data/esmfold2``:

- ``<entry>.<model>.cif``: the predicted complex (mmCIF, B-factor column holds per-residue pLDDT)
- ``<entry>.<model>.npz``: ``plddt`` (per residue, unit scale), ``pae`` (residue x residue, Angstrom),
  ``ptm`` and ``iptm`` (``nan`` for a single chain)
- ``manifest.json``: the sequences, chain ids, sampler settings, pinned Platform model settings and model ids
  behind every file

The script needs the Biohub ``esm`` SDK, which is not a boileroom dependency, and an API key in
``ESM_API_KEY``::

    uv run --with "esm==3.4.1.post1" python scripts/testing/esmfold2_biohub_reference.py

Pass ``--entry 1UBQ --entry 1BRS:A,D`` to choose entries (``<pdb id>[:<chain>,<chain>...]``; without chains
every unique entity of the entry becomes one chain). The Platform exposes no sampler seed, so a refreshed
reference differs slightly from the previous one; the test tolerances allow for that.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import sys
import urllib.request

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "tests" / "data" / "esmfold2"
RCSB_FASTA_URL = "https://www.rcsb.org/fasta/entry/{entry}"
#: Biohub Platform model id -> boileroom ``config["model_name"]``.
MODELS: dict[str, str] = {
    "esmfold2-2026-05": "biohub/ESMFold2",
    "esmfold2-fast-2026-05": "biohub/ESMFold2-Fast",
}
#: The sampler settings the integration test uses; they match ``ESMFold2Core.DEFAULT_CONFIG``.
SAMPLER: dict[str, int] = {"num_loops": 3, "num_sampling_steps": 50}
#: Platform-side knobs pinned to what the local checkpoints apply, so the reference and the local fold run the same model
#: settings. Both ``biohub/ESMFold2`` and ``biohub/ESMFold2-Fast`` ship ``lm_mask_pct = 0.0`` in their config (the SDK
#: would otherwise default Fast to 0.1), and the per-loop language-model dropout of the local checkpoints is
#: ``lm_encoder.lm_dropout = 0.25`` (the SDK default is 0.3; esm>=3.4.1 exposes no local override).
PLATFORM_MODEL_SETTINGS: dict[str, float] = {"lm_mask_pct": 0.0, "lm_dropout": 0.25}
DEFAULT_ENTRIES = ("1UBQ", "1BRS:A,D")
_FASTA_HEADER = re.compile(r"^>(?P<entry>\w+)_(?P<entity>\d+)\|Chains? (?P<chains>[^|]+)\|")


def fetch_entry_chains(entry: str, wanted: list[str] | None) -> list[tuple[str, str]]:
    """Return ``(chain id, sequence)`` pairs of a PDB entry from the RCSB FASTA service.

    Parameters
    ----------
    entry : str
        Four-character PDB id.
    wanted : list[str] | None
        Author chain ids to keep, in order; ``None`` keeps one chain per unique entity.
    """
    with urllib.request.urlopen(RCSB_FASTA_URL.format(entry=entry), timeout=60) as response:  # noqa: S310
        text = response.read().decode("utf-8")
    by_chain: dict[str, str] = {}
    entity_first_chain: list[tuple[str, str]] = []
    for block in text.strip().split(">")[1:]:
        header, *lines = block.strip().splitlines()
        match = _FASTA_HEADER.match(">" + header)
        if match is None:
            raise ValueError(f"Unexpected RCSB FASTA header for {entry}: {header!r}")
        sequence = "".join(lines).strip()
        chains = [re.sub(r"\[auth (\w+)\]", r"\1", chain.strip()) for chain in match["chains"].split(",")]
        for chain in chains:
            by_chain[chain] = sequence
        entity_first_chain.append((chains[0], sequence))
    if wanted is None:
        return entity_first_chain
    missing = [chain for chain in wanted if chain not in by_chain]
    if missing:
        raise ValueError(f"{entry} has no chain(s) {missing}; available: {sorted(by_chain)}")
    return [(chain, by_chain[chain]) for chain in wanted]


def fold_on_biohub(model: str, chains: list[tuple[str, str]], token: str) -> dict[str, object]:
    """Fold one complex on the Biohub Platform and return the mmCIF text plus confidence arrays."""
    from esm.sdk import esmfold2_client
    from esm.sdk.api import ESMProteinError, FoldingConfig
    from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput

    client = esmfold2_client(model=model, token=token, request_timeout=900)
    prediction_input = StructurePredictionInput(
        sequences=[ProteinInput(id=chain_id, sequence=sequence) for chain_id, sequence in chains]
    )
    result = client.fold_all_atom(
        prediction_input, config=FoldingConfig(include_pae=True, **SAMPLER, **PLATFORM_MODEL_SETTINGS)
    )
    if isinstance(result, ESMProteinError):
        raise RuntimeError(f"Biohub Platform refused {model}: {result.error_code} {result.error_msg}")
    if isinstance(result, list):
        result = result[0]
    plddt = result.plddt.float().numpy()
    if plddt.size and np.nanmax(plddt) > 1.0:
        plddt = plddt / 100.0
    return {
        "cif": result.complex.to_mmcif(),
        "plddt": plddt.astype(np.float32),
        "pae": result.pae.float().numpy().astype(np.float32),
        "ptm": np.float32(result.ptm),
        "iptm": np.float32(np.nan if result.iptm is None else result.iptm),
    }


def parse_entry(spec: str) -> tuple[str, list[str] | None]:
    """Split ``1BRS:A,D`` into ``("1BRS", ["A", "D"])``; ``1UBQ`` gives ``("1UBQ", None)``."""
    entry, _, chains = spec.partition(":")
    return entry.upper(), [chain.strip() for chain in chains.split(",") if chain.strip()] or None


def main(argv: list[str] | None = None) -> int:
    """Write the reference fixtures and manifest."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entry", action="append", help="PDB id with optional chains, e.g. 1BRS:A,D")
    parser.add_argument("--model", action="append", choices=sorted(MODELS), help="Biohub model id(s) to fold with")
    parser.add_argument("--out", type=pathlib.Path, default=DATA_DIR, help=f"Output directory (default {DATA_DIR})")
    args = parser.parse_args(argv)

    token = os.environ.get("ESM_API_KEY", "")
    if not token:
        print("ESM_API_KEY is not set; create a key in the Biohub developer console.", file=sys.stderr)
        return 2
    entries = [parse_entry(spec) for spec in (args.entry or DEFAULT_ENTRIES)]
    models = args.model or sorted(MODELS)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out / "manifest.json"
    manifest: dict[str, object] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["sampler"] = SAMPLER
    manifest["platform_model_settings"] = PLATFORM_MODEL_SETTINGS
    manifest["generated_by"] = "scripts/testing/esmfold2_biohub_reference.py"
    references: dict[str, object] = dict(manifest.get("references", {}))
    for entry, wanted in entries:
        chains = fetch_entry_chains(entry, wanted)
        for model in models:
            prediction = fold_on_biohub(model, chains, token)
            stem = f"{entry}.{model}"
            (args.out / f"{stem}.cif").write_text(str(prediction["cif"]))
            np.savez(
                args.out / f"{stem}.npz",
                plddt=prediction["plddt"],
                pae=prediction["pae"],
                ptm=prediction["ptm"],
                iptm=prediction["iptm"],
            )
            references[stem] = {
                "pdb_entry": entry,
                "chains": [{"id": chain_id, "sequence": sequence} for chain_id, sequence in chains],
                "biohub_model": model,
                "model_name": MODELS[model],
                "cif": f"{stem}.cif",
                "confidence": f"{stem}.npz",
                "generated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
            }
            plddt = np.asarray(prediction["plddt"])
            print(f"{stem}: {plddt.size} residues, mean pLDDT {plddt.mean():.3f}, pTM {float(prediction['ptm']):.3f}")
    manifest["references"] = dict(sorted(references.items()))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
