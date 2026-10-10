"""Generate stock ``rf3 fold`` reference predictions for the RF3 integration tests.

The integration test ``tests/rf3/test_rf3_integration.py::test_rf3_matches_stock_rf3_reference`` folds a few sequences with
boileroom and compares the result with what RoseTTAFold 3's own command line wrote for the same sequences and settings.
This script produces those references: it runs ``rf3 fold`` from the RF3 runtime image on Modal, with the checkpoint
boileroom keeps in the model volume, and writes under ``tests/data/rf3``:

- ``<entry>_model.cif``, ``<entry>_summary_confidences.json``, ``<entry>_confidences.json``: the files ``rf3 fold`` wrote
- ``manifest.json``: the sequences, settings, checkpoint and GPU behind every file

The checkpoint must already be in the volume: fold anything once with ``RF3`` first. Run::

    uv run python scripts/testing/rf3_stock_reference.py

A refreshed reference may differ from the committed one in the last digits (bf16 kernels and RNG order); the test
tolerances allow for that.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import modal

from boileroom.images.modal import get_modal_image
from boileroom.images.volumes import model_weights
from boileroom.models.rf3.runtime import CHECKPOINT_BYTES, CHECKPOINT_NAME, CHECKPOINT_SHA256
from boileroom.utils import HOURS, MODAL_MODEL_DIR

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "tests" / "data" / "rf3"
FOUNDRY_COMMIT = "4010e3e2e7350edada3e25a45c908c6bf407df4d"
#: Each entry's output must contain all of these; a fold that writes a different layout fails instead of a partial reference.
#: Checked in this order because ``_summary_confidences.json`` also ends with ``_confidences.json``.
ARTIFACT_SUFFIXES = ("_model.cif", "_summary_confidences.json", "_confidences.json")
#: Below the one-hour Modal deadline, so one stalled fold names itself instead of the whole job timing out.
FOLD_TIMEOUT_SECONDS = 20 * 60
#: The stock settings the integration test mirrors; ``seed`` and the rest match the test's fold options.
SETTINGS: dict[str, int | float] = {
    "n_recycles": 10,
    "num_steps": 50,
    "diffusion_batch_size": 1,
    "seed": 0,
    "early_stopping_plddt_threshold": 0.0,
}
#: entry -> (PDB entry, chain sequences). Canonical residues only: boileroom's adapter is protein-only.
ENTRIES: dict[str, tuple[str, list[str]]] = {
    "1ubq": ("1UBQ", ["MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"]),
    "2zta": ("2ZTA", ["RMKQLEDKVEELLSKNYHLENEVARLKKLVGER"] * 2),
}
RF3_BIN = "/opt/rf3/bin/rf3"
GPU = "A100-40GB"

app = modal.App("boileroom-rf3-stock-reference")


def _artifact_suffix(name: str) -> str:
    """Return the entry of ``ARTIFACT_SUFFIXES`` that names this file, preferring the most specific one."""
    return next(suffix for suffix in ARTIFACT_SUFFIXES if name.endswith(suffix))


@app.function(
    image=get_modal_image("rf3"),
    gpu=GPU,
    timeout=1 * HOURS,
    volumes={MODAL_MODEL_DIR: model_weights},
)
def run_stock(settings: dict[str, int | float], entries: dict[str, list[str]]) -> dict[str, dict[str, bytes]]:
    """Run ``rf3 fold`` for every entry and return ``{entry: {file name: bytes}}`` of the files it wrote."""
    import os
    import subprocess
    import tempfile

    checkpoint = pathlib.Path(MODAL_MODEL_DIR) / "rf3" / CHECKPOINT_NAME
    if not checkpoint.is_file() or checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise SystemExit(f"{checkpoint} is missing: fold once with boileroom's RF3 so it downloads the checkpoint")
    overrides = [f"{key}={value}" for key, value in settings.items()]
    written: dict[str, dict[str, bytes]] = {}
    with tempfile.TemporaryDirectory() as work:
        for entry, sequences in entries.items():
            spec = [
                {
                    "name": entry,
                    "components": [{"seq": seq, "chain_id": chr(65 + i)} for i, seq in enumerate(sequences)],
                }
            ]
            input_json = os.path.join(work, f"{entry}.json")
            with open(input_json, "w", encoding="utf-8") as handle:
                json.dump(spec, handle)
            out_dir = os.path.join(work, f"out_{entry}")
            try:
                subprocess.run(
                    [
                        RF3_BIN,
                        "fold",
                        f"inputs={input_json}",
                        f"ckpt_path={checkpoint}",
                        f"out_dir={out_dir}",
                        *overrides,
                    ],
                    check=True,
                    timeout=FOLD_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired as error:
                raise SystemExit(f"rf3 fold for {entry} did not finish within {FOLD_TIMEOUT_SECONDS} s") from error
            sample_dir = pathlib.Path(out_dir) / entry
            files = {
                path.name: path.read_bytes()
                for path in sample_dir.glob(f"{entry}_*")
                if path.name.endswith(ARTIFACT_SUFFIXES)
            }
            kinds = {_artifact_suffix(name) for name in files}
            if missing := [suffix for suffix in ARTIFACT_SUFFIXES if suffix not in kinds]:
                raise SystemExit(f"rf3 fold wrote no file ending {missing} for {entry} under {sample_dir}")
            written[entry] = files
    return written


def main() -> None:
    """Run the stock command on Modal and write the references and their manifest."""
    with app.run():
        written = run_stock.remote(SETTINGS, {entry: sequences for entry, (_, sequences) in ENTRIES.items()})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for files in written.values():
        for name, data in files.items():
            (DATA_DIR / name).write_bytes(data)
    manifest = {
        "generated_by": "scripts/testing/rf3_stock_reference.py",
        "generated_at": dt.date.today().isoformat(),
        "foundry_commit": FOUNDRY_COMMIT,
        "checkpoint": CHECKPOINT_NAME,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "gpu": GPU,
        "command": "rf3 fold inputs=<json> ckpt_path=<checkpoint> out_dir=<dir> "
        + " ".join(f"{key}={value}" for key, value in SETTINGS.items()),
        "settings": SETTINGS,
        "references": {
            entry: {
                "pdb_entry": pdb_entry,
                "chains": [{"id": chr(65 + i), "sequence": seq} for i, seq in enumerate(sequences)],
            }
            for entry, (pdb_entry, sequences) in ENTRIES.items()
        },
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {sum(len(files) for files in written.values())} reference files and manifest.json to {DATA_DIR}")


if __name__ == "__main__":
    main()
