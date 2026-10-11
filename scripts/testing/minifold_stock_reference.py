"""Generate raw, independent pinned-upstream MiniFold references on one bounded Modal worker.

Paid execution must be explicitly approved. This script never deploys or changes credentials.
Run with a validated candidate BOILEROOM_IMAGE_TAG, then point integration tests at --output.
The reference constructs its own upstream model, not MiniFoldCore, and uses upstream prepare_input.
Only artifact acquisition is shared for integrity/security. Sources, precision and hashes are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import modal

from boileroom.images.modal import get_modal_image
from boileroom.images.volumes import model_weights
from boileroom.utils import MODAL_MODEL_DIR

ROOT = Path(__file__).resolve().parents[2]
PREDICT_SHA256 = "600ba96e5dccd4fa534d3c5b59d75fffd67583b671db7da348e9b173728099f4"
app = modal.App("boileroom-minifold-stock-reference")


@app.function(
    image=get_modal_image("minifold").add_local_dir(ROOT / "tests", remote_path="/validation/tests"), gpu="L4", memory=32768, timeout=1200,
    volumes={MODAL_MODEL_DIR: model_weights}, max_containers=1,
)
def run_stock(model_size: str, cases: list[dict]) -> dict:
    """Execute the independent reference implementation in the canonical runtime."""
    import sys

    sys.path.insert(0, "/validation/tests")
    from _minifold_reference import run_stock as reference

    return reference(model_size, cases)


def main() -> None:
    """Run only the named variant; write references and their checksums outside tracked source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("12L", "48L"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads((ROOT / "tests/data/minifold/manifest.json").read_text())["entries"]
    with app.run():
        result = run_stock.remote(args.model_size, cases)
    target = args.output / args.model_size
    target.mkdir(parents=True, exist_ok=True)
    for filename, content in result["files"].items():
        (target / filename).write_bytes(content)
    manifest = result["manifest"]
    manifest["files"] = {filename: hashlib.sha256(content).hexdigest() for filename, content in result["files"].items()}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote independent {args.model_size} references to {target}")


if __name__ == "__main__":
    main()
