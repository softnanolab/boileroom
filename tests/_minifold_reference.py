"""Independent pinned-upstream reference, shared by explicit validation entrypoints."""

import hashlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PREDICT_SHA256 = "600ba96e5dccd4fa534d3c5b59d75fffd67583b671db7da348e9b173728099f4"
MINIFORMER_SHA256 = "cda7edb9aa14e5f905659f99360f2a99233cd93b34a18c38730cdc6e742c16d4"


def _original_triangular(source: bytes) -> Callable[..., Any]:
    """Recover only the original function from independently hash-checked Git bytes."""
    import ast

    import torch
    import torch.nn.functional as functional

    assert hashlib.sha256(source).hexdigest() == MINIFORMER_SHA256, "Unexpected stock MiniFormer source"
    parsed = ast.parse(source)
    functions: list[ast.stmt] = [
        node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "triangular"
    ]
    assert len(functions) == 1
    namespace: dict[str, Any] = {"torch": torch, "F": functional}
    # The digest authenticates the complete pinned source before execution. Do not
    # execute its imports: kernel decorators are irrelevant to this stock path.
    exec(compile(ast.Module(body=functions, type_ignores=[]), "pinned-stock-miniformer.py", "exec"), namespace)
    return namespace["triangular"]


@contextmanager
def _stock_triangular(module: Any) -> Iterator[None]:
    """Bind original math during stock construction, restoring the installed adapter."""
    import subprocess

    from boileroom.models.minifold.loading import UPSTREAM_COMMIT

    source = subprocess.run(
        ["git", "-C", "/opt/minifold", "show", f"{UPSTREAM_COMMIT}:minifold/model/miniformer.py"],
        check=True,
        capture_output=True,
        timeout=10,
    ).stdout
    original = _original_triangular(source)
    installed = module.triangular
    module.triangular = original
    try:
        yield
    finally:
        module.triangular = installed


def run_stock(model_size: str, cases: list[dict]) -> dict:
    """Create a fresh independent stock model and return raw CPU tensors/provenance."""
    import gc
    import importlib.util
    import io
    import os
    from contextlib import nullcontext

    import numpy as np
    import torch
    from minifold.data.config import model_config
    from minifold.model import miniformer
    from minifold.model.model import MiniFoldModel
    from minifold.utils.residue_constants import atom_types

    from boileroom.models.minifold.loading import (
        CHECKPOINT_REVISION,
        CHECKPOINT_SHA256,
        ESM_ARTIFACTS,
        UPSTREAM_COMMIT,
        checkpoint_path,
    )
    from boileroom.provenance import runtime_provenance
    from boileroom.utils import get_model_dir

    source = Path("/opt/minifold/predict.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == PREDICT_SHA256
    assert os.environ["MINIFOLD_COMMIT"] == UPSTREAM_COMMIT
    spec = importlib.util.spec_from_file_location("stock_minifold_predict", source)
    assert spec and spec.loader
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    torch.set_float32_matmul_precision("highest")  # Pinned upstream CLI's explicit setting.
    cache = get_model_dir() / "minifold"
    previous_cache = torch.hub.get_dir()
    model = checkpoint = None
    try:
        checkpoint = torch.load(checkpoint_path(cache, model_size), map_location="cpu", weights_only=True)
        hparams = checkpoint["hyper_parameters"]
        assert hparams["esm_model_name"] == "esm2_t36_3B_UR50D"
        config = model_config("initial_training", train=False, low_prec=False, long_sequence_inference=False)
        torch.hub.set_dir(str(cache))
        with _stock_triangular(miniformer):
            model = MiniFoldModel(
                esm_model_name=hparams["esm_model_name"],
                num_blocks=hparams["num_blocks"],
                no_bins=hparams["no_bins"],
                config_of=config,
                use_structure_module=True,
                kernels=False,
            )
        # Match stock create_model normalization independently of the production loader.
        state = {
            key: value
            for key, value in checkpoint["state_dict"].items()
            if "boundaries" not in key and "mid_points" not in key
        }
        state = {key.replace("_orig_mod.", "").replace("model.", ""): value for key, value in state.items()}
        incompatible = model.load_state_dict(state, strict=False)
        assert not incompatible.unexpected_keys
        del state
        checkpoint = None
        model = model.eval().to("cuda:0")
        bf16 = torch.cuda.is_bf16_supported()
        artifacts = {}
        for case in cases:
            for recycling in [3, 0] if case is cases[0] else [3]:
                sequence = case["sequence"]
                seq, mask, features = upstream.prepare_input(sequence, config.data, model.lm.alphabet)
                batch = {
                    "seq": seq[None].cuda(),
                    "mask": mask[None].cuda(),
                    "batch_of": {key: value[None].cuda() for key, value in features.items()},
                }
                autocast = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
                with torch.inference_mode(), autocast:
                    raw = model(batch, num_recycling=recycling)
                    arrays = {
                        key: raw[key][0].float().cpu().numpy()
                        for key in ("final_atom_positions", "final_atom_mask", "plddt")
                    }
                stream = io.BytesIO()
                np.savez_compressed(stream, **arrays, atom_types=np.asarray(atom_types))
                artifacts[f"{case['id']}-r{recycling}.npz"] = stream.getvalue()
                del batch, raw, arrays
        return {
            "files": artifacts,
            "manifest": {
                "model_size": model_size,
                "upstream_commit": UPSTREAM_COMMIT,
                "predict_sha256": PREDICT_SHA256,
                "miniformer_sha256": MINIFORMER_SHA256,
                "triangular_implementation": "original-pinned-upstream",
                "checkpoint_revision": CHECKPOINT_REVISION,
                "checkpoint_sha256": CHECKPOINT_SHA256[model_size],
                "esm_artifacts": ESM_ARTIFACTS,
                "gpu": torch.cuda.get_device_name(),
                "precision": "fp32-lm/bf16-fold" if bf16 else "fp32",
                "matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "compile": False,
                "kernels": False,
                "batch_mode": "one-unpadded-sequence",
                "missing_state_keys": incompatible.missing_keys,
                "runtime": runtime_provenance(("minifold", "fair-esm", "torch")),
                "cases": [
                    {"id": case["id"], "sequence": case["sequence"], "experimental_sha256": case["sha256"]}
                    for case in cases
                ],
            },
        }
    finally:
        model = checkpoint = None
        torch.hub.set_dir(previous_cache)
        gc.collect()
        torch.cuda.empty_cache()
