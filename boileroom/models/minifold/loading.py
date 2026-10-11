"""Pinned, integrity-checked MiniFold checkpoint acquisition and state loading."""

import argparse
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import httpx

UPSTREAM_COMMIT = "63db8b91f5bb64535f20810f014e69df66b0faad"
CHECKPOINT_REVISION = "f8612572818292ecef1c7fef580c0b17646ae0b1"
# Hugging Face LFS SHA256 values at CHECKPOINT_REVISION, verified 2026-10-11.
CHECKPOINT_SHA256 = {
    "12L": "f287377d4c5945e5abd186a35eaf7ddb6d9b68e8856aa416a95115f3b1fd2ccf",
    "48L": "6cc1d85edaf23ab2665e45a7c123d6d6f1a68f38d6e3a7cdb5fbaa4d459089bb",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _download_verified(url: str, destination: Path, expected: str) -> Path:
    """Stream to a unique temporary file and promote only a verified artifact."""
    if destination.is_file() and _sha256(destination) == expected:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            digest = hashlib.sha256()
            with httpx.stream(
                "GET", url, timeout=httpx.Timeout(120.0, connect=30.0), follow_redirects=True
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    stream.write(chunk)
                    digest.update(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        if digest.hexdigest() != expected:
            raise RuntimeError(f"MiniFold artifact SHA256 mismatch: {destination.name}")
        temporary.replace(destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def checkpoint_path(cache: Path, model_size: str) -> Path:
    """Return an atomic, SHA256-verified checkpoint at the pinned HF revision."""
    expected = CHECKPOINT_SHA256[model_size]
    destination = cache / f"minifold_{model_size}_{CHECKPOINT_REVISION}.ckpt"
    url = f"https://huggingface.co/jwohlwend/minifold/resolve/{CHECKPOINT_REVISION}/minifold_{model_size}_final.ckpt"
    return _download_verified(url, destination, expected)


ESM_MODEL_NAME = "esm2_t36_3B_UR50D"
# Official Meta artifacts, independently streamed and SHA256-checked 2026-10-11.
ESM_ARTIFACTS = {
    "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt": "7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500",
    "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt": "4da500eab246481dc9c8c95bc7b1d02f2803d761c380b0e95186d4a07d0fc84e",
}


def load_esm_artifact(url: str) -> dict:
    """Load only the two verified official backbone files with restricted unpickling.

    ESM's configuration uses argparse.Namespace. This single inert standard-library
    class is allowlisted in a scoped weights-only context, never arbitrary globals.
    Unknown artifacts and safe-load failures are refused without a pickle fallback.
    """
    import torch

    if url not in ESM_ARTIFACTS:
        raise ValueError(f"Unrecognized MiniFold ESM backbone artifact: {url}")
    checksum = ESM_ARTIFACTS[url]
    path = Path(torch.hub.get_dir()) / "checkpoints" / f"{checksum}-{url.rsplit('/', 1)[-1]}"
    _download_verified(url, path, checksum)
    with torch.serialization.safe_globals([argparse.Namespace]):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_state_dict(model: Any, state_dict: dict) -> None:
    """Normalize upstream training prefixes and fail if learned parameters do not match."""
    normalized = {
        key.replace("_orig_mod.", "").removeprefix("model."): value
        for key, value in state_dict.items()
        if not any(part in key for part in ("boundaries", "mid_points"))
    }
    incompatible = model.load_state_dict(normalized, strict=False)
    # Official MiniFold checkpoints omit frozen LM parameters, which have already
    # been loaded from the exact verified backbone. Fine-tuned LM layers must match.
    frozen_backbone = {
        name for name, parameter in model.named_parameters() if name.startswith("lm.") and not parameter.requires_grad
    }
    # Upstream on_save_checkpoint also omits deterministic rotary buffers for
    # frozen ESM layers 0..33; fine-tuned layers 34/35 must still be present.
    frozen_backbone.update(
        name
        for name, _ in model.named_buffers()
        if re.fullmatch(r"lm\.layers\.(?:[0-9]|[12][0-9]|3[0-3])\.self_attn\.rot_emb\.inv_freq", name)
    )
    missing = [key for key in incompatible.missing_keys if key not in frozen_backbone]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"MiniFold checkpoint mismatch: missing={missing}, unexpected={incompatible.unexpected_keys}"
        )
