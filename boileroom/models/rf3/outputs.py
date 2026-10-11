"""Read RoseTTAFold 3 prediction files: AF3-style summary and per-atom confidences next to each model CIF."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

_SAMPLE_DIR = re.compile(r"seed-(\d+)_sample-(\d+)$")


def sample_identity(path: Path) -> tuple[int, int]:
    """Return the seed and diffusion-batch index of a ``seed-<s>_sample-<i>`` directory."""
    match = _SAMPLE_DIR.search(path.parent.name)
    if match is None:
        raise RuntimeError(f"Unrecognized RF3 sample path: {path}")
    return int(match.group(1)), int(match.group(2))


def read_json(path: Path, label: str = "RF3") -> dict[str, Any]:
    """Read a required confidence file, failing on incomplete prediction output."""
    if not path.is_file():
        raise RuntimeError(f"{label} produced incomplete output: missing {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} confidence must be a JSON object: {path.name}")
    return value


def early_stop_message(example_dir: Path, name: str) -> str | None:
    """Describe an early stop recorded in ``<name>_ranking_scores.csv``, or return ``None`` when there was none."""
    scores = example_dir / f"{name}_ranking_scores.csv"
    if not scores.is_file():
        return None
    header, *rows = scores.read_text(encoding="utf-8").splitlines()
    columns = header.split(",")
    if "early_stopped" not in columns or not rows:
        return None
    values = dict(zip(columns, rows[0].split(","), strict=False))
    return f"mean pLDDT {values.get('mean_plddt', 'unknown')} is below early_stopping_plddt_threshold"


def read_token_confidence(confidences: dict[str, Any], atoms: Any, label: str = "RF3") -> dict[str, np.ndarray]:
    """Map RF3's token-level PAE onto CIF chains and residues and average atom pLDDT per residue.

    RF3 writes ``atom_plddts`` (one value per CIF atom row), a token-by-token ``pae`` and
    ``token_chain_ids``. Its ``token_res_ids`` is only a running index, so residue numbers come
    from the CIF. The protein-only adapter requires every token to be one residue.
    """
    required = {"atom_plddts", "pae", "token_chain_ids"}
    if missing := required - confidences.keys():
        raise RuntimeError(f"{label} full confidence is missing {sorted(missing)}")
    pae = np.asarray(confidences["pae"], dtype=np.float32)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or not pae.size:
        raise RuntimeError(f"{label} PAE must be a nonempty square matrix, got {pae.shape}")
    if not np.isfinite(pae).all() or (pae < 0).any():
        raise RuntimeError(f"{label} PAE contains invalid values")
    atom_plddt = np.asarray(confidences["atom_plddts"], dtype=np.float32)
    if atom_plddt.shape != (len(atoms),):
        raise RuntimeError(f"{label} has {atom_plddt.size} atom pLDDT values for {len(atoms)} CIF atoms")
    if not np.isfinite(atom_plddt).all() or (atom_plddt < 0).any():
        raise RuntimeError(f"{label} atom pLDDT contains invalid values")

    chain_ids = np.asarray(atoms.chain_id)
    res_ids = np.asarray(atoms.res_id)
    changes = (chain_ids[1:] != chain_ids[:-1]) | (res_ids[1:] != res_ids[:-1])
    starts = np.concatenate([[0], np.flatnonzero(changes) + 1])
    n_tokens = len(pae)
    if len(starts) != n_tokens:
        raise RuntimeError(f"{label} PAE has {n_tokens} tokens but the CIF has {len(starts)} residues")
    token_chains = np.asarray(confidences["token_chain_ids"]).astype(str)
    if token_chains.shape != (n_tokens,):
        raise RuntimeError(f"{label} token chain IDs do not match the PAE")
    cif_chains = chain_ids[starts].astype(str)
    # RF3 names chains by instance ("A_1"); the mapping to the CIF's chain names must be one to one.
    pairs = set(zip(token_chains.tolist(), cif_chains.tolist(), strict=True))
    if len(pairs) != len(set(token_chains.tolist())) or len(pairs) != len(set(cif_chains.tolist())):
        raise RuntimeError(f"{label} token chain IDs disagree with the CIF chains")

    plddt = np.add.reduceat(atom_plddt, starts) / np.diff(np.append(starts, len(atoms)))
    return {
        "pae": pae,
        "token_chain_ids": cif_chains,
        "token_res_ids": res_ids[starts].astype(np.int64),
        "plddt": plddt.astype(np.float32),
        "atom_plddt": atom_plddt,
    }
