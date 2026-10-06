"""Read Protenix 2.0 confidence data using its explicit atom-to-token map."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def sample_identity(path: Path, label: str = "Protenix") -> tuple[int, int]:
    """Return seed and within-seed confidence rank, not diffusion sample index."""
    try:
        seed = int(path.parent.parent.name.removeprefix("seed_"))
        rank = int(path.stem.rsplit("_sample_", 1)[1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"Unrecognized {label} sample path: {path}") from exc
    return seed, rank


def read_json(path: Path, label: str = "Protenix") -> dict[str, Any]:
    """Read a required confidence file, failing on incomplete prediction output."""
    if not path.is_file():
        raise RuntimeError(f"{label} produced incomplete output: missing {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} confidence must be a JSON object: {path.name}")
    return value


def read_token_confidence(full: dict[str, Any], atoms: Any, label: str = "Protenix") -> dict[str, np.ndarray]:
    """Map PAE tokens to CIF chain/residue IDs and average atom pLDDT per token.

    Protenix 2.0.0 writes ``token_pair_pae``, ``atom_to_token_idx`` and
    ``token_asym_id``. Atom rows have the same order as the prediction CIF.
    The protein-only adapter requires each token to represent one residue.
    """
    required = {"token_pair_pae", "atom_to_token_idx", "token_asym_id", "atom_plddt"}
    if missing := required - full.keys():
        raise RuntimeError(f"{label} full confidence is missing {sorted(missing)}")
    pae = np.asarray(full["token_pair_pae"], dtype=np.float32)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or not pae.size:
        raise RuntimeError(f"{label} PAE must be a nonempty square matrix, got {pae.shape}")
    if not np.isfinite(pae).all() or (pae < 0).any():
        raise RuntimeError(f"{label} PAE contains invalid values")
    n_tokens = len(pae)
    raw_mapping = np.asarray(full["atom_to_token_idx"])
    if raw_mapping.shape != (len(atoms),) or not np.issubdtype(raw_mapping.dtype, np.integer):
        raise RuntimeError(f"{label} atom-to-token indices do not match CIF atom rows")
    mapping = raw_mapping.astype(np.int64)
    if not np.array_equal(np.unique(mapping), np.arange(n_tokens)):
        raise RuntimeError(f"{label} atom-to-token mapping does not cover exactly the PAE tokens")
    asym = np.asarray(full["token_asym_id"])
    atom_plddt = np.asarray(full["atom_plddt"], dtype=np.float32)
    if asym.shape != (n_tokens,) or atom_plddt.shape != (len(atoms),):
        raise RuntimeError(f"{label} confidence dimensions do not match structure and PAE")
    if not np.isfinite(atom_plddt).all() or (atom_plddt < 0).any() or (atom_plddt > 1).any():
        raise RuntimeError(f"{label} 2.0 atom pLDDT must be in [0, 1]")
    chains: list[str] = []
    residues: list[int] = []
    for token in range(n_tokens):
        rows = atoms[mapping == token]
        if len(np.unique(rows.chain_id)) != 1 or len(np.unique(rows.res_id)) != 1:
            raise RuntimeError(f"{label} token spans multiple protein residues")
        chains.append(str(rows.chain_id[0]))
        residues.append(int(rows.res_id[0]))
    chain_ids = np.asarray(chains)
    for chain in np.unique(chain_ids):
        if len(np.unique(asym[chain_ids == chain])) != 1:
            raise RuntimeError(f"{label} token asymmetric IDs disagree with CIF chain IDs")
    if len(np.unique(asym)) != len(np.unique(chain_ids)):
        raise RuntimeError(f"{label} token asymmetric IDs merge distinct CIF chains")
    counts = np.bincount(mapping, minlength=n_tokens)
    plddt = np.bincount(mapping, weights=atom_plddt, minlength=n_tokens) / counts
    return {
        "pae": pae,
        "token_chain_ids": chain_ids,
        "token_res_ids": np.asarray(residues),
        "plddt": plddt.astype(np.float32),
        "atom_plddt": atom_plddt,
    }
