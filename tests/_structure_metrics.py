"""Interface metrics, sequences and reference values shared by the kit integration tests.

Only ``numpy`` and the standard library are imported here, so the CPU unit tests (``tests/test_structure_metrics.py``) and
the GPU integration tests (``tests/*/test_*_kit_integration.py``) can both import this module at module scope.

The ipSAE definition is the one the bakeoff scores with (``evals/metrics/ipsae.py``, a port of DunbrackLab's
``ipsae.py`` at commit 6174cf9e): for each residue ``i`` of one chain, keep the residues ``j`` of the other chain with
``PAE[i, j] < cutoff``, convert those PAE values to a TM-like score with ``d0`` computed from how many residues were
kept, average, and take the maximum over ``i``. The headline value is the minimum of the two directions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

#: MDM2 N-terminal domain (residues 17-125), the target chain of the bakeoff p53 heterodimer.
TARGET: Final[str] = (
    "SQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVVVNQQESSDSGTSVSEN"
)
#: The binder chain: the p53 transactivation peptide (binds MDM2) and a scrambled decoy of the same composition.
BINDERS: Final[Mapping[str, str]] = {"p53": "SQETFSDLWKLLPEN", "decoy": "LPNKSWDLTFSEQLE"}
#: Seeds the p53 golden is averaged over (one diffusion sample per seed), and the one seed the decoy is folded with.
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
DECOY_SEED: Final[int] = 0
#: PAE cutoff of the ipSAE definition (Å).
PAE_CUTOFF: Final[float] = 10.0
#: Allowed distance of a seed-averaged ipSAE from its golden, and of a kit mode's mean from the vanilla mean.
TOLERANCE: Final[float] = 0.02
OPTIMIZATION_MODES: Final[tuple[str, ...]] = ("vanilla", "exact", "fast")

#: Golden ipSAE (min of both directions, PAE cutoff 10 Å) per family and mode: ``p53`` is the mean over :data:`SEEDS`
#: with one diffusion sample each, ``decoy`` the value at :data:`DECOY_SEED`. ``None`` = not measured yet; a test that
#: needs it skips with an explicit message.
#:
#: ESMFold2 (biohub/ESMFold2, A100-80GB, num_loops=3, num_sampling_steps=50, num_diffusion_samples=1,
#: msa_max_depth=1024, msa_column_mask_rate=0.1, no MSA; measured 2026-10 during the PR #116 review): vanilla on the
#: stock image 0.4.4-alpha.9, seeds 0/1/2 = 0.38205/0.39766/0.38321, decoy 0.01418; exact on kit image sha-dc652b0,
#: 0.38490/0.38516/0.39988, decoy 0.01439; fast, 0.38788/0.38854/0.39549, decoy 0.01413. (The pre-fix image 8ddf62e1
#: that dropped the sliding attention window scored p53 ≈ 0.291: the regression this golden guards against.)
#:
#: Protenix (protenix-v2, A100-40GB, cycle=10, step=200, sample=1, bf16, use_msa=True with ``msa=[tests/data/mdm2.a3m,
#: None]``, no templates, no TFG guidance): vanilla 0.59122/0.59173/0.57400, decoy 0.08789; exact
#: 0.59222/0.59232/0.57284, decoy 0.08746; fast 0.59279/0.59230/0.57327, decoy 0.08846.
#:
#: OpenDDE (opendde_v1, same protocol as Protenix, A100-40GB, image sha-06a2b81593d2; measured 2026-10-06): vanilla
#: 0.53490/0.53364/0.53552, decoy 0.22831; exact 0.53496/0.53370/0.53458, decoy 0.22996; fast 0.53410/0.53311/0.53471,
#: decoy 0.22573. Vanilla with the torch triangle kernels: 0.53477/0.53376/0.53432.
GOLDENS: Final[Mapping[str, Mapping[str, Mapping[str, float | None]]]] = {
    "esmfold2": {
        "vanilla": {"p53": 0.38764, "decoy": 0.01418},
        "exact": {"p53": 0.38998, "decoy": 0.01439},
        "fast": {"p53": 0.39064, "decoy": 0.01413},
    },
    "protenix": {
        "vanilla": {"p53": 0.58565, "decoy": 0.08789},
        "exact": {"p53": 0.58579, "decoy": 0.08746},
        "fast": {"p53": 0.58612, "decoy": 0.08846},
    },
    "opendde": {
        "vanilla": {"p53": 0.53468, "decoy": 0.22831},
        "exact": {"p53": 0.53441, "decoy": 0.22996},
        "fast": {"p53": 0.53398, "decoy": 0.22573},
    },
}


@dataclass(frozen=True)
class Separation:
    """How far the binder must score above the decoy for a family.

    Attributes
    ----------
    p53_min : float | None
        Lower bound of the seed-averaged p53 ipSAE, or ``None`` to skip it.
    decoy_max : float | None
        Upper bound of the decoy ipSAE, or ``None`` to skip it.
    margin : float
        Minimum of ``p53 mean - decoy``; always checked.
    """

    p53_min: float | None
    decoy_max: float | None
    margin: float


#: The bounds sit well inside the measured values: ESMFold2 p53 ≈ 0.39 vs decoy ≈ 0.014, Protenix ≈ 0.59 vs ≈ 0.09,
#: OpenDDE ≈ 0.53 vs ≈ 0.23 (its decoy scores higher, so its margin is narrower).
SEPARATION: Final[Mapping[str, Separation]] = {
    "esmfold2": Separation(p53_min=0.25, decoy_max=0.05, margin=0.2),
    "protenix": Separation(p53_min=0.45, decoy_max=0.2, margin=0.3),
    "opendde": Separation(p53_min=0.4, decoy_max=0.35, margin=0.2),
}


@dataclass(frozen=True)
class IpsaeScore:
    """ipSAE of one two-chain prediction.

    Attributes
    ----------
    target_to_binder : float
        Rows of the target chain (``chain_index == 0``) scored against the binder columns.
    binder_to_target : float
        Rows of the binder chain (``chain_index == 1``) scored against the target columns.
    """

    target_to_binder: float
    binder_to_target: float

    @property
    def min(self) -> float:
        """The headline ipSAE: the smaller of the two directions."""
        return min(self.target_to_binder, self.binder_to_target)


def d0(n: np.ndarray | int) -> np.ndarray:
    """Return the TM-score distance scale ``d0`` for ``n`` aligned residues, as ipSAE uses it.

    Parameters
    ----------
    n : np.ndarray | int
        Number of residues kept per row (``n < 26`` is treated as 26).

    Returns
    -------
    np.ndarray
        ``max(1, 1.24 * (max(26, n) - 15) ** (1/3) - 1.8)``, elementwise.
    """
    clipped = np.maximum(26, np.asarray(n, dtype=float))
    return np.maximum(1.0, 1.24 * np.cbrt(clipped - 15.0) - 1.8)


def _ptm(pae: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + (pae / scale) ** 2)


def asymmetric_ipsae(pae: np.ndarray, rows: np.ndarray, columns: np.ndarray, pae_cutoff: float = PAE_CUTOFF) -> float:
    """Return ipSAE of ``rows`` scored against ``columns`` (one direction).

    Parameters
    ----------
    pae : np.ndarray
        Square ``(N, N)`` predicted aligned error in Å, aligned on the row residue.
    rows, columns : np.ndarray
        Boolean masks of length ``N`` selecting the two chains.
    pae_cutoff : float
        Pairs with ``PAE >= pae_cutoff`` are dropped.

    Returns
    -------
    float
        The maximum over row residues of the mean TM-like score over the kept pairs, or ``0.0`` when no pair is kept.
    """
    valid = np.outer(rows, columns) & (pae < pae_cutoff)
    scale = d0(valid.sum(axis=1))
    best = 0.0
    for i in np.flatnonzero(rows):
        if valid[i].any():
            best = max(best, float(_ptm(pae[i], scale[i])[valid[i]].mean()))
    return best


def ipsae(pae: np.ndarray, chain_index: np.ndarray, pae_cutoff: float = PAE_CUTOFF) -> IpsaeScore:
    """Return the ipSAE of a two-chain prediction.

    Parameters
    ----------
    pae : np.ndarray
        Square ``(N, N)`` predicted aligned error in Å.
    chain_index : np.ndarray
        Length-``N`` chain label per token: ``0`` for the target, ``1`` for the binder.
    pae_cutoff : float
        PAE cutoff in Å.

    Returns
    -------
    IpsaeScore
        Both directions; ``.min`` is the headline value.

    Raises
    ------
    ValueError
        If ``pae`` is not a finite square matrix, ``chain_index`` does not match it, or the labels are not exactly
        ``{0, 1}``.
    """
    pae = np.asarray(pae, dtype=float)
    chains = np.asarray(chain_index)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"pae must be a square matrix, got shape {pae.shape}")
    if not np.isfinite(pae).all():
        raise ValueError("pae has non-finite entries")
    if chains.shape != (pae.shape[0],):
        raise ValueError(f"chain_index has shape {chains.shape}, expected ({pae.shape[0]},) to match pae")
    if set(np.unique(chains).tolist()) != {0, 1}:
        raise ValueError(f"chain_index must label both chains with 0 and 1, got {sorted(set(chains.tolist()))}")
    target = chains == 0
    binder = chains == 1
    return IpsaeScore(
        target_to_binder=asymmetric_ipsae(pae, target, binder, pae_cutoff),
        binder_to_target=asymmetric_ipsae(pae, binder, target, pae_cutoff),
    )


def chain_index_from_lengths(target_length: int, binder_length: int) -> np.ndarray:
    """Return the chain labels of a target followed by a binder, one token per residue."""
    if target_length < 1 or binder_length < 1:
        raise ValueError("both chains need at least one residue")
    return np.concatenate([np.zeros(target_length, dtype=int), np.ones(binder_length, dtype=int)])


def chain_index_from_token_chain_ids(token_chain_ids: Sequence[str] | np.ndarray) -> np.ndarray:
    """Return chain labels from per-token chain ids: the last token's chain is the binder (``1``).

    Raises
    ------
    ValueError
        If the tokens do not come from exactly two chains, target first.
    """
    ids = np.asarray(token_chain_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("token_chain_ids must be a non-empty 1-D sequence")
    if len(np.unique(ids)) != 2 or ids[0] == ids[-1]:
        raise ValueError(f"expected two chains, target first, got {sorted(set(ids.tolist()))}")
    return (ids == ids[-1]).astype(int)


def scalar(value: object) -> float:
    """Return a one-element array, list or number as a ``float`` (ipTM / pTM fields)."""
    array = np.asarray(value, dtype=float).ravel()
    if array.size != 1:
        raise ValueError(f"expected one value, got {array.size}")
    return float(array[0])


def ca_coordinates(atom_array: Any) -> np.ndarray:
    """Return the ``(N, 3)`` CA coordinates of a biotite ``AtomArray``, in residue order."""
    names = np.asarray(atom_array.atom_name)
    coords = np.asarray(atom_array.coord, dtype=float)
    return coords[names == "CA"]


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Return the pairwise Euclidean distance matrix of ``(N, 3)`` coordinates."""
    coords = np.asarray(coords, dtype=float)
    return np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)


def mean_distance_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Return the mean absolute difference of the CA distance matrices of two same-length structures (Å).

    Superposition-free, so two predictions can be compared without aligning them.
    """
    if np.shape(a) != np.shape(b):
        raise ValueError(f"structures differ in shape: {np.shape(a)} vs {np.shape(b)}")
    return float(np.abs(distance_matrix(a) - distance_matrix(b)).mean())


def drop_residues_in_cif(cif_text: str, first_seq_id: int) -> str:
    """Return ``cif_text`` without the atoms of residues ``label_seq_id >= first_seq_id``.

    A text edit of the ``_atom_site`` loop only, so SEQRES, entities and every other category stay byte-identical: the
    result describes the same sequence with structure for its first part only (a template that must change a
    prediction, and that a stale template cache would fold like the full one). Atoms without a residue number (``.`` or
    ``?``, e.g. waters) are kept.

    Raises
    ------
    ValueError
        If the file has no single ``_atom_site`` loop with ``label_seq_id``, an ``_atom_site`` row does not split into
        one field per column (a quoted value with a space, a multi-line value), or the edit would remove no atom or
        every atom.
    """
    out: list[str] = []
    state = "outside"  # outside | header (tags after loop_) | rows
    columns: list[str] = []
    atom_site_loops = kept = dropped = seq_column = 0
    for line in cif_text.splitlines(keepends=True):
        fields = line.split()
        token = fields[0] if fields else ""
        if token == "loop_" or token.startswith(("data_", "save_")):
            # A loop ends only at the next loop, tag or block; blank lines and comments do not end it.
            state, columns = ("header" if token == "loop_" else "outside"), []
        elif token.startswith("_"):
            if state == "header":
                columns.append(token)
            else:
                if token.startswith("_atom_site."):
                    raise ValueError("an _atom_site category outside a loop is not supported")
                state, columns = "outside", []
        elif token and not token.startswith("#") and state == "header":
            state = "rows"
            if columns and columns[0].startswith("_atom_site."):
                atom_site_loops += 1
                if atom_site_loops > 1:
                    raise ValueError("more than one _atom_site loop")
                if "_atom_site.label_seq_id" not in columns:
                    raise ValueError("_atom_site lacks label_seq_id")
                seq_column = columns.index("_atom_site.label_seq_id")
            else:
                columns = []
        if state == "rows" and columns and token and not token.startswith("#"):
            if len(fields) != len(columns):
                raise ValueError(f"an _atom_site row does not split into {len(columns)} fields: {line!r}")
            seq_id = fields[seq_column]
            if seq_id not in (".", "?") and int(seq_id) >= first_seq_id:
                dropped += 1
                continue
            kept += 1
        out.append(line)
    if dropped == 0:
        raise ValueError(f"no atom with label_seq_id >= {first_seq_id} in an _atom_site loop")
    if kept == 0:
        raise ValueError(f"dropping label_seq_id >= {first_seq_id} would remove every atom")
    return "".join(out)


def cif_protein_sequence(cif_text: str) -> str:
    """Return the one-letter SEQRES of the single polymer entity of an mmCIF (``pdbx_seq_one_letter_code_can``)."""
    import io

    from biotite.structure.io import pdbx

    block = pdbx.CIFFile.read(io.StringIO(cif_text)).block
    codes = block["entity_poly"]["pdbx_seq_one_letter_code_can"].as_array(str)
    if len(codes) != 1:
        raise ValueError(f"expected one polymer entity, got {len(codes)}")
    return "".join(str(codes[0]).split())
