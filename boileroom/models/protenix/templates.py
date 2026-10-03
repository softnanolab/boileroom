"""Stage caller-supplied structure templates in the form Protenix 2.0 reads.

Protenix does not take a template structure directly. For every protein chain it
reads ``templatesPath``: an hmmsearch-style A3M (or an ``.hhr``) whose rows name
PDB entries (``<pdbid>_<chain>/<start>-<end> ... mol:protein length:<n>``), then
looks each entry up as ``<prot_template_mmcif_dir>/<pdbid>.cif``, refuses it
unless a release date is on record and no later than 2021-09-30, and realigns
the query to the structure with kalign.

So a caller's mmCIF has to be turned into three things: a CIF under a synthetic
four-character-style id that cannot collide with a real PDB entry, a hit file
aligning the query chain to it, and a release-date record. Nothing here
searches a database or touches the network.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from io import StringIO
from pathlib import Path

import numpy as np
from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal
from biotite.structure.io.pdbx import CIFCategory, CIFFile

#: Stamped on every staged template. It only has to precede the featurizer's
#: 2021-09-30 cutoff; a caller's structure has no meaningful PDB release date.
TEMPLATE_RELEASE_DATE = "2000-01-01"

#: Ids start with a letter, which no real PDB id does, so they can never shadow
#: an entry in a shared mmCIF directory.
ID_PREFIX = "x"

MAX_TEMPLATES = 4


def template_id(index: int) -> str:
    """Synthetic PDB-style id for the ``index``-th template."""
    return f"{ID_PREFIX}{index:03d}"


def stage_templates(templates: Mapping[str, str], query: str, stage_dir: Path) -> dict[str, str]:
    """Write everything Protenix needs to use ``templates`` for ``query``.

    ``templates`` maps a caller's name to mmCIF text. Returns the per-request
    settings the runtime applies: the ``templatesPath`` for the query chain and
    the mmCIF directory, release-date file and obsolete-entry file to point the
    featurizer at. Only the query chain gets templates; the others are left
    alone so Protenix does not try to search for them.
    """
    if not templates:
        raise ValueError("no templates to stage")
    if len(templates) > MAX_TEMPLATES:
        raise ValueError(f"at most {MAX_TEMPLATES} templates are supported, got {len(templates)}")

    mmcif_dir = stage_dir / "mmcif"
    mmcif_dir.mkdir(parents=True, exist_ok=True)
    rows = [f">query\n{query}\n"]
    release_dates: dict[str, dict[str, str]] = {}
    for index, name in enumerate(sorted(templates)):
        pdb_id = template_id(index)
        try:
            cif_text, chain, seqres = _normalise_cif(templates[name])
            rows.append(_hit_row(pdb_id, chain, seqres, query))
        except ValueError as error:
            raise ValueError(f"template {name!r}: {error}") from None
        (mmcif_dir / f"{pdb_id}.cif").write_text(cif_text, encoding="utf-8")
        release_dates[pdb_id] = {"release_date": TEMPLATE_RELEASE_DATE}

    hits = stage_dir / "hmmsearch.a3m"
    hits.write_text("".join(rows), encoding="utf-8")
    dates = stage_dir / "release_dates.json"
    dates.write_text(json.dumps(release_dates), encoding="utf-8")
    obsolete = stage_dir / "obsolete.json"
    obsolete.write_text("{}", encoding="utf-8")
    return {
        "templates_path": str(hits),
        "mmcif_dir": str(mmcif_dir),
        "release_dates_path": str(dates),
        "obsolete_pdbs_path": str(obsolete),
    }


def _normalise_cif(text: str) -> tuple[str, str, str]:
    """Return ``(cif_text, auth_chain, seqres)`` for a one-chain template.

    The release date is overwritten rather than trusted: Protenix drops any
    template whose revision date is after its cutoff, and a recent structure
    would otherwise vanish silently.
    """
    try:
        cif = CIFFile.read(StringIO(text))
        block = cif.block
    except Exception as error:  # biotite raises several unrelated types on bad input
        raise ValueError(f"not a readable mmCIF: {error}") from None
    for category in ("atom_site", "entity_poly_seq", "struct_asym"):
        if category not in block:
            raise ValueError(f"mmCIF has no _{category} loop, which Protenix needs to read a template")

    atoms = block["atom_site"]
    asyms = block["struct_asym"]
    poly = block["entity_poly_seq"]
    entity_of = dict(zip(asyms["id"].as_array(str), asyms["entity_id"].as_array(str), strict=False))
    polymer_entities = set(poly["entity_id"].as_array(str))
    # Ligands, ions and waters each carry their own label_asym_id; only polymer entities count as chains.
    label_chains = [
        chain
        for chain in _ordered_unique(atoms["label_asym_id"].as_array(str))
        if entity_of.get(chain) in polymer_entities
    ]
    if len(label_chains) != 1:
        raise ValueError(f"expected exactly one polymer chain, found {sorted(label_chains)}")
    label_chain = label_chains[0]
    on_chain = atoms["label_asym_id"].as_array(str) == label_chain
    if not on_chain.all():
        # Keep the staged structure to the template chain alone.
        block["atom_site"] = CIFCategory({name: atoms[name].as_array(str)[on_chain] for name in atoms})
        atoms = block["atom_site"]
    auth_chain = str(atoms["auth_asym_id"].as_array(str)[0]) if "auth_asym_id" in atoms else label_chain

    entity = entity_of[label_chain]
    residues = [
        (int(num), mon)
        for ent, num, mon in zip(
            poly["entity_id"].as_array(str), poly["num"].as_array(str), poly["mon_id"].as_array(str), strict=False
        )
        if ent == entity
    ]
    seqres = "".join(_one_letter(mon) for _, mon in sorted(residues))
    if not seqres:
        raise ValueError("the template chain has no residues in _entity_poly_seq")

    revision = CIFCategory(
        {
            "ordinal": ["1"],
            "data_content_type": ["Structure model"],
            "major_revision": ["1"],
            "minor_revision": ["0"],
            "revision_date": [TEMPLATE_RELEASE_DATE],
        }
    )
    block["pdbx_audit_revision_history"] = revision
    buffer = StringIO()
    cif.write(buffer)
    return buffer.getvalue(), auth_chain, seqres


def _one_letter(three: str) -> str:
    try:
        return ProteinSequence.convert_letter_3to1(three)
    except KeyError:
        return "X"


def _ordered_unique(values: np.ndarray) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(str(value), None)
    return list(seen)


def _hit_row(pdb_id: str, chain: str, seqres: str, query: str) -> str:
    """One hmmsearch-style A3M row aligning ``query`` to the template's SEQRES.

    Uppercase and ``-`` columns line up with the query, lowercase letters are
    template residues the query lacks, and the leading number in the header is
    the 1-based SEQRES position the row starts at. Protenix realigns with
    kalign afterwards, so this only has to be a sensible seed.
    """
    matrix = SubstitutionMatrix.std_protein_matrix()
    alignment = align_optimal(
        ProteinSequence(query),
        ProteinSequence(seqres.replace("X", "A")),
        matrix,
        gap_penalty=(-10, -1),
        terminal_penalty=False,
        local=False,
        max_number=1,
    )[0]
    trace = alignment.trace  # rows: (query index or -1, template index or -1)
    aligned = [(q, t) for q, t in trace if t != -1 and q != -1]
    if len(aligned) < 10:
        raise ValueError("the template shares fewer than 10 aligned residues with the target chain")
    first = next(i for i, (q, t) in enumerate(trace) if q != -1 and t != -1)
    last = max(i for i, (q, t) in enumerate(trace) if q != -1 and t != -1)
    cells: list[str] = []
    start_t = trace[first][1]
    end_t = trace[last][1]
    for q, t in trace[first : last + 1]:
        if q != -1 and t != -1:
            cells.append(seqres[t])
        elif q != -1:
            cells.append("-")
        else:
            cells.append(seqres[t].lower())
    header = f">{pdb_id}_{chain}/{start_t + 1}-{end_t + 1} mol:protein length:{len(seqres)}  template"
    # Columns before the first aligned query residue and after the last are
    # gaps against the query, so pad the row to the query's full width.
    lead = int(trace[first][0])
    tail = len(query) - 1 - int(trace[last][0])
    return f"{header}\n{'-' * lead}{''.join(cells)}{'-' * tail}\n"
