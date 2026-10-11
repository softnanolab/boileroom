"""Stage caller-supplied structure templates in the form Protenix 2.0 and OpenDDE read.

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

Upstream's featurizer drops hits without raising (``template_utils.py``: the
prefilter at 333-354 and 376-381, the de-duplication in ``get_templates``, and
``template_parser.py:341-394`` for a CIF without peptide ``_chem_comp`` types).
Staging therefore refuses, with ``ValueError``, every case it can see would be
dropped, and shapes the hit row so the prefilter keeps the target's own
structure. The worker counts what was featurized and fails the request on any
remaining difference.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

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

#: Upstream's prefilter refuses a hit whose aligned columns cover this fraction of the query or less.
MIN_ALIGN_RATIO = 0.1

#: Letters substituted into a self-template's hit row so it is not a substring of the query, tried in order.
MASK_LETTERS = "XBZJOU"

#: The ``_chem_comp.type`` stamped on a template's residues when its mmCIF has no ``_chem_comp`` loop.
PEPTIDE_CHEM_COMP_TYPE = "L-peptide linking"


@dataclass(frozen=True)
class StagedTemplates:
    """What :func:`stage_templates` wrote for one request.

    Attributes
    ----------
    templates_path : str
        The hit file to set as the templated chain's ``templatesPath``.
    mmcif_dir : str
        Directory of the staged ``<id>.cif`` files.
    release_dates_path : str
        JSON release-date record for the staged ids.
    obsolete_pdbs_path : str
        Empty obsolete-entry record.
    cache_dir : str
        A per-request template parse cache directory (OpenDDE writes parsed templates there).
    query : str
        The templated chain's sequence, as the featurizer sees it.
    count : int
        How many templates were staged; the worker checks that exactly this many were featurized.
    names : tuple[str, ...]
        The caller's template names, in staged-id order.
    """

    templates_path: str
    mmcif_dir: str
    release_dates_path: str
    obsolete_pdbs_path: str
    cache_dir: str
    query: str
    count: int
    names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the record as plain data for the worker process."""
        return {
            "templates_path": self.templates_path,
            "mmcif_dir": self.mmcif_dir,
            "release_dates_path": self.release_dates_path,
            "obsolete_pdbs_path": self.obsolete_pdbs_path,
            "cache_dir": self.cache_dir,
            "query": self.query,
            "count": self.count,
            "names": list(self.names),
        }


def template_id(index: int) -> str:
    """Synthetic PDB-style id for the ``index``-th template."""
    return f"{ID_PREFIX}{index:03d}"


def stage_templates(templates: Mapping[str, str], query: str, stage_dir: Path) -> StagedTemplates:
    """Write everything Protenix or OpenDDE needs to use ``templates`` for ``query``.

    Parameters
    ----------
    templates : Mapping[str, str]
        A caller's template name mapped to one-chain mmCIF text.
    query : str
        The templated chain's sequence.
    stage_dir : Path
        A request-private directory to write into.

    Returns
    -------
    StagedTemplates
        The hit file for the templated chain, the mmCIF directory, release-date and obsolete-entry files to
        point the featurizer at, a per-request cache directory, and the staged count. Other protein chains of
        the request need :func:`write_query_only_hits` so the featurizer does not search for them.

    Raises
    ------
    ValueError
        If there are no or too many templates, a template is unreadable or not one protein chain, two
        templates share a SEQRES or a hit row, or a template aligns to 10% of the query or less: every case
        upstream's featurizer would drop without an error.
    """
    if not templates:
        raise ValueError("no templates to stage")
    if len(templates) > MAX_TEMPLATES:
        raise ValueError(f"at most {MAX_TEMPLATES} templates are supported, got {len(templates)}")

    mmcif_dir = stage_dir / "mmcif"
    mmcif_dir.mkdir(parents=True, exist_ok=True)
    rows = [f">query\n{query}\n"]
    release_dates: dict[str, dict[str, str]] = {}
    names = tuple(sorted(templates))
    seen_seqres: dict[str, str] = {}
    seen_rows: dict[str, str] = {}
    seen_columns: dict[str, str] = {}
    for index, name in enumerate(names):
        pdb_id = template_id(index)
        try:
            cif_text, chain, seqres = _normalise_cif(templates[name])
            row = _hit_row(pdb_id, chain, seqres, query)
        except ValueError as error:
            raise ValueError(f"template {name!r}: {error}") from None
        body = row.split("\n", 1)[1].strip()
        # get_templates keeps one hit per gap-stripped hit_sequence, then one feature per template_sequence.
        for seen, key, what in (
            (seen_seqres, seqres, "SEQRES"),
            (seen_rows, body.replace("-", "").upper(), "alignment"),
            (seen_columns, "".join(c for c in body if not c.islower()), "aligned residues"),
        ):
            if key in seen:
                raise ValueError(
                    f"templates {seen[key]!r} and {name!r} have the same {what}; the featurizer keeps only one of "
                    "them, so pass one"
                )
            seen[key] = name
        rows.append(row)
        (mmcif_dir / f"{pdb_id}.cif").write_text(cif_text, encoding="utf-8")
        release_dates[pdb_id] = {"release_date": TEMPLATE_RELEASE_DATE}

    hits = stage_dir / "hmmsearch.a3m"
    hits.write_text("".join(rows), encoding="utf-8")
    dates = stage_dir / "release_dates.json"
    dates.write_text(json.dumps(release_dates), encoding="utf-8")
    obsolete = stage_dir / "obsolete.json"
    obsolete.write_text("{}", encoding="utf-8")
    cache_dir = stage_dir / "template_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return StagedTemplates(
        templates_path=str(hits),
        mmcif_dir=str(mmcif_dir),
        release_dates_path=str(dates),
        obsolete_pdbs_path=str(obsolete),
        cache_dir=str(cache_dir),
        query=query,
        count=len(names),
        names=names,
    )


def write_query_only_hits(query: str, path: Path) -> Path:
    """Write a hit file holding only the query, for a protein chain that gets no caller template.

    Upstream's template search skips a chain whose ``templatesPath`` exists, and a file with only the query
    row parses to zero hits, so the chain is featurized with no templates instead of searched (the kit images
    carry no hmmsearch, and a search against the staged directory finds nothing usable).

    Parameters
    ----------
    query : str
        The chain's sequence.
    path : Path
        Where to write the file; its parent is created.

    Returns
    -------
    Path
        ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">query\n{query}\n", encoding="utf-8")
    return path


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
    _ensure_peptide_chem_comp(block, [mon for _, mon in residues])

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


def _ensure_peptide_chem_comp(block: Any, monomers: list[str]) -> None:
    """Make upstream's parser see the template chain as protein.

    ``template_parser.py:341-362`` keeps a polymer only when one of its monomers has a ``_chem_comp.type``
    containing "peptide", and otherwise reports "No protein chains found" and drops the template. Writers
    such as gemmi may omit ``_chem_comp``; it is synthesised for the chain's monomers then. A loop that is
    present but types none of them as peptide is a non-protein chain, refused here.
    """
    unique = list(dict.fromkeys(monomers))
    if "chem_comp" not in block:
        block["chem_comp"] = CIFCategory({"id": unique, "type": [PEPTIDE_CHEM_COMP_TYPE] * len(unique)})
        return
    comps = block["chem_comp"]
    if "id" not in comps or "type" not in comps:
        raise ValueError("_chem_comp has no id or type column, so Protenix cannot tell the chain is protein")
    types = dict(zip(comps["id"].as_array(str), comps["type"].as_array(str), strict=False))
    if not any("peptide" in types.get(mon, "").lower() for mon in unique):
        raise ValueError("no residue of the template chain is a peptide in _chem_comp; only protein templates are used")


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
    the 1-based SEQRES position the row starts at. Upstream realigns the query to
    the template's full SEQRES with kalign (``template_utils.py:551-575``) and
    replaces the row's sequence, so the row only has to be a seed that passes
    the prefilter (``template_utils.py:333-354``):

    - the aligned columns must cover more than 10% of the query (else the hit is
      dropped as ``AlignRatioError``; refused here instead);
    - the gap-stripped row must not be a substring of the query covering more
      than 95% of it (``DuplicateError``), which the target's own structure always
      is; such a row gets one aligned residue replaced by a letter the query does
      not contain.
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
    ratio = len(aligned) / len(query)
    if ratio <= MIN_ALIGN_RATIO:
        raise ValueError(
            f"the template aligns to {len(aligned)} of {len(query)} target residues ({ratio:.2f}); "
            f"Protenix drops templates covering {MIN_ALIGN_RATIO:.0%} of the target or less"
        )
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
    if "".join(cells).replace("-", "").upper() in query:
        cells = _mask_first_aligned(cells, query)
    header = f">{pdb_id}_{chain}/{start_t + 1}-{end_t + 1} mol:protein length:{len(seqres)}  template"
    # Columns before the first aligned query residue and after the last are
    # gaps against the query, so pad the row to the query's full width.
    lead = int(trace[first][0])
    tail = len(query) - 1 - int(trace[last][0])
    return f"{header}\n{'-' * lead}{''.join(cells)}{'-' * tail}\n"


def _mask_first_aligned(cells: list[str], query: str) -> list[str]:
    """Replace the first aligned residue with a letter absent from ``query``.

    The row then cannot be a substring of the query, so upstream's ``DuplicateError`` prefilter keeps it;
    the aligned-column count is unchanged, and kalign's realignment against the full SEQRES discards the row.
    """
    letter = next((c for c in MASK_LETTERS if c not in query.upper()), None)
    if letter is None:
        raise ValueError(f"the target uses every letter of {MASK_LETTERS!r}, so its own structure cannot be staged")
    masked = list(cells)
    index = next(i for i, cell in enumerate(masked) if cell.isupper())
    masked[index] = letter
    return masked
