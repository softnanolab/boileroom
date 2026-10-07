"""Template staging and the monomer-with-MSA input, without importing Protenix."""

import io
import json
from pathlib import Path

import pytest
from biotite.structure.io.pdbx import CIFFile

from boileroom.models.protenix.core import ProtenixCore
from boileroom.models.protenix.templates import (
    PEPTIDE_CHEM_COMP_TYPE,
    TEMPLATE_RELEASE_DATE,
    StagedTemplates,
    stage_templates,
    template_id,
    write_query_only_hits,
)

CIF = Path(__file__).parents[1] / "data" / "chai" / "pred.rank_0.cif"


def _seqres(text: str) -> str:
    from biotite.sequence import ProteinSequence

    block = CIFFile.read(io.StringIO(text)).block
    return "".join(ProteinSequence.convert_letter_3to1(m) for m in block["entity_poly_seq"]["mon_id"].as_array(str))


@pytest.fixture
def cif_text() -> str:
    return CIF.read_text()


@pytest.fixture
def query(cif_text) -> str:
    # Shifted and mutated so the template is not a near-duplicate of the query.
    return "MK" + _seqres(cif_text)[3:-2] + "GGAA"


def test_stage_writes_hit_file_cif_and_release_dates(tmp_path, cif_text, query) -> None:
    staged = stage_templates({"mine": cif_text}, query, tmp_path)

    hits = Path(staged.templates_path).read_text().splitlines()
    assert hits[0] == ">query" and hits[1] == query
    assert hits[2].startswith(f">{template_id(0)}_A/") and "mol:protein" in hits[2]
    aligned = "".join(c for c in hits[3] if not c.islower())
    assert len(aligned) == len(query)
    dates = json.loads(Path(staged.release_dates_path).read_text())
    assert dates == {template_id(0): {"release_date": TEMPLATE_RELEASE_DATE}}
    assert json.loads(Path(staged.obsolete_pdbs_path).read_text()) == {}
    # The staged structure carries the synthetic date, not whatever the caller's said.
    staged_cif = CIFFile.read(Path(staged.mmcif_dir) / f"{template_id(0)}.cif").block
    assert staged_cif["pdbx_audit_revision_history"]["revision_date"].as_array(str)[0] == TEMPLATE_RELEASE_DATE
    assert isinstance(staged, StagedTemplates)
    assert staged.count == 1 and staged.names == ("mine",) and staged.query == query
    assert Path(staged.cache_dir).is_dir() and Path(staged.cache_dir).parent == tmp_path


def test_staged_record_is_what_the_worker_reads(tmp_path, cif_text, query) -> None:
    """``to_dict()`` crosses the process boundary; it carries every key the runtime reads."""
    staged = stage_templates({"mine": cif_text}, query, tmp_path).to_dict()
    assert json.loads(json.dumps(staged)) == staged
    for key in ("mmcif_dir", "release_dates_path", "obsolete_pdbs_path", "cache_dir", "query", "count"):
        assert key in staged
    assert staged["names"] == ["mine"] and staged["count"] == 1


def _prefilter(query: str, row_body: str) -> str | None:
    """Upstream's hit prefilter (``template_utils.py:333-354``), reimplemented: the error it would drop with.

    ``parse_hmmsearch_a3m`` upper-cases the hit row and counts uppercase non-gap columns as aligned.
    """
    hit_sequence = row_body.upper()
    aligned_cols = sum(c.isupper() and c != "-" for c in row_body)
    template_sequence = hit_sequence.replace("-", "")
    if aligned_cols / len(query) <= 0.1:
        return "AlignRatioError"
    if template_sequence in query and len(template_sequence) / len(query) > 0.95:
        return "DuplicateError"
    if len(template_sequence) < 10:
        return "LengthError"
    return None


@pytest.mark.parametrize("trim", [0, 3], ids=["identical", "template-longer"])
def test_target_own_structure_passes_upstream_prefilter(tmp_path, cif_text, trim) -> None:
    """A template of the target itself is the common case; upstream drops its row as a duplicate unless masked."""
    query = _seqres(cif_text)[trim:]
    staged = stage_templates({"self": cif_text}, query, tmp_path)
    body = Path(staged.templates_path).read_text().splitlines()[3]
    assert body.replace("-", "").upper() not in query
    assert _prefilter(query, body) is None
    # Masking swaps one residue, so the aligned-column count (and the align ratio) is unchanged.
    assert sum(c.isupper() for c in body) == len(query)


def test_shifted_query_row_is_left_unmasked(tmp_path, cif_text, query) -> None:
    staged = stage_templates({"mine": cif_text}, query, tmp_path)
    body = Path(staged.templates_path).read_text().splitlines()[3]
    assert _prefilter(query, body) is None
    assert set(body.upper()) - set(query) - {"-"} == set()


def test_duplicate_templates_are_refused(tmp_path, cif_text, query) -> None:
    """Upstream keeps one of two templates with the same sequence, silently."""
    with pytest.raises(ValueError, match="templates 'a' and 'b' have the same SEQRES"):
        stage_templates({"a": cif_text, "b": cif_text}, query, tmp_path)


def test_template_covering_ten_percent_or_less_is_refused(tmp_path, cif_text) -> None:
    """Upstream drops a hit aligned to 10% of the query or less (``AlignRatioError``); staging says so."""
    query = _seqres(cif_text)[:15] + "W" * 200
    with pytest.raises(ValueError, match=r"template 'short': the template aligns to \d+ of 215 .*10%"):
        stage_templates({"short": cif_text}, query, tmp_path)


def _without_chem_comp(text: str) -> str:
    cif = CIFFile.read(io.StringIO(text))
    del cif.block["chem_comp"]
    out = io.StringIO()
    cif.write(out)
    return out.getvalue()


def test_template_without_chem_comp_gets_peptide_types(tmp_path, cif_text, query) -> None:
    """Upstream's parser drops a chain with no peptide ``_chem_comp.type`` ("No protein chains found")."""
    staged = stage_templates({"gemmi": _without_chem_comp(cif_text)}, query, tmp_path)
    block = CIFFile.read(Path(staged.mmcif_dir) / f"{template_id(0)}.cif").block
    comps = dict(zip(block["chem_comp"]["id"].as_array(str), block["chem_comp"]["type"].as_array(str), strict=True))
    monomers = set(block["entity_poly_seq"]["mon_id"].as_array(str))
    assert set(comps) == monomers and set(comps.values()) == {PEPTIDE_CHEM_COMP_TYPE}


def test_template_with_no_peptide_chem_comp_is_refused(tmp_path, cif_text, query) -> None:
    cif = CIFFile.read(io.StringIO(cif_text))
    comps = cif.block["chem_comp"]
    columns = {name: list(comps[name].as_array(str)) for name in comps}
    columns["type"] = ["non-polymer"] * len(columns["type"])
    from biotite.structure.io.pdbx import CIFCategory

    cif.block["chem_comp"] = CIFCategory(columns)
    out = io.StringIO()
    cif.write(out)
    with pytest.raises(ValueError, match="no residue of the template chain is a peptide"):
        stage_templates({"rna": out.getvalue()}, query, tmp_path)


def test_query_only_hits_file_holds_no_hits(tmp_path) -> None:
    path = write_query_only_hits("CCCCCCCCCC", tmp_path / "chain_1" / "hits.a3m")
    assert path.read_text() == ">query\nCCCCCCCCCC\n"


def _with_water(text: str) -> str:
    """Append a water (its own entity and label_asym_id, as in any RCSB entry) to the template."""
    from biotite.structure.io.pdbx import CIFCategory

    cif = CIFFile.read(io.StringIO(text))
    block = cif.block
    atoms = block["atom_site"]
    columns = {name: list(atoms[name].as_array(str)) for name in atoms}
    for values in columns.values():
        values.append(values[-1])
    columns["label_asym_id"][-1] = "Z"
    columns["auth_asym_id"][-1] = "Z"
    columns["label_comp_id"][-1] = "HOH"
    block["atom_site"] = CIFCategory(columns)
    asyms = block["struct_asym"]
    rows = {name: list(asyms[name].as_array(str)) for name in asyms}
    for values in rows.values():
        values.append(values[-1])
    rows["id"][-1] = "Z"
    rows["entity_id"][-1] = "99"
    block["struct_asym"] = CIFCategory(rows)
    out = io.StringIO()
    cif.write(out)
    return out.getvalue()


def test_stage_ignores_waters_and_ligands_and_strips_them(tmp_path, cif_text, query) -> None:
    staged = stage_templates({"holo": _with_water(cif_text)}, query, tmp_path)

    staged_cif = CIFFile.read(Path(staged.mmcif_dir) / f"{template_id(0)}.cif").block
    assert set(staged_cif["atom_site"]["label_asym_id"].as_array(str)) == {"A"}


def test_two_polymer_chains_are_still_rejected(tmp_path, cif_text, query) -> None:
    two = _with_water(cif_text).replace(" HOH ", " ALA ")
    cif = CIFFile.read(io.StringIO(two))
    block = cif.block
    block["struct_asym"]["entity_id"] = block["struct_asym"]["entity_id"].as_array(str).tolist()[:-1] + [
        block["struct_asym"]["entity_id"].as_array(str)[0]
    ]
    out = io.StringIO()
    cif.write(out)
    with pytest.raises(ValueError, match="exactly one polymer chain"):
        stage_templates({"two": out.getvalue()}, query, tmp_path)


def test_stage_rejects_unusable_templates(tmp_path, cif_text, query) -> None:
    with pytest.raises(ValueError, match="at most"):
        stage_templates({str(i): cif_text for i in range(5)}, query, tmp_path)
    with pytest.raises(ValueError, match="template 'bad'"):
        stage_templates({"bad": "not a cif"}, query, tmp_path)
    with pytest.raises(ValueError, match="fewer than 10"):
        stage_templates({"far": cif_text}, "W" * 8, tmp_path)


def test_templates_path_goes_on_the_chosen_chain_only(tmp_path, cif_text) -> None:
    core = ProtenixCore()
    target = _seqres(cif_text)
    config = {**core.config, "templates": {"mine": cif_text}, "templates_chain": 0}
    staged = core._stage_templates(f"{target}:CCCCCCCCCC", tmp_path, config)
    assert staged is not None
    path = core._write_input_json(f"{target}:CCCCCCCCCC", tmp_path, None, staged, 0)
    chains = json.loads(path.read_text())[0]["sequences"]
    assert chains[0]["proteinChain"]["templatesPath"] == staged.templates_path
    # The other chain reads a query-only hit file: featurized without templates, never searched.
    assert Path(chains[1]["proteinChain"]["templatesPath"]).read_text() == ">query\nCCCCCCCCCC\n"


def test_no_templates_leaves_input_unchanged(tmp_path) -> None:
    core = ProtenixCore()
    assert core._stage_templates("AAAA:CCCC", tmp_path, {**core.config}) is None


def test_templates_chain_must_exist(tmp_path, cif_text) -> None:
    core = ProtenixCore()
    with pytest.raises(ValueError, match="templates_chain"):
        core._stage_templates(
            "AAAA:CCCC", tmp_path, {**core.config, "templates": {"m": cif_text}, "templates_chain": 2}
        )


def test_unsupporting_family_refuses_templates_given_at_construction(cif_text) -> None:
    """The shared refusal checks the merged config, so templates set through ``config=`` are refused too."""

    class NoTemplates(ProtenixCore):
        SUPPORTS_USER_TEMPLATES = False

    with pytest.raises(ValueError, match="Protenix does not support user-supplied 'templates'"):
        NoTemplates({"templates": {"m": cif_text}})
    core = NoTemplates()
    with pytest.raises(ValueError, match="does not support user-supplied 'templates'"):
        core._merge_options({"templates": {"m": cif_text}})


def test_opendde_stages_templates_like_protenix(tmp_path, cif_text, query) -> None:
    from boileroom.models.opendde.core import OpenDDECore

    core = OpenDDECore()
    staged = core._stage_templates(query, tmp_path, {**core.config, "templates": {"m": cif_text}})
    assert staged is not None and Path(staged.templates_path).is_file()


def test_binder_only_input_accepts_its_own_msa(tmp_path) -> None:
    """A monomer fold takes the binder's alignment as its only chain's unpairedMsaPath."""
    binder = ">query\nCCCCCC\n>hom\nCCaCCCC\n"
    path = ProtenixCore()._write_input_json("CCCCCC", tmp_path, [binder])
    chains = json.loads(path.read_text())[0]["sequences"]
    assert len(chains) == 1
    assert Path(chains[0]["proteinChain"]["unpairedMsaPath"]).read_text() == binder


def test_unsupporting_models_refuse_msa_and_templates():
    class Unsupporting(ProtenixCore):
        SUPPORTS_USER_MSA = False
        SUPPORTS_USER_TEMPLATES = False

    model = Unsupporting.__new__(Unsupporting)
    model.config = {}
    for options in ({"msa": [">q\nAAAA\n"]}, {"templates": {"t": "data_x"}}):
        with pytest.raises(ValueError, match="does not support"):
            model._merge_options(options)


def test_msa_is_the_one_msa_option() -> None:
    """``msa`` is the only MSA option; the unreleased ``unpaired_msa`` alias is an unknown key."""
    core = ProtenixCore()
    a3m = [">q\nAAAA\n"]
    assert core._resolve_msa({**core.config, "msa": a3m}) == a3m
    assert core._resolve_msa(core.config) is None
    with pytest.raises(ValueError, match="unpaired_msa"):
        core._merge_options({"unpaired_msa": a3m})


def test_supplied_msa_is_refused_when_the_run_would_ignore_it() -> None:
    core = ProtenixCore()
    with pytest.raises(ValueError, match="needs use_msa=True"):
        core._resolve_msa({**core.config, "use_msa": False, "msa": [">q\nAAAA\n"]})


def test_msa_with_repeated_headers_is_validated_row_by_row(tmp_path) -> None:
    """Search tools repeat headers; a dict-backed parser would drop rows and miss a ragged one."""
    ragged = ">q\nAAAA\n>101\nAAAA\n>101\nAA\n"
    with pytest.raises(ValueError, match="aligned length"):
        ProtenixCore()._write_input_json("AAAA", tmp_path, [ragged])
