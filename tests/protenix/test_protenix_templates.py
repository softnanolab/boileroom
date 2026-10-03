"""Template staging and the monomer-with-MSA input, without importing Protenix."""

import io
import json
from pathlib import Path

import pytest
from biotite.structure.io.pdbx import CIFFile

from boileroom.models.protenix.core import ProtenixCore
from boileroom.models.protenix.templates import TEMPLATE_RELEASE_DATE, stage_templates, template_id

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

    hits = Path(staged["templates_path"]).read_text().splitlines()
    assert hits[0] == ">query" and hits[1] == query
    assert hits[2].startswith(f">{template_id(0)}_A/") and "mol:protein" in hits[2]
    aligned = "".join(c for c in hits[3] if not c.islower())
    assert len(aligned) == len(query)
    dates = json.loads(Path(staged["release_dates_path"]).read_text())
    assert dates == {template_id(0): {"release_date": TEMPLATE_RELEASE_DATE}}
    assert json.loads(Path(staged["obsolete_pdbs_path"]).read_text()) == {}
    # The staged structure carries the synthetic date, not whatever the caller's said.
    staged_cif = CIFFile.read(Path(staged["mmcif_dir"]) / f"{template_id(0)}.cif").block
    assert staged_cif["pdbx_audit_revision_history"]["revision_date"].as_array(str)[0] == TEMPLATE_RELEASE_DATE


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

    staged_cif = CIFFile.read(Path(staged["mmcif_dir"]) / f"{template_id(0)}.cif").block
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
    assert chains[0]["proteinChain"]["templatesPath"] == staged["templates_path"]
    assert "templatesPath" not in chains[1]["proteinChain"]


def test_no_templates_leaves_input_unchanged(tmp_path) -> None:
    core = ProtenixCore()
    assert core._stage_templates("AAAA:CCCC", tmp_path, {**core.config}) is None


def test_templates_chain_must_exist(tmp_path, cif_text) -> None:
    core = ProtenixCore()
    with pytest.raises(ValueError, match="templates_chain"):
        core._stage_templates(
            "AAAA:CCCC", tmp_path, {**core.config, "templates": {"m": cif_text}, "templates_chain": 2}
        )


def test_unsupporting_family_refuses_templates(tmp_path, cif_text) -> None:
    class NoTemplates(ProtenixCore):
        SUPPORTS_USER_TEMPLATES = False

    with pytest.raises(ValueError, match="does not support"):
        NoTemplates()._stage_templates("AAAA:CCCC", tmp_path, {**NoTemplates().config, "templates": {"m": cif_text}})


def test_opendde_stages_templates_like_protenix(tmp_path, cif_text, query) -> None:
    from boileroom.models.opendde.core import OpenDDECore

    core = OpenDDECore()
    staged = core._stage_templates(query, tmp_path, {**core.config, "templates": {"m": cif_text}})
    assert staged is not None and Path(staged["templates_path"]).is_file()


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


def test_msa_aliases_are_one_option() -> None:
    core = ProtenixCore()
    a3m = [">q\nAAAA\n"]
    assert core._resolve_msa({**core.config, "msa": a3m}) == a3m
    assert core._resolve_msa({**core.config, "unpaired_msa": a3m}) == a3m
    assert core._resolve_msa(core.config) is None
    with pytest.raises(ValueError, match="not both"):
        core._resolve_msa({**core.config, "msa": a3m, "unpaired_msa": a3m})


def test_supplied_msa_is_refused_when_the_run_would_ignore_it() -> None:
    core = ProtenixCore()
    with pytest.raises(ValueError, match="needs use_msa=True"):
        core._resolve_msa({**core.config, "use_msa": False, "msa": [">q\nAAAA\n"]})


def test_msa_with_repeated_headers_is_validated_row_by_row(tmp_path) -> None:
    """Search tools repeat headers; a dict-backed parser would drop rows and miss a ragged one."""
    ragged = ">q\nAAAA\n>101\nAAAA\n>101\nAA\n"
    with pytest.raises(ValueError, match="aligned length"):
        ProtenixCore()._write_input_json("AAAA", tmp_path, [ragged])
