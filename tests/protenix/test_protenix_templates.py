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
        core._stage_templates("AAAA:CCCC", tmp_path, {**core.config, "templates": {"m": cif_text}, "templates_chain": 2})


def test_opendde_refuses_templates(tmp_path, cif_text) -> None:
    from boileroom.models.opendde.core import OpenDDECore

    core = OpenDDECore()
    with pytest.raises(ValueError, match="does not support"):
        core._stage_templates("AAAA:CCCC", tmp_path, {**core.config, "templates": {"m": cif_text}})


def test_binder_only_input_accepts_its_own_msa(tmp_path) -> None:
    """A monomer fold takes the binder's alignment as its only chain's unpairedMsaPath."""
    binder = ">query\nCCCCCC\n>hom\nCCaCCCC\n"
    path = ProtenixCore()._write_input_json("CCCCCC", tmp_path, [binder])
    chains = json.loads(path.read_text())[0]["sequences"]
    assert len(chains) == 1
    assert Path(chains[0]["proteinChain"]["unpairedMsaPath"]).read_text() == binder


def test_unsupporting_models_refuse_msa_and_templates():
    from boileroom.models.esmfold2.core import ESMFold2Core
    from boileroom.models.opendde.core import OpenDDECore

    for cls in (OpenDDECore, ESMFold2Core):
        model = cls.__new__(cls)
        model.config = {}
        for options in ({"msa": [">q\nAAAA\n"]}, {"templates": {"t": "data_x"}}):
            with pytest.raises(ValueError, match="does not support"):
                model._merge_options(options)
