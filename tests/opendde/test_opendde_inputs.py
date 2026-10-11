"""OpenDDE integration test for caller-supplied MSAs and templates (GPU, weights required)."""

import io
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from biotite.sequence import ProteinSequence
from biotite.structure.io.pdbx import CIFFile

from boileroom import OpenDDE

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.xdist_group("opendde"),
]

CIF = Path(__file__).parents[1] / "data" / "chai" / "pred.rank_0.cif"
FAST = {"sample": 1, "cycle": 2, "step": 20}


def _template_sequence() -> str:
    block = CIFFile.read(io.StringIO(CIF.read_text())).block
    return "".join(ProteinSequence.convert_letter_3to1(m) for m in block["entity_poly_seq"]["mon_id"].as_array(str))


def test_opendde_uses_caller_supplied_msa_and_templates(
    backend_option: str, device_option: str | None, output_ctx: Callable[[], AbstractContextManager[Any]]
) -> None:
    """A caller MSA replaces the MSA search, and a caller template changes the prediction."""
    query = "MK" + _template_sequence()[3:-2] + "GGAA"
    options = {"include_fields": ["plddt", "cif"], "seeds": "7"}
    # An unreachable MSA server proves the supplied alignment is used: a search would fail the fold.
    config = {**FAST, "msa_server_url": "http://127.0.0.1:9"}

    with output_ctx(), OpenDDE(backend=backend_option, device=device_option, config=config) as model:
        with_msa = model.fold(query, options={**options, "msa": [f">query\n{query}\n"]})
        templated = model.fold(
            query,
            options={**options, "msa": [f">query\n{query}\n"], "templates": {"mine": CIF.read_text()}},
        )

    assert with_msa.seeds == [7] and templated.seeds == [7]
    assert np.isfinite(with_msa.plddt[0]).all() and np.isfinite(templated.plddt[0]).all()
    # Same seed and MSA; adding a template moves the structure.
    assert not np.allclose(with_msa.atom_array[0].coord, templated.atom_array[0].coord, atol=1e-2)


def test_opendde_rejects_malformed_inputs(backend_option: str, device_option: str | None) -> None:
    """Bad MSA and template inputs are refused with a clear error."""
    with OpenDDE(backend=backend_option, device=device_option, config=FAST) as model:
        with pytest.raises(ValueError, match="A3M"):
            model.fold("MKTAYIAKQR", options={"msa": [">query\nAAAA\n"]})
        with pytest.raises(ValueError, match="templates_chain"):
            model.fold("MKTAYIAKQR", options={"templates": {"x": CIF.read_text()}, "templates_chain": 3})


def test_opendde_folds_with_the_targets_own_structure(
    backend_option: str, device_option: str | None, output_ctx
) -> None:
    """A template of the target itself survives upstream's duplicate prefilter.

    The worker fails a request whose staged templates were not all featurized, so a successful fold means the
    template was used rather than silently dropped.
    """
    query = _template_sequence()
    options = {"include_fields": ["plddt"], "seeds": "7", "msa": [f">query\n{query}\n"]}
    with output_ctx(), OpenDDE(backend=backend_option, device=device_option, config=FAST) as model:
        templated = model.fold(query, options={**options, "templates": {"self": CIF.read_text()}})
        with pytest.raises(ValueError, match="same SEQRES"):
            model.fold(query, options={**options, "templates": {"a": CIF.read_text(), "b": CIF.read_text()}})
    assert np.isfinite(templated.plddt[0]).all()
