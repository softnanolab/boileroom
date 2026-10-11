"""Offline MiniFold input, output, lifecycle, and download regressions."""

import hashlib
from contextlib import contextmanager
from io import StringIO
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

pytestmark = pytest.mark.contract


@pytest.fixture
def core(monkeypatch):
    """Build a CPU fake without importing MiniFold or acquiring weights."""
    torch = pytest.importorskip("torch")
    from boileroom.models.minifold.core import MiniFoldCore

    constants: Any = ModuleType("minifold.utils.residue_constants")
    constants.atom_types = ["N", "CA", "C", "CB", "O"] + [f"C{i}" for i in range(32)]
    constants.restype_1to3 = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU"}
    monkeypatch.setitem(__import__("sys").modules, constants.__name__, constants)
    core = MiniFoldCore({"device": "cpu"})
    core._device = torch.device("cpu")
    core.ready = True
    seen = []

    def prepare(sequence):
        seen.append(sequence)
        return {"seq": torch.zeros((1, len(sequence)), dtype=torch.long)}

    def predict(batch, num_recycling):
        length = batch["seq"].shape[1]
        mask = torch.zeros((1, length, 37))
        mask[:, :, :5] = 1
        return {
            "final_atom_positions": torch.arange(length * 37 * 3).reshape(1, length, 37, 3),
            "final_atom_mask": mask,
            "plddt": torch.full((1, length), 50.0),
        }

    core.model = Mock(side_effect=predict)
    core.alphabet = object()
    monkeypatch.setattr(core, "_prepare_input", prepare)
    monkeypatch.setattr(core, "seen", seen, raising=False)
    return core


def test_batches_are_unpadded_ordered_and_use_one_resident_model(core):
    first = core.fold(["ACDE", "AC"], {"include_fields": ["*"]})
    model = core.model
    second = core.fold("ACD", {"include_fields": ["plddt"]})
    assert core.model is model
    assert core.seen == ["ACDE", "AC", "ACD"]
    assert [call.args[0]["seq"].shape for call in model.call_args_list] == [(1, 4), (1, 2), (1, 3)]
    assert first.metadata is not second.metadata
    assert first.metadata.sequence_lengths == [4, 2]
    assert second.metadata.sequence_lengths == [3]
    assert len(first.atom_array) == 2
    assert first.plddt[0].shape == (4,)
    np.testing.assert_array_equal(first.residue_index, [[0, 1, 2, 3], [0, 1, -1, -1]])
    np.testing.assert_array_equal(first.chain_index, [[0, 0, 0, 0], [0, 0, -1, -1]])
    np.testing.assert_allclose(first.plddt[0], 0.5)
    np.testing.assert_allclose(first.atom_array[0].b_factor, 50.0)
    assert first.metadata.inference_time >= 0
    assert first.metadata.runtime["minifold.batch_mode"] == "sequential-unpadded"
    assert first.pae is None
    core.close()
    assert not core.ready and core.model is None and core.alphabet is None


@pytest.mark.parametrize("fields", [None, [], ["plddt"], ["pdb"], ["cif"], ["*"]])
def test_output_filtering_and_signed_transport(core, fields):
    from boileroom.backend.transport import deserialize_transport_payload, serialize_transport_payload

    result = core.fold("ACD", {"include_fields": fields})
    result = deserialize_transport_payload(serialize_transport_payload(result, "test-secret"), "test-secret")
    assert result.metadata.sequence_lengths == [3]
    assert len(result.atom_array) == 1
    assert (result.plddt is not None) == bool(fields and ("plddt" in fields or "*" in fields))
    assert (result.pdb is not None) == bool(fields and ("pdb" in fields or "*" in fields))
    assert (result.cif is not None) == bool(fields and ("cif" in fields or "*" in fields))


def test_structural_numbering_and_confidence_agree_across_formats(core):
    from biotite.structure.io import pdb, pdbx

    result = core.fold("ACD", {"include_fields": ["*"]})
    expected = result.atom_array[0]
    pdb_atoms = pdb.get_structure(pdb.PDBFile.read(StringIO(result.pdb[0])), model=1, extra_fields=["b_factor"])
    cif_atoms = pdbx.get_structure(pdbx.CIFFile.read(StringIO(result.cif[0])), model=1, extra_fields=["b_factor"])
    for atoms in (pdb_atoms, cif_atoms):
        np.testing.assert_array_equal(atoms.res_id, expected.res_id)
        np.testing.assert_allclose(atoms.coord, expected.coord)
        np.testing.assert_allclose(atoms.b_factor, 50.0)
    assert min(expected.res_id) == 1
    assert max(expected.res_id) == 3


@pytest.mark.parametrize("sequences", [[], "", ["AC", ""], "AC:DE", "AC:DE:FG", [None], "AC?D", "acd", "A" * 2049])
def test_invalid_input_refused_before_load(core, monkeypatch, sequences):
    loader = Mock()
    monkeypatch.setattr(core, "_load", loader)
    with pytest.raises((ValueError, TypeError)):
        core.fold(sequences)
    loader.assert_not_called()
    core.model.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [
        {"model_size": "6L"},
        {"num_recycling": -1},
        {"num_recycling": True},
        {"max_sequence_length": 0},
        {"max_sequence_length": 2.5},
        {"compile": "yes"},
        {"kernels": True},
        {"glycine_linker": "GG"},
        {"msa": [">a\nAC"]},
        {"templates": {"x": "data_x"}},
        {"bogus": True},
    ],
)
def test_invalid_config_refused_without_model_dependencies(config):
    from boileroom.models.minifold.core import MiniFoldCore

    with pytest.raises(ValueError):
        MiniFoldCore(config)


def test_static_options_and_nonkit_mode_refused_before_load(core, monkeypatch):
    from boileroom.models.minifold.core import MiniFoldCore
    from boileroom.optimization import OptimizationUnavailableError

    monkeypatch.setattr(core, "_load", Mock())
    for key in core.STATIC_CONFIG_KEYS:
        with pytest.raises(ValueError, match="initialization"):
            core.fold("AC", {key: core.config[key]})
    with pytest.raises(OptimizationUnavailableError):
        MiniFoldCore({"optimization": "exact"})
    core._load.assert_not_called()


def test_nonfinite_output_refused(core):
    torch = pytest.importorskip("torch")
    result = core.model({"seq": torch.zeros((1, 2))}, 1)
    result["plddt"][0, 0] = float("nan")
    core.model = Mock(return_value=result)
    with pytest.raises(RuntimeError, match="non-finite"):
        core.fold("AC")


def test_cpu_autocast_does_not_request_cuda_or_bfloat16(core, monkeypatch):
    import torch

    autocast = Mock(side_effect=AssertionError("autocast unexpectedly enabled"))
    monkeypatch.setattr(torch, "autocast", autocast)
    core.fold("AC")
    autocast.assert_not_called()


def test_checkpoint_download_is_pinned_atomic_verified_and_reused(tmp_path, monkeypatch):
    from boileroom.models.minifold import loading

    payload = b"verified checkpoint fixture"
    monkeypatch.setitem(loading.CHECKPOINT_SHA256, "12L", hashlib.sha256(payload).hexdigest())
    response = Mock()
    response.iter_bytes.return_value = [payload[:5], payload[5:]]

    @contextmanager
    def stream(method, url, **kwargs):
        assert method == "GET" and loading.CHECKPOINT_REVISION in url
        assert kwargs["timeout"].connect == 30.0 and kwargs["timeout"].read == 120.0
        yield response

    mocked = Mock(side_effect=stream)
    monkeypatch.setattr(loading.httpx, "stream", mocked)
    path = loading.checkpoint_path(tmp_path, "12L")
    assert path.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [path]
    assert loading.checkpoint_path(tmp_path, "12L") == path
    assert mocked.call_count == 1
    path.write_bytes(b"corrupted")
    response.iter_bytes.return_value = [payload]
    assert loading.checkpoint_path(tmp_path, "12L").read_bytes() == payload
    assert mocked.call_count == 2


@pytest.mark.parametrize("failure", ["http", "checksum", "interrupted"])
def test_failed_download_leaves_no_partial_file(tmp_path, monkeypatch, failure):
    from boileroom.models.minifold import loading

    @contextmanager
    def stream(*args, **kwargs):
        response = Mock()
        if failure == "http":
            response.raise_for_status.side_effect = RuntimeError("HTTP failure")
        if failure == "interrupted":

            def interrupted(**kwargs):
                yield b"partial"
                raise RuntimeError("connection lost")

            response.iter_bytes.side_effect = interrupted
        else:
            response.iter_bytes.return_value = [b"bad checksum"]
        yield response

    monkeypatch.setattr(loading.httpx, "stream", stream)
    with pytest.raises(RuntimeError):
        loading.checkpoint_path(tmp_path, "12L")
    assert list(tmp_path.iterdir()) == []


def test_checkpoint_rejects_missing_learned_parameters():
    from boileroom.models.minifold.loading import load_state_dict

    model = Mock()
    model.named_buffers.return_value = [("lm.layers.0.self_attn.rot_emb.inv_freq", object())]
    model.named_parameters.return_value = [
        ("lm.layers.0.weight", SimpleNamespace(requires_grad=False)),
        ("lm.layers.34.weight", SimpleNamespace(requires_grad=True)),
    ]
    model.load_state_dict.return_value = SimpleNamespace(
        missing_keys=["lm.layers.0.weight", "lm.layers.0.self_attn.rot_emb.inv_freq"], unexpected_keys=[]
    )
    load_state_dict(model, {"model._orig_mod.fold.weight": 1, "loss.boundaries": 2})
    model.load_state_dict.assert_called_once_with({"fold.weight": 1}, strict=False)
    model.load_state_dict.return_value = SimpleNamespace(missing_keys=["fold.weight"], unexpected_keys=[])
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        load_state_dict(model, {})
    model.load_state_dict.return_value = SimpleNamespace(missing_keys=["lm.layers.34.weight"], unexpected_keys=[])
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        load_state_dict(model, {})
    model.load_state_dict.return_value = SimpleNamespace(
        missing_keys=["lm.layers.34.self_attn.rot_emb.inv_freq"], unexpected_keys=[]
    )
    with pytest.raises(RuntimeError, match="checkpoint mismatch"):
        load_state_dict(model, {})


@pytest.fixture
def loader_case(monkeypatch, tmp_path):
    """Exercise the real loader control flow without allocating model weights."""
    import sys

    import torch

    from boileroom.models.minifold import core as module

    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.lm.alphabet = object()
    model.named_parameters.return_value = []
    model.named_buffers.return_value = []
    model.load_state_dict.return_value = SimpleNamespace(missing_keys=[], unexpected_keys=[])
    factory = Mock(return_value=model)
    config_module: Any = ModuleType("minifold.data.config")
    config_module.model_config = Mock(return_value=SimpleNamespace(data={}))
    model_module: Any = ModuleType("minifold.model.model")
    model_module.MiniFoldModel = factory
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    checkpoint = {
        "hyper_parameters": {"esm_model_name": module.ESM_MODEL_NAME, "num_blocks": 12, "no_bins": 64},
        "state_dict": {"model.fold.weight": 1},
    }
    monkeypatch.setattr(module, "checkpoint_path", Mock(return_value=tmp_path / "checkpoint"))
    safe_load = Mock(return_value=checkpoint)
    monkeypatch.setattr(torch, "load", safe_load)
    instance = module.MiniFoldCore({"device": "cpu", "cache_dir": str(tmp_path)})
    return instance, model, factory, safe_load


def test_loader_is_resident_safe_and_reuses_esm_alphabet(loader_case):
    import torch

    instance, model, factory, safe_load = loader_case
    original_cache = torch.hub.get_dir()
    instance._load()
    instance._load()
    factory.assert_called_once()
    assert safe_load.call_args.kwargs == {"map_location": "cpu", "weights_only": True}
    assert instance.alphabet is model.lm.alphabet
    assert torch.hub.get_dir() == original_cache
    assert instance.ready
    instance.close()
    instance._load()
    assert factory.call_count == 2


@pytest.mark.parametrize("where", ["constructor", "state_dict"])
def test_failed_loader_restores_cache_and_keeps_core_unready(loader_case, where):
    import torch

    instance, model, factory, safe_load = loader_case
    original_cache = torch.hub.get_dir()
    (factory if where == "constructor" else model.load_state_dict).side_effect = RuntimeError("load failed")
    with pytest.raises(RuntimeError, match="load failed"):
        instance._load()
    assert torch.hub.get_dir() == original_cache
    assert instance.model is None and not instance.ready


def test_unknown_backbone_is_refused_before_model_construction(loader_case):
    instance, model, factory, safe_load = loader_case
    safe_load.return_value["hyper_parameters"]["esm_model_name"] = "../../untrusted"
    with pytest.raises(ValueError, match="unrecognized ESM backbone"):
        instance._load()
    factory.assert_not_called()


def test_esm_loader_is_allowlisted_verified_and_weights_only(tmp_path, monkeypatch):
    import argparse

    import torch

    from boileroom.models.minifold import loading

    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    before = torch.serialization.get_safe_globals().copy()
    observed = []

    def download(url, destination, checksum):
        observed.append((url, checksum))
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": argparse.Namespace(encoder_layers=36), "model": {"x": torch.zeros(1)}}, destination)
        return destination

    monkeypatch.setattr(loading, "_download_verified", download)
    url = next(iter(loading.ESM_ARTIFACTS))
    result = loading.load_esm_artifact(url)
    assert result["cfg"].encoder_layers == 36
    assert observed == [(url, loading.ESM_ARTIFACTS[url])]
    assert torch.serialization.get_safe_globals() == before
    with pytest.raises(ValueError, match="Unrecognized"):
        loading.load_esm_artifact("https://example.com/model.pt")


def test_upstream_patch_is_checked_and_preserves_non_kernel_math(tmp_path):
    from boileroom.models.minifold.patch_upstream import patch

    (tmp_path / "minifold/model").mkdir(parents=True)
    (tmp_path / "minifold/utils").mkdir()
    (tmp_path / "pyproject.toml").write_text('[tool.setuptools]\npackages = ["minifold"]\n')
    model = tmp_path / "minifold/model/miniformer.py"
    model.write_text(
        "from minifold.model.kernels.gating import gating_kernel\n"
        "from minifold.model.kernels.mlp import mlp_kernel\n"
        "def ordinary(x):\n    return x + 1\n"
        "def mlp_kernel_func(x, w1, w2, b1, b2, wn, bn):\n"
        "    return mlp_kernel(x, w1, w2, b1, b2, wn, bn)\n"
        "def triangular_kernel_func(x, gi_w, pi_w, gi_b, pi_b, ni_w, ni_b):\n"
        "    x = gating_kernel(x, gi_w, pi_w, gi_b, pi_b, ni_w, ni_b)\n"
    )
    esm = tmp_path / "minifold/utils/esm.py"
    esm.write_text(
        "def load_hub_workaround(url):\n    torch.hub.load_state_dict_from_url(url)\n    torch.load(url)\n\ndef load_regression_hub(name):\n    return name\n"
    )
    patch(tmp_path)
    assert "def ordinary(x):\n    return x + 1" in model.read_text()
    assert "\nfrom minifold.model.kernels" not in model.read_text()
    assert "    from minifold.model.kernels.mlp" in model.read_text()
    assert "torch.load" not in esm.read_text()
    assert "return load_esm_artifact(url)" in esm.read_text()
    assert 'find = {include = ["minifold*"]}' in (tmp_path / "pyproject.toml").read_text()
    with pytest.raises(ValueError, match="packaging"):
        patch(tmp_path)
