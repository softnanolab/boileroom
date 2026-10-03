"""MSA input validation, per-chain A3M support and transport for AlphaFold2-Multimer."""

import json
from pathlib import Path

import pytest

from boileroom.inputs import MSAInput
from boileroom.models.alphafold.msa import encode_msa_option, materialize_msa

HETERO = ["AAAA", "CCC"]
HOMO = ["AAAA", "AAAA"]


def _a3m(*rows: str) -> str:
    return "".join(f">r{index}\n{row}\n" for index, row in enumerate(rows))


# -- (b) validation against the requested sequence ------------------------------


def test_msa_input_single_chain_passes_and_strips_insertions() -> None:
    text = materialize_msa(MSAInput(sequences=["AAAA", "AaAAA"], remove_insertions=True), ["AAAA"])

    assert text == ">seq_0\nAAAA\n>seq_1\nAAAA\n"


def test_msa_input_single_chain_query_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="first MSA row must be the requested sequence"):
        materialize_msa(MSAInput(sequences=["AAAG", "AAAA"]), ["AAAA"])


def test_msa_input_row_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="aligned columns"):
        materialize_msa(MSAInput(sequences=["AAAA", "AAA"]), ["AAAA"])


def test_msa_input_complex_query_mismatch_rejected_per_chain() -> None:
    with pytest.raises(ValueError, match="chain 1 differs"):
        materialize_msa(MSAInput(sequences=["AAAA:CCD"]), HETERO)


def test_msa_input_complex_segment_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="chain 1 has 2 aligned columns"):
        materialize_msa(MSAInput(sequences=["AAAA:CCC", "AAAA:CC"]), HETERO)


def test_msa_input_empty_rows_rejected() -> None:
    with pytest.raises(ValueError, match="no sequences"):
        materialize_msa(MSAInput(sequences=[]), ["AAAA"])


def test_msa_file_query_mismatch_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wrong.a3m"
    path.write_text(">query\nAAAG\n", encoding="utf-8")

    with pytest.raises(ValueError, match="first MSA row must be the requested sequence"):
        materialize_msa(MSAInput(path=path), ["AAAA"])


def test_msa_file_complex_header_must_match_chains(tmp_path: Path) -> None:
    path = tmp_path / "complex.a3m"
    path.write_text("#4,4\t1,1\n>101\t102\nAAAACCCC\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the requested chains"):
        materialize_msa(MSAInput(path=path), HETERO)
    assert materialize_msa(MSAInput(path=path), ["AAAA", "CCCC"]).startswith("#4,4\t1,1\n")


def test_msa_file_complex_query_mismatch_rejected(tmp_path: Path) -> None:
    path = tmp_path / "complex.a3m"
    path.write_text("#4,3\t1,1\n>101\t102\nAAAACCD\n", encoding="utf-8")

    with pytest.raises(ValueError, match="first MSA row must be the requested sequence"):
        materialize_msa(MSAInput(path=path), HETERO)


def test_plain_a3m_file_rejected_for_multichain(tmp_path: Path) -> None:
    """ColabFold would fold a headerless a3m as one chain, so it is refused for complexes."""
    path = tmp_path / "plain.a3m"
    path.write_text(">query\nAAAACCC\n", encoding="utf-8")

    with pytest.raises(ValueError, match="complex a3m"):
        materialize_msa(MSAInput(path=path), HETERO)


def test_non_a3m_text_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("AAAA\n", encoding="utf-8")

    with pytest.raises(ValueError, match="a3m/FASTA"):
        materialize_msa(MSAInput(path=path), ["AAAA"])


def test_paired_rows_of_a_heteromer_also_get_unpaired_rows() -> None:
    """ColabFold rejects a chain whose unpaired MSA is empty, so paired-only input needs both sections."""
    text = materialize_msa(MSAInput(sequences=["AAAA:CCC", "AAAG:CCD", "AAAA:CCD"]), HETERO)

    assert text.splitlines() == [
        "#4,3\t1,1",
        ">101\t102",
        "AAAACCC",
        ">seq_0",
        "AAAACCC",
        ">seq_1",
        "AAAGCCD",
        ">seq_2",
        "AAAACCD",
        ">101",
        "AAAA---",
        ">101",
        "AAAG---",
        ">102",
        "----CCC",
        ">102",
        "----CCD",
    ]


# -- (a) per-chain A3M text list -----------------------------------------------


def test_per_chain_single_chain_passes_through() -> None:
    assert materialize_msa([_a3m("AAAA", "AaAAA")], ["AAAA"]) == ">r0\nAAAA\n>r1\nAaAAA\n"


def test_per_chain_complex_builds_gap_padded_unpaired_complex_a3m() -> None:
    text = materialize_msa([_a3m("AAAA", "AAAG"), _a3m("CCC", "CCD", "CDD")], HETERO)

    assert text.splitlines() == [
        "#4,3\t1,1",
        ">101\t102",
        "AAAACCC",
        ">101",
        "AAAA---",
        ">r1",
        "AAAG---",
        ">102",
        "----CCC",
        ">r1",
        "----CCD",
        ">r2",
        "----CDD",
    ]


def test_per_chain_none_entry_is_single_sequence_for_that_chain() -> None:
    text = materialize_msa([_a3m("AAAA", "AAAG"), None], HETERO)

    assert text.splitlines()[-2:] == [">102", "----CCC"]
    assert ">r1" in text


def test_per_chain_homodimer_uses_one_msa_with_cardinality() -> None:
    msa = _a3m("AAAA", "AAAG")
    text = materialize_msa([msa, msa], HOMO)

    assert text.startswith("#4\t2\n>101\nAAAA\n")
    assert text.count(">r1") == 1


def test_per_chain_homodimer_with_one_none_uses_the_other() -> None:
    assert materialize_msa([None, _a3m("AAAA", "AAAG")], HOMO).startswith("#4\t2\n")


@pytest.mark.parametrize(
    ("msas", "chains", "match"),
    [
        ([_a3m("AAAA")], HETERO, "one entry per chain"),
        ([None, None], HETERO, "no A3M text for any chain"),
        ([_a3m("AAAG"), _a3m("CCC")], HETERO, "entry 0.*first MSA row"),
        ([_a3m("AAAA"), _a3m("CCC", "CC")], HETERO, "entry 1.*aligned columns"),
        (["AAAA", _a3m("CCC")], HETERO, "A3M text"),
        ([42, _a3m("CCC")], HETERO, "options\\['msa'\\] must be an MSAInput or a list"),
        ([_a3m("AAAA", "AAAG"), _a3m("AAAA", "AAAC")], HOMO, "identical chains must share one MSA"),
    ],
)
def test_per_chain_failures(msas: list, chains: list[str], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        materialize_msa(msas, chains)


def test_non_msa_option_type_rejected() -> None:
    with pytest.raises(ValueError, match="MSAInput or a list"):
        materialize_msa("AAAA", ["AAAA"])


# -- (c) JSON-safe transport ---------------------------------------------------


def test_encode_msa_input_sequences_is_json_safe_and_round_trips() -> None:
    encoded = encode_msa_option(MSAInput(sequences=["AAAA:CCC", "AAAG:CCD"], remove_insertions=True))

    assert json.loads(json.dumps(encoded)) == encoded
    assert materialize_msa(encoded, HETERO) == materialize_msa(
        MSAInput(sequences=["AAAA:CCC", "AAAG:CCD"], remove_insertions=True), HETERO
    )


def test_encode_msa_input_path_reads_file_on_caller_side(tmp_path: Path) -> None:
    path = tmp_path / "m.a3m"
    path.write_text(">q\nAAAA\n>h\nAaAAA\n", encoding="utf-8")
    encoded = encode_msa_option(MSAInput(path=path, remove_insertions=True))
    path.unlink()  # the consumer (container) cannot see the local file

    assert json.loads(json.dumps(encoded)) == encoded
    assert materialize_msa(encoded, ["AAAA"]) == ">q\nAAAA\n>h\nAAAA\n"


def test_encode_msa_list_round_trips_and_none_passes() -> None:
    assert encode_msa_option(None) is None
    assert encode_msa_option((_a3m("AAAA"), None)) == [_a3m("AAAA"), None]


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "msa_input"},
        {"kind": "msa_input", "sequences": ["A"], "a3m": ">q\nA\n"},
        {"kind": "msa_input", "sequences": "A"},
        {"kind": "msa_input", "sequences": ["A"], "remove_insertions": "yes"},
        {"kind": "msa_input", "a3m": "AAAA"},
    ],
)
def test_malformed_encoded_payload_rejected(payload: dict) -> None:
    with pytest.raises(ValueError):
        materialize_msa(payload, ["A"])


def test_wrapper_encodes_msa_for_backend_and_validates_early(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from boileroom.models.alphafold.alphafold2_multimer import AlphaFold2Multimer

    wrapper = AlphaFold2Multimer.__new__(AlphaFold2Multimer)
    calls: list[tuple] = []
    monkeypatch.setattr(wrapper, "_call_backend_method", lambda *args, **kwargs: calls.append((args, kwargs)))
    path = tmp_path / "m.a3m"
    path.write_text("#4,3\t1,1\n>101\t102\nAAAACCC\n", encoding="utf-8")

    wrapper.fold("AAAA:CCC", options={"msa": MSAInput(path=path), "random_seed": 3})
    wrapper.fold("AAAA:CCC", options={"msa": [_a3m("AAAA"), None]})

    sent = calls[0][1]["options"]
    assert json.loads(json.dumps(sent)) == sent
    assert sent["random_seed"] == 3 and sent["msa"]["a3m"].startswith("#4,3")
    assert calls[1][1]["options"]["msa"] == [_a3m("AAAA"), None]

    with pytest.raises(ValueError, match="first MSA row"):
        wrapper.fold("AAAA:CCC", options={"msa": MSAInput(sequences=["AAAG:CCC"])})
    with pytest.raises(ValueError, match="MSAInput or a list"):
        wrapper.fold("AAAA:CCC", options={"msa": "AAAA"})
    assert len(calls) == 2


def test_core_fold_rejects_mismatched_msa_before_loading_model(tmp_path: Path) -> None:
    from boileroom.models.alphafold.core import AlphaFold2MultimerCore

    core = AlphaFold2MultimerCore({"data_dir": str(tmp_path)})

    with pytest.raises(ValueError, match="first MSA row"):
        core.fold("AAAA", options={"msa": MSAInput(sequences=["AAAG"])})
    assert not core.ready


def test_core_resolves_encoded_and_per_chain_msa(tmp_path: Path) -> None:
    from boileroom.models.alphafold.core import AlphaFold2MultimerCore

    core = AlphaFold2MultimerCore({"data_dir": str(tmp_path)})
    for msa in (encode_msa_option(MSAInput(sequences=["AAAA:CCC"])), [_a3m("AAAA"), _a3m("CCC")]):
        input_path, msa_mode, cache_key = core._resolve_msa_input("AAAA:CCC", HETERO, msa, tmp_path, core.config)

        assert input_path.read_text(encoding="utf-8").startswith("#4,3\t1,1\n")
        assert msa_mode is None and cache_key is None
