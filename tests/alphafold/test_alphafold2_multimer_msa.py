"""MSA input validation, per-chain A3M support and transport for AlphaFold2-Multimer."""

import json
from pathlib import Path

import pytest

from boileroom.inputs import MSAInput, a3m_rows, aligned_columns, parse_a3m
from boileroom.models.alphafold.msa import encode_msa_option, materialize_msa

HETERO = ["AAAA", "CCC"]
HOMO = ["AAAA", "AAAA"]


def _a3m(*rows: str) -> str:
    return "".join(f">r{index}\n{row}\n" for index, row in enumerate(rows))


def _file(tmp_path: Path, text: str) -> MSAInput:
    path = tmp_path / "given.a3m"
    path.write_text(text, encoding="utf-8")
    return MSAInput(path=path)


def _colabfold_split(text: str) -> tuple[list[str], list[list[str]]]:
    """Split a complex a3m the way ColabFold's ``unserialize_msa`` does.

    Every non-lowercase character counts as an aligned column, and a row is cut
    into chains after ``length`` such columns. Returns the query slices and each
    hit row's per-chain segments.
    """
    header, _, body = text.partition("\n")
    lengths = [int(part) for part in header[1:].split("\t")[0].split(",")]
    rows = [row for _, row in parse_a3m(body)]

    def split(row: str) -> list[str]:
        segments: list[str] = []
        start, count = 0, 0
        for position, char in enumerate(row):
            if not char.islower():
                count += 1
                if count == lengths[len(segments)]:
                    segments.append(row[start : position + 1])
                    start, count = position + 1, 0
                    if len(segments) == len(lengths):
                        break
        return segments

    query = rows[0]
    offsets = [sum(lengths[:index]) for index in range(len(lengths) + 1)]
    return [query[offsets[i] : offsets[i + 1]] for i in range(len(lengths))], [split(row) for row in rows[1:]]


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


# -- first row as ColabFold folds it, '.' and invalid characters ----------------


@pytest.mark.parametrize(
    ("msa", "chains"),
    [
        (MSAInput(sequences=["AAaAA", "AAAA"]), ["AAAA"]),
        (MSAInput(sequences=["AAaAA:CCC", "AAAA:CCC"]), HETERO),
        ([_a3m("AAaAA", "AAAG"), None], HETERO),
        ([_a3m("AAaAA", "AAAG")], ["AAAA"]),
    ],
    ids=["rows-single", "rows-complex", "per-chain", "per-chain-single"],
)
def test_first_row_with_insertions_rejected(msa: object, chains: list[str]) -> None:
    """ColabFold folds the raw first row, so an insertion in it would change the folded sequence."""
    with pytest.raises(ValueError, match="must not contain insertions"):
        materialize_msa(msa, chains)


def test_complex_file_query_line_with_insertion_rejected(tmp_path: Path) -> None:
    """ColabFold slices the raw query line by the header lengths, so 'AAAaACCCC' would fold AAAa + ACCC."""
    with pytest.raises(ValueError, match="must not contain insertions"):
        materialize_msa(_file(tmp_path, "#4,4\t1,1\n>101\t102\nAAAaACCCC\n>h\nAAAACCCC\n"), ["AAAA", "CCCC"])


def test_first_row_insertions_are_fine_once_removed(tmp_path: Path) -> None:
    msa = _file(tmp_path, "#4,4\t1,1\n>101\t102\nAAAaACCCC\n>h\nAAAACCCC\n")
    text = materialize_msa(MSAInput(path=msa.path, remove_insertions=True), ["AAAA", "CCCC"])

    assert _colabfold_split(text)[0] == ["AAAA", "CCCC"]


@pytest.mark.parametrize(
    ("msa", "chains"),
    [
        (MSAInput(sequences=["AAAA:CDEF", "--.--:CDEF"]), ["AAAA", "CDEF"]),
        (MSAInput(sequences=["AAAA:CDEF", "AA.AA:CDEF"]), ["AAAA", "CDEF"]),
        ([">q\nAAAA\n>h\n-.--W\n", None], ["AAAA", "CDEF"]),
        (MSAInput(sequences=["AAAA", "AA.AA"]), ["AAAA"]),
        ([">q\nAAAA\n>h\nAA.AA\n"], ["AAAA"]),
    ],
    ids=["rows-unpaired-gap", "rows-paired", "per-chain", "rows-single", "per-chain-single"],
)
def test_dot_in_rows_rejected(msa: object, chains: list[str]) -> None:
    """'.' is an aligned column to ColabFold's chain split, so it would shift residues across chains."""
    with pytest.raises(ValueError, match="contains '\\.'") as excinfo:
        materialize_msa(msa, chains)

    # Only MSAInput has remove_insertions; pointing list users there would swap unpaired-only for paired rows.
    if isinstance(msa, list):
        assert "remove_insertions=True" not in str(excinfo.value)
        assert "per-chain list form has no remove_insertions option" in str(excinfo.value)
    else:
        assert "MSAInput(..., remove_insertions=True)" in str(excinfo.value)


def test_dot_in_complex_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="contains '\\.'"):
        materialize_msa(_file(tmp_path, "#4,4\t1,1\n>101\t102\nAAAACDEF\n>h\n--.--CDEF\n"), ["AAAA", "CDEF"])


@pytest.mark.parametrize("dotted", ["--.--:CDEF", "AA.AA:CDeEF"])
def test_dot_removed_with_insertions_keeps_chain_boundaries(dotted: str) -> None:
    text = materialize_msa(MSAInput(sequences=["AAAA:CDEF", dotted], remove_insertions=True), ["AAAA", "CDEF"])

    assert "." not in text
    query, hits = _colabfold_split(text)
    assert query == ["AAAA", "CDEF"]
    assert {segment for _, segment in hits} == {"CDEF", "----"}
    assert {segment for segment, _ in hits} <= {"AAAA", "----"}


@pytest.mark.parametrize(
    ("text", "bad"),
    [
        (">q\nAAAA\n# comment\n>h\nAAAG\n", "#"),
        (">q\nAAAA\n>h\nAA AG\n", " "),
        (">q\nAAAA\n>h\nAA\tAG\n", "\t"),
        (">q\nAAAA*\n>h\nAAAG\n", "*"),
        (">q\nAAAA\n>h\nAA1G\n", "1"),
    ],
    ids=["comment-line", "space", "tab", "star", "digit"],
)
def test_invalid_characters_rejected(tmp_path: Path, text: str, bad: str) -> None:
    with pytest.raises(ValueError, match="invalid characters") as excinfo:
        materialize_msa(_file(tmp_path, text), ["AAAA"])
    assert repr(bad) in str(excinfo.value)


def test_comment_line_inside_complex_file_rejected(tmp_path: Path) -> None:
    """ColabFold's normalize_a3m keeps only the first '#' line; a later one would join a sequence row."""
    with pytest.raises(ValueError, match="invalid characters.*'#'"):
        materialize_msa(_file(tmp_path, "#4,3\t1,1\n>101\t102\nAAAACCC\n#note\n>h\nAAAGCCD\n"), HETERO)


@pytest.mark.parametrize("header", ["#4,3\t1,1\t9", "#4,3", "#4,x\t1,1"])
def test_complex_header_must_have_two_integer_fields(tmp_path: Path, header: str) -> None:
    with pytest.raises(ValueError, match="'#<lengths>"):
        materialize_msa(_file(tmp_path, f"{header}\n>101\t102\nAAAACCC\n"), HETERO)


def test_complex_file_is_re_rendered_from_validated_rows(tmp_path: Path) -> None:
    """Wrapped rows, blank lines and padding are normalised; headers and row order are kept."""
    text = materialize_msa(
        _file(tmp_path, "\n#4,3\t1,1  \n>101\t102\nAAAA\nCCC\n\n>h1\n  AAAG\nC-D\n>h1\nAAAGC-D\n"), HETERO
    )

    assert text == "#4,3\t1,1\n>101\t102\nAAAACCC\n>h1\nAAAGC-D\n>h1\nAAAGC-D\n"
    assert _colabfold_split(text) == (["AAAA", "CCC"], [["AAAG", "C-D"], ["AAAG", "C-D"]])


# -- repeated chains (one segment per unique chain in ColabFold's format) -------


@pytest.mark.parametrize(
    ("rows", "chains"),
    [
        (["AAAA:AAAA:CCC", "AAAG:ADDD:CCD"], ["AAAA", "AAAA", "CCC"]),
        (["AAAA:AAAA:CCC", "----:AKKA:---"], ["AAAA", "AAAA", "CCC"]),
        (["AAAA:AAAA", "AAAG:----"], HOMO),
        (["AAAA:CCC:AAAA", "AAAG:CCD:AADA"], ["AAAA", "CCC", "AAAA"]),
    ],
)
def test_differing_segments_for_copies_of_a_chain_rejected(rows: list[str], chains: list[str]) -> None:
    """The complex a3m keeps one segment per unique chain, so a copy-specific segment would be dropped."""
    with pytest.raises(ValueError, match="copies of one sequence but carry different segments"):
        materialize_msa(MSAInput(sequences=rows), chains)


def test_identical_segments_for_copies_of_a_chain_accepted() -> None:
    text = materialize_msa(MSAInput(sequences=["AAAA:CCC:AAAA", "AAAG:CCD:AAAG"]), ["AAAA", "CCC", "AAAA"])

    assert text.startswith("#4,3\t2,1\n>101\t102\nAAAACCC\n>seq_0\nAAAACCC\n>seq_1\nAAAGCCD\n")


def test_copy_segments_compared_after_removing_insertions() -> None:
    msa = MSAInput(sequences=["AAAA:AAAA", "AAaAG:AAAG"], remove_insertions=True)

    assert materialize_msa(msa, HOMO) == "#4\t2\n>101\nAAAA\n>seq_0\nAAAA\n>seq_1\nAAAG\n"


# -- one shared A3M parser -------------------------------------------------------

Q = "MKTAYIAK"


@pytest.mark.parametrize(
    ("text", "af2_only_error"),
    [
        (f">q\n{Q}\n>h1\nMKT-YIAK\n", None),
        (">q\nMKTA\nYIAK\n>h1\nMKT-\nYIAK\n", None),
        (f">q\n{Q}\n>h\nMKabT-YIAK\n>h\nMKT-YIAK\n", None),
        (f">q\n{Q}\n>h1\nMKT-YI.AK\n", "contains '\\.'"),
        (">q\nMKTaAYIAK\n>h1\nMKT-YIAK\n", "must not contain insertions"),
        (f">q\n{Q}\n# comment\n>h1\nMKT-YIAK\n", "*"),
        (f"#A3M#\n>q\n{Q}\n", "*"),
        (">q\nMKTA YIAK\n>h1\nMKT-YIAK\n", "*"),
        (f">q\n{Q}\n>h1\nMKT-\tYIAK\n", "*"),
        (f">q\n{Q}*\n>h1\nMKT-YIAK\n", "*"),
        (f">q\n{Q}\n>h1\n\n", "*"),
        (">q\nMKTAYIAG\n", "*"),
        (f">q\n{Q}\n>h1\nMKT\n", "*"),
    ],
    ids=[
        "plain",
        "wrapped",
        "insertions-repeated-header",
        "dot",
        "first-row-insertion",
        "comment-mid",
        "comment-top",
        "space",
        "tab",
        "star",
        "empty-row",
        "wrong-query",
        "short-row",
    ],
)
def test_af2_and_shared_a3m_rows_agree(text: str, af2_only_error: str | None) -> None:
    """AF2 parses with ``a3m_rows``' parser: same rows when both accept, plus AF2's own ColabFold rules.

    ``af2_only_error`` is ``None`` when both accept, ``"*"`` when both reject, and
    otherwise the message of the AF2-only rule that rejects text ``a3m_rows`` accepts.
    """
    try:
        shared: list[str] | None = a3m_rows(text, Q)
    except ValueError:
        shared = None
    try:
        af2: list[str] | None = [aligned_columns(row) for _, row in parse_a3m(materialize_msa([text], [Q]))]
    except ValueError as exc:
        af2, af2_error = None, str(exc)

    if af2_only_error is None:
        assert af2 == shared is not None
    elif af2_only_error == "*":
        assert af2 is None and shared is None
    else:
        assert shared is not None and af2 is None
        assert af2_only_error.replace("\\", "") in af2_error


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
