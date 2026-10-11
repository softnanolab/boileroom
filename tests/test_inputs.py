"""The shared A3M parser and the ``a3m_rows`` validator used by the Protenix, OpenDDE and ESMFold2 cores."""

import pytest

from boileroom.inputs import a3m_rows, aligned_columns, parse_a3m

Q = "MKTAYIAK"


def test_parse_a3m_joins_wrapped_rows_and_keeps_repeated_headers() -> None:
    text = "\n>q desc\nMKTA\n  YIAK  \n\n>h\nMKT-\nYIAK\n>h\nMKabT-YIAK\n"

    assert parse_a3m(text) == [("q desc", Q), ("h", "MKT-YIAK"), ("h", "MKabT-YIAK")]


def test_parse_a3m_leaves_comment_lines_and_inner_whitespace_for_callers() -> None:
    """The parser does not silently drop content: callers' checks see it and refuse."""
    assert parse_a3m(">q\nMK TA\n#note\n>h\nMK\tTA\n") == [("q", "MK TA#note"), ("h", "MK\tTA")]


@pytest.mark.parametrize(("text", "match"), [("MKTA\n>q\nMKTA\n", "before the first '>'"), ("\n \n", "no sequences")])
def test_parse_a3m_failures(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_a3m(text)


def test_aligned_columns_drops_lowercase_insertions_and_dots() -> None:
    assert aligned_columns("MKab.T-YI.AK") == "MKT-YIAK"


@pytest.mark.parametrize(
    ("text", "rows"),
    [
        (f">q\n{Q}\n>h1\nMKT-YIAK\n", [Q, "MKT-YIAK"]),
        (">q\nMKTA\nYIAK\n>h1\nMKT-\nYIAK\n", [Q, "MKT-YIAK"]),
        (f">q\n{Q}\n>h\nMKabT-YI.AK\n>h\nMKT-YIAK\n", [Q, "MKT-YIAK", "MKT-YIAK"]),
        (">q\nMKTaAYIAK\n", [Q]),
        (f"\n  >q\r\n{Q}\r\n", [Q]),
    ],
    ids=["plain", "wrapped", "insertions-dots-repeated-header", "first-row-insertion", "leading-blank-crlf"],
)
def test_a3m_rows_accepts(text: str, rows: list[str]) -> None:
    assert a3m_rows(text, Q) == rows


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (Q, "A3M text or None"),
        (None, "A3M text or None"),
        (f"#A3M#\n>q\n{Q}\n", "A3M text or None"),
        (">q\nMKTAYIAG\n", "first A3M row must match"),
        (">q\nMKTA YIAK\n", "first A3M row must match"),
        (f">q\n{Q}*\n", "first A3M row must match"),
        (f">q\n{Q}\n#note\n", "first A3M row must match"),
        (f">q\n{Q}\n>h\nMKT\n", "aligned length"),
        (f">q\n{Q}\n>h\n\n", "aligned length"),
        (f">q\n{Q}\n>h\nMKT-\tYIAK\n", "aligned length"),
        (f">q\n{Q}\n#note\n>h\nMKT-YIAK\n", "first A3M row must match"),
    ],
    ids=[
        "no-header",
        "none",
        "leading-comment",
        "wrong-query",
        "space-in-query",
        "star",
        "trailing-comment",
        "short-row",
        "empty-row",
        "tab-in-row",
        "comment-mid",
    ],
)
def test_a3m_rows_rejects(text: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        a3m_rows(text, Q)
