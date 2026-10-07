"""The kit entries of ``PredictionMetadata.runtime`` documented in ``docs/optimization.md`` are the ones the code writes.

Every kit family records its kit facts under the shared ``kit.`` prefix: the shared table must list exactly
:data:`boileroom.provenance.KIT_RUNTIME_KEYS`, and the ESMFold2 kit-mode bullet must name each kernel word of
``ESMFold2Core`` as ``kit.attn.<word>`` and document no kit key outside the ``kit.`` namespace.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

REPO = Path(__file__).resolve().parents[2]
OPTIMIZATION_DOC = REPO / "docs" / "optimization.md"
_BACKTICKED = re.compile(r"`([^`]+)`")
_SENTENCE_END = re.compile(r"\.\s+(?=[A-Z])")


def shared_kit_table(text: str) -> list[str]:
    """Return the keys of the table that follows the shared kit-entries paragraph of ``text``.

    Raises
    ------
    ValueError
        If the paragraph or its table is missing.
    """
    match = re.search(r"also records the same kit entries.*?\n\n((?:\|.*\n)+)", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("no shared kit-entries table")
    rows = [line for line in match.group(1).splitlines() if line.startswith("|")][2:]
    keys = [_BACKTICKED.findall(row.split("|")[1]) for row in rows]
    return [key for cell in keys for key in cell]


def esmfold2_kit_bullet(text: str) -> str:
    """Return the "ESMFold2 kit modes" bullet of ``text``, continuation lines joined.

    Raises
    ------
    ValueError
        If the bullet is missing.
    """
    match = re.search(r"^- ESMFold2 kit modes:(.*?)(?=^- |^\S|\Z)", text, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError("no 'ESMFold2 kit modes' bullet")
    return " ".join(match.group(1).split())


def documented_keys(bullet: str) -> list[str]:
    """Return the backticked runtime keys of ``bullet``'s key list.

    The key list is the bullet's first sentence; later sentences describe behaviour and may name levers. Inside it,
    the provenance words (``none``, ``unknown``, ...) and placeholders such as ``<lever>=<kind>: <note>`` are values,
    not keys.
    """
    key_list = _SENTENCE_END.split(bullet, maxsplit=1)[0]
    return [
        token
        for token in _BACKTICKED.findall(key_list)
        if token not in provenance_words() and not any(char.isspace() for char in token)
    ]


def provenance_words() -> frozenset[str]:
    """Return the words ``boileroom.provenance`` records for a fact that is not there."""
    from boileroom import provenance

    return frozenset({provenance._NOT_LOADED, provenance._ABSENT, provenance._NONE, provenance._UNKNOWN})


def check_esmfold2_bullet(bullet: str, kernel_words: tuple[str, ...]) -> None:
    """Assert the bullet names each kernel word as ``kit.attn.<word>`` and documents only ``kit.`` keys."""
    keys = documented_keys(bullet)
    outside = [key for key in keys if not key.startswith("kit.")]
    assert not outside, f"ESMFold2 kit-mode keys outside the kit. namespace: {outside}"
    missing = [word for word in kernel_words if f"kit.attn.{word}" not in keys]
    assert not missing, f"kernel words not documented as kit.attn.<word>: {missing}"


def kernel_words() -> tuple[str, ...]:
    """Return ``ESMFold2Core``'s kernel words (the core imports numpy only, never esm or torch)."""
    return tuple(importlib.import_module("boileroom.models.esmfold2.core").KIT_ATTN_WORDS)


def test_shared_kit_table_lists_the_shared_keys() -> None:
    from boileroom.provenance import KIT_RUNTIME_KEYS

    assert shared_kit_table(OPTIMIZATION_DOC.read_text()) == list(KIT_RUNTIME_KEYS)


def test_esmfold2_kit_bullet_names_kernel_words_under_kit_attn() -> None:
    check_esmfold2_bullet(esmfold2_kit_bullet(OPTIMIZATION_DOC.read_text()), kernel_words())


_GOOD = (
    "- ESMFold2 kit modes: `kit.stack`, `kit.attn` and one entry per kernel word,\n"
    "  `kit.attn.atom_attn`, `kit.attn.esmc_mlp` (`unknown` when unreported), `kit.guards` (as\n"
    "  `<lever>=<kind>: <note>`), and `kit.scope` (`none` when empty). One exception: `t16` is gated.\n"
    "- Protenix and OpenDDE: `worker.<key>`.\n"
)


def test_check_accepts_kit_attn_keys() -> None:
    bullet = esmfold2_kit_bullet(_GOOD)
    assert "worker" not in bullet
    assert documented_keys(bullet) == [
        "kit.stack",
        "kit.attn",
        "kit.attn.atom_attn",
        "kit.attn.esmc_mlp",
        "kit.guards",
        "kit.scope",
    ]
    check_esmfold2_bullet(bullet, ("atom_attn", "esmc_mlp"))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (_GOOD.replace("`kit.attn.atom_attn`", "`atom_attn`"), "outside the kit. namespace"),
        (_GOOD.replace("`kit.scope`", "`scope`"), r"outside the kit. namespace: \['scope'\]"),
        (_GOOD.replace("`kit.attn.esmc_mlp` (`unknown` when unreported), ", ""), r"kit.attn.<word>: \['esmc_mlp'\]"),
    ],
    ids=["bare-word", "bare-key", "missing-word"],
)
def test_check_refuses_bare_or_missing_kernel_words(text: str, match: str) -> None:
    with pytest.raises(AssertionError, match=match):
        check_esmfold2_bullet(esmfold2_kit_bullet(text), ("atom_attn", "esmc_mlp"))


def test_missing_sections_are_reported() -> None:
    with pytest.raises(ValueError, match="ESMFold2 kit modes"):
        esmfold2_kit_bullet("- Protenix and OpenDDE: `worker.<key>`.\n")
    with pytest.raises(ValueError, match="shared kit-entries"):
        shared_kit_table("no table here\n")


def test_shared_kit_table_parses_rows() -> None:
    text = (
        "In `exact` every family also records the same kit entries, under the same names:\n\n"
        "| Key | Value |\n| --- | --- |\n| `kit.commit` | the commit |\n| `kit.partial` | `true` or `false` |\n\n"
        "After.\n"
    )
    assert shared_kit_table(text) == ["kit.commit", "kit.partial"]
