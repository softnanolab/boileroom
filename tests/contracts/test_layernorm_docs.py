"""The docs state the per-mode ``LAYERNORM_TYPE`` the Protenix and OpenDDE cores actually set.

The cores' ``PROTENIX_LAYERNORM`` and ``OPENDDE_LAYERNORM`` are the source of truth; the decided values are pinned once,
in ``test_layernorm_contract.py``. The "LayerNorm per mode" table in ``docs/optimization.md`` must equal the constants,
and no page may claim that vanilla OpenDDE runs the fused ``fast_layernorm`` (or that the OpenDDE image pins it).
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

REPO = Path(__file__).resolve().parents[2]
OPTIMIZATION_DOC = REPO / "docs" / "optimization.md"
#: Per family: the core module and its per-mode constant (imported inside the test, never at module scope).
CONSTANTS = {
    "protenix": ("boileroom.models.protenix.core", "PROTENIX_LAYERNORM"),
    "opendde": ("boileroom.models.opendde.core", "OPENDDE_LAYERNORM"),
}
#: Pages scanned for stale OpenDDE LayerNorm claims.
SCANNED_PAGES = sorted((REPO / "docs").glob("*.md")) + [REPO / "README.md"]

#: Claims that the OpenDDE image or environment pins the fused LayerNorm for every mode, or cannot build it.
_PINNED_CLAIMS = (
    re.compile(r"(image|dockerfile|environment)\s+sets\s+`?LAYERNORM_TYPE=fast_layernorm", re.IGNORECASE),
    re.compile(r"`?LAYERNORM_TYPE=fast_layernorm`?\s+is\s+set", re.IGNORECASE),
    re.compile(r"`?ninja`?\s+is\s+not\s+installed", re.IGNORECASE),
    re.compile(r"same\s+default\s+for\s+every\s+mode", re.IGNORECASE),
)
#: Phrases that put ``fast_layernorm`` on vanilla OpenDDE unless the sentence also gives torch as vanilla's value.
_ALL_MODES = re.compile(r"\bvanilla\b|all\s+(three\s+)?modes|every\s+mode|each\s+mode", re.IGNORECASE)


def core_layernorm(family: str) -> Mapping[str, str]:
    """Return the core's per-mode ``LAYERNORM_TYPE`` constant for ``family``."""
    module, name = CONSTANTS[family]
    return dict(getattr(importlib.import_module(module), name))


def documented_layernorm(text: str) -> dict[str, dict[str, str]]:
    """Return the "LayerNorm per mode" table of ``text`` as family -> mode -> ``LAYERNORM_TYPE``.

    Raises
    ------
    ValueError
        If the section or its table is missing.
    """
    match = re.search(r"^#+ LayerNorm per mode\s*$(.*?)(?=^#|\Z)", text, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError("no 'LayerNorm per mode' section")
    table = [line for line in match.group(1).splitlines() if line.startswith("|")]
    if len(table) < 3:
        raise ValueError("the 'LayerNorm per mode' section has no table")

    def cells(line: str) -> list[str]:
        return [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]

    header = cells(table[0])
    return {row[0].lower(): dict(zip(header[1:], row[1:], strict=True)) for row in map(cells, table[2:])}


def _blocks(text: str) -> list[tuple[str, str]]:
    """Split Markdown into paragraphs and list items, each paired with its nearest heading."""
    blocks: list[tuple[str, str]] = []
    heading = ""
    current: list[str] = []

    def flush() -> None:
        if current:
            blocks.append((heading, " ".join(current)))
            current.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            heading = stripped
        elif not stripped or re.match(r"[-*]\s|\d+\.\s|\|", stripped):
            flush()
            if stripped:
                current.append(stripped)
        else:
            current.append(stripped)
    flush()
    return blocks


def stale_opendde_layernorm_claims(text: str) -> list[str]:
    """Return the sentences of ``text`` that misstate OpenDDE's LayerNorm.

    A block (paragraph, list item or table row) is about OpenDDE when it or its heading names OpenDDE. In such a block a
    sentence is stale when it says an image or environment pins ``LAYERNORM_TYPE=fast_layernorm`` (or that ``ninja``
    is missing), or when it ties ``fast_layernorm`` to vanilla or to every mode without giving ``torch`` as vanilla's
    value.
    """
    stale: list[str] = []
    for heading, block in _blocks(text):
        if "opendde" not in f"{heading} {block}".lower():
            continue
        for sentence in re.split(r"(?<=[.;])\s+", block):
            if (
                any(pattern.search(sentence) for pattern in _PINNED_CLAIMS)
                or "fast_layernorm" in sentence
                and _ALL_MODES.search(sentence)
                and "torch" not in sentence.lower()
            ):
                stale.append(sentence)
    return stale


@pytest.mark.parametrize("family", sorted(CONSTANTS))
def test_layernorm_table_matches_core_constants(family: str) -> None:
    """The optimization page's table row for each family equals the core's per-mode constant."""
    table = documented_layernorm(OPTIMIZATION_DOC.read_text(encoding="utf-8"))
    assert table.get(family) == core_layernorm(family), (
        f"docs/optimization.md 'LayerNorm per mode' disagrees with the {CONSTANTS[family][1]} constant in "
        f"{CONSTANTS[family][0]}, the source of truth; update the docs table to match it"
    )


@pytest.mark.parametrize("page", SCANNED_PAGES, ids=lambda page: page.name)
def test_no_page_claims_vanilla_opendde_uses_fast_layernorm(page: Path) -> None:
    """No doc page or the README claims vanilla OpenDDE (or its image) uses ``fast_layernorm``."""
    assert stale_opendde_layernorm_claims(page.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    ("text", "stale"),
    [
        ("### OpenDDE\n\nThe OpenDDE image sets `LAYERNORM_TYPE=fast_layernorm`.\n", True),
        ("Vanilla OpenDDE uses fast_layernorm.\n", True),
        ("- **opendde**: `LAYERNORM_TYPE=fast_layernorm` is set but `ninja` is not installed.\n", True),
        ("### OpenDDE\n\nThe core exports the same default for every mode.\n", True),
        ("OpenDDE runs `fast_layernorm` in all three modes.\n", True),
        ("### OpenDDE\n\n`vanilla` runs torch LayerNorm, and `exact` / `fast` run `fast_layernorm`.\n", False),
        ("| OpenDDE | `torch` | `fast_layernorm` | `fast_layernorm` |\n", False),
        ("### Protenix\n\nThe Protenix kit image ships `fast_layernorm` compiled for every mode it serves.\n", False),
    ],
)
def test_stale_claim_scan_flags_only_the_wrong_statements(text: str, stale: bool) -> None:
    """The page scan above passes only because no page is stale: it does flag each old, wrong statement form."""
    assert bool(stale_opendde_layernorm_claims(text)) is stale
