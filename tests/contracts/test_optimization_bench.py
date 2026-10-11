"""The benchmark numbers in ``docs/optimization.md`` trace to ``docs/optimization-bench/summary.json``.

Every row of the table under ``## Measured`` must be a cell of ``summary.json`` printed at the documented precision
(warm seconds to 2 decimals, first-fold seconds to 1 decimal, cost per warm fold as ``warm * price`` to 2 significant
figures), and every cell of ``summary.json`` must have a row. The derived OpenDDE figures quoted under the table are
recomputed from the same cells. Only the standard library is used, so no model dependency is imported.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "optimization.md"
SUMMARY = REPO / "docs" / "optimization-bench" / "summary.json"
TABLE_HEADER = ("Model", "GPU", "Mode", "Warm (s)", "First (s)", "Cost per warm fold (USD)")


@dataclass(frozen=True)
class BenchRow:
    """One row of the documented benchmark table, as printed."""

    model: str
    gpu: str
    mode: str
    warm: str
    first: str
    cost: str

    @property
    def cell(self) -> str:
        """Return the ``summary.json`` key of this row within its model, such as ``"A100 fast"``."""
        return f"{self.gpu} {self.mode}"


def _cells(line: str) -> list[str]:
    """Return the stripped cells of one Markdown table line."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def measured_rows(text: str) -> list[BenchRow]:
    """Return the rows of the benchmark table in the ``## Measured`` section of ``text``.

    Parameters
    ----------
    text : str
        Markdown of the optimization page.

    Returns
    -------
    list[BenchRow]
        The table rows in document order.

    Raises
    ------
    ValueError
        If the section or its table is missing, or a row has the wrong number of cells.
    """
    match = re.search(r"^## Measured\s*$(.*?)(?=^## |\Z)", text, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError("docs/optimization.md has no '## Measured' section")
    lines = match.group(1).splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("|") and tuple(_cells(line)) == TABLE_HEADER),
        None,
    )
    if header_index is None:
        raise ValueError(f"the Measured section has no table with header {TABLE_HEADER}")
    rows: list[BenchRow] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        cells = _cells(line)
        if len(cells) != len(TABLE_HEADER):
            raise ValueError(f"benchmark row has {len(cells)} cells, expected {len(TABLE_HEADER)}: {line!r}")
        rows.append(BenchRow(*cells))
    if not rows:
        raise ValueError("the Measured table has no rows")
    return rows


def table_errors(rows: list[BenchRow], results: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    """Return every disagreement between the documented rows and the ``results`` of ``summary.json``.

    Parameters
    ----------
    rows : list[BenchRow]
        The documented table rows.
    results : dict[str, dict[str, dict[str, Any]]]
        ``summary.json["results"]``: model -> ``"<GPU> <mode>"`` -> ``first``/``warm``/``price``.

    Returns
    -------
    list[str]
        One message per wrong, duplicated, unbacked or missing row; empty when the table matches.
    """
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.model, row.cell)
        if key in seen:
            errors.append(f"{row.model} {row.cell}: duplicated row")
            continue
        seen.add(key)
        cell = results.get(row.model, {}).get(row.cell)
        if cell is None:
            errors.append(f"{row.model} {row.cell}: not in summary.json")
            continue
        warm, first, price = float(cell["warm"]), float(cell["first"]), float(cell["price"])
        if row.warm != f"{warm:.2f}":
            errors.append(f"{row.model} {row.cell}: warm {row.warm} != {warm:.2f}")
        if row.first != f"{first:.1f}":
            errors.append(f"{row.model} {row.cell}: first {row.first} != {first:.1f}")
        try:
            documented_cost = float(row.cost)
        except ValueError:
            documented_cost = float("nan")
        if documented_cost != float(f"{warm * price:.2g}"):
            errors.append(f"{row.model} {row.cell}: cost {row.cost} != {warm * price:.2g}")
    for model, cells in results.items():
        for cell_name in cells:
            if (model, cell_name) not in seen:
                errors.append(f"{model} {cell_name}: in summary.json but not in the table")
    return errors


@pytest.fixture(scope="module")
def results() -> dict[str, dict[str, dict[str, Any]]]:
    """Return ``summary.json["results"]``."""
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert {"unit", "source", "results"} <= summary.keys()
    return summary["results"]


@pytest.fixture(scope="module")
def doc_text() -> str:
    """Return the optimization page."""
    return DOC.read_text(encoding="utf-8")


def test_summary_covers_doc_tables(doc_text: str, results: dict[str, dict[str, dict[str, Any]]]) -> None:
    """Every documented number is a ``summary.json`` cell at the stated precision, and every cell is documented."""
    rows = measured_rows(doc_text)
    assert table_errors(rows, results) == []


def test_opendde_derived_figures_match_summary(doc_text: str, results: dict[str, dict[str, dict[str, Any]]]) -> None:
    """The OpenDDE same-GPU speedups and first-fold overhead quoted under the table follow from the cells."""
    opendde = results["OpenDDE"]
    for gpu in ("A100", "H100"):
        vanilla = float(opendde[f"{gpu} vanilla"]["warm"])
        for mode in ("exact", "fast"):
            speedup = vanilla / float(opendde[f"{gpu} {mode}"]["warm"])
            assert f"{speedup:.2f}x" in doc_text, f"OpenDDE {gpu} {mode} speedup {speedup:.2f}x is not quoted"
    deltas = [
        float(opendde[f"{gpu} {mode}"]["first"]) - float(opendde[f"{gpu} vanilla"]["first"])
        for gpu in ("A100", "H100")
        for mode in ("exact", "fast")
    ]
    assert f"{min(deltas):.1f}-{max(deltas):.1f} s" in doc_text


def test_measured_section_states_provenance(doc_text: str) -> None:
    """The section names the kit commit and the OpenDDE image, and flags the unverified OpenDDE lever set."""
    section = doc_text[doc_text.index("## Measured") :]
    assert "f4f62fa" in section
    assert "cuda12.6-sha-bdc9a27" in section
    assert "unverified" in section
    assert "predate" in section


SYNTHETIC_RESULTS: dict[str, dict[str, dict[str, Any]]] = {
    "M": {
        "A100 fast": {"first": 27.24, "warm": 0.4651, "price": 0.000694},
        "L4 vanilla": {"first": 7.31, "warm": 4.4049, "price": 0.000222},
    }
}
SYNTHETIC_DOC = """## Measured

| Model | GPU | Mode | Warm (s) | First (s) | Cost per warm fold (USD) |
| --- | --- | --- | --- | --- | --- |
| M | A100 | fast | 0.47 | 27.2 | 0.00032 |
| M | L4 | vanilla | 4.40 | 7.3 | 0.00098 |

## Next
"""


def test_synthetic_table_matches() -> None:
    """A table printed from its summary at the documented precision has no errors."""
    assert table_errors(measured_rows(SYNTHETIC_DOC), SYNTHETIC_RESULTS) == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("| 0.47 | 27.2 |", "| 0.48 | 27.2 |", "warm 0.48"),
        ("| 0.47 | 27.2 |", "| 0.47 | 27.9 |", "first 27.9"),
        ("| 0.00098 |", "| 0.0010 |", "cost 0.0010"),
        ("| M | A100 | fast |", "| M | H100 | fast |", "H100 fast: not in summary.json"),
        (
            "| M | L4 | vanilla | 4.40 | 7.3 | 0.00098 |\n",
            "| M | L4 | vanilla | 4.40 | 7.3 | 0.00098 |\n| N | L4 | vanilla | 1.00 | 1.0 | 0.001 |\n",
            "N L4 vanilla: not in summary.json",
        ),
        ("| M | L4 | vanilla | 4.40 | 7.3 | 0.00098 |\n", "", "L4 vanilla: in summary.json but not in the table"),
        (
            "| M | L4 | vanilla | 4.40 | 7.3 | 0.00098 |\n",
            "| M | L4 | vanilla | 4.40 | 7.3 | 0.00098 |\n| M | L4 | vanilla | 4.40 | 7.3 | 0.00098 |\n",
            "duplicated row",
        ),
    ],
)
def test_table_errors_detect_drift(old: str, new: str, expected: str) -> None:
    """A wrong value, an unbacked row, a missing row or a duplicate is reported."""
    errors = table_errors(measured_rows(SYNTHETIC_DOC.replace(old, new)), SYNTHETIC_RESULTS)
    assert any(expected in error for error in errors), errors


@pytest.mark.parametrize(
    "doc",
    [
        "## Elsewhere\n\n| a |\n",
        "## Measured\n\nNo table here.\n",
        SYNTHETIC_DOC.replace("| 0.00098 |", ""),
    ],
)
def test_measured_rows_refuses_malformed_doc(doc: str) -> None:
    """A missing section, a missing table or a short row raises instead of passing vacuously."""
    with pytest.raises(ValueError):
        measured_rows(doc)
