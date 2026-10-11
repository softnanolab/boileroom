"""Type definitions for OpenDDE outputs without heavy runtime dependencies."""

from dataclasses import dataclass

from ..protenix.types import ProtenixOutput


@dataclass
class OpenDDEOutput(ProtenixOutput):
    """Output from OpenDDE structure prediction.

    OpenDDE writes the same sample, summary-confidence and full-data files as Protenix 2.0, so the
    output schema (per-sample atom arrays, pTM/ipTM, PAE, token chain/residue ids, pLDDT) is shared.
    """
