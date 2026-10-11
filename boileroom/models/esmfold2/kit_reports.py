"""ESMFold2 kit provenance and data-dependent lever report settlement."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from typing import Any

from ...provenance import _UNKNOWN, _joined, kit_provenance

KIT_COMMIT_ENV = "BOILEROOM_KIT_COMMIT"
KIT_COMMIT_ATTRIBUTES = ("__commit__", "KIT_COMMIT", "COMMIT")


def _kit_commit(kit: Any) -> str | None:
    """The kit's source commit: its own attribute (:data:`KIT_COMMIT_ATTRIBUTES`), else ``$BOILEROOM_KIT_COMMIT``.

    The same order as ``protenix/runtime.py``, without its ``.git`` lookup: the ESMFold2 kit image installs the kit
    without its repository, and a ``.git`` above an installed package would belong to some other project.

    Parameters
    ----------
    kit : Any
        The imported ``esmfold2_opt`` module, or ``None``.

    Returns
    -------
    str | None
        The commit, or ``None`` when neither source names one.
    """
    for attribute in KIT_COMMIT_ATTRIBUTES:
        value = getattr(kit, attribute, None)
        if isinstance(value, str) and value:
            return value
    return os.environ.get(KIT_COMMIT_ENV) or None


def _kit_lever_words(report: Mapping[str, Any], num_diffusion_samples: int, stack: Any) -> dict[str, str]:
    """The ``metadata.runtime`` lever entries of a kit report, and the kit's scope notes at this sample count.

    The keys every kit family shares (:func:`~boileroom.provenance.kit_provenance`: ``kit.commit``,
    ``kit.levers_applied``, ``kit.levers_fallback``, ``kit.partial``), then ``kit.levers_gated`` / ``kit.gated``,
    which name levers whose declared guard sent calls to the previous chain (e.g. ``mk`` at
    ``num_diffusion_samples > 1``), ``kit.guards``, the verdict of every data-dependent guard the settle step read
    (``<lever>=<kind>: <note>``), and ``kit.scope``, the kit's own plan note for this call's sample count.
    """
    words = kit_provenance(
        commit=_kit_commit(sys.modules.get("esmfold2_opt")),
        levers_applied=report.get("levers_applied") or [],
        levers_fallback=report.get("levers_fallback") or [],
        partial=bool(report.get("partial")),
    )
    words["kit.levers_gated"] = _joined(report.get("levers_gated") or [])
    words["kit.gated"] = _joined(report.get("gated") or [], "; ")
    guards = report.get("guards") or {}
    words["kit.guards"] = _joined(
        (f"{lever}={guard.get('kind')}: {guard.get('note')}" for lever, guard in sorted(guards.items())), "; "
    )
    notes = [
        note(report.get("levers_planned"), num_diffusion_samples)
        for note in (getattr(stack, "mk_plan_note", None), getattr(stack, "guards_plan_note", None))
        if callable(note)
    ]
    words["kit.scope"] = _joined((note for note in notes if note), "; ")
    return words


def _gate_fallen_through_t16(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Settle ``t16`` as gated, not partial, when every call reached its wrapper and fell through by name.

    ``stack.guards_after_run`` calls ``t16`` "unreached" whenever no Transition / PairTransition call was served by
    the kernel, even when every call did reach the wrapper and fell through: a by-name refusal (the provider's
    cell table names the statement for that size, or the shape is outside its envelope) or an ineligible call.
    Each such call runs the previous statement, the fork's own fused forward, so the values are the previous
    chain's and only the speed differs: the same evidence the kit settles as "inactive" for other levers. Such a
    run is recorded (``t16`` in ``kit.levers_gated``, the counters and refusal reasons in ``kit.gated`` and
    ``kit.guards``) rather than refused. Zero calls at the wrapper (``fallthrough == 0``) stays "unreached" and is
    still refused: nothing shows the lever's code ever ran.
    """
    guard = (report.get("guards") or {}).get("t16") or {}
    if guard.get("kind") != "unreached" or "t16" not in (report.get("partial") or []):
        return report
    cute = sys.modules.get("ef2_transition_cute")
    stats = dict(getattr(cute, "STATS", None) or {})
    served = int(stats.get("transition_calls", 0) or 0) + int(stats.get("pair_transition_calls", 0) or 0)
    fell = int(stats.get("fallthrough_transition", 0) or 0) + int(stats.get("fallthrough_pair_transition", 0) or 0)
    if served or not fell:
        return report
    describe = getattr(cute, "describe", None)
    refused = (describe() or {}).get("refused") if callable(describe) else None
    reasons = ",".join(f"{reason}={count}" for reason, count in sorted((refused or {}).items())) or _UNKNOWN
    note = (
        f"all {fell} Transition / PairTransition call(s) took the previous statements, none served "
        f"(fallthrough_transition={int(stats.get('fallthrough_transition', 0) or 0)} "
        f"fallthrough_pair_transition={int(stats.get('fallthrough_pair_transition', 0) or 0)} "
        f"face_refused={int(stats.get('face_refused', 0) or 0)}; refused: {reasons}); "
        "gated by boileroom, not partial"
    )
    fallback_reasons = dict(report.get("fallback_reasons") or {})
    fallback_reasons.pop("t16", None)
    return {
        **report,
        "partial": [lever for lever in report.get("partial") or [] if lever != "t16"],
        "levers_fallback": [lever for lever in report.get("levers_fallback") or [] if lever != "t16"],
        "fallback_reasons": fallback_reasons,
        "levers_gated": sorted({*(report.get("levers_gated") or []), "t16"}),
        "gated": [*(report.get("gated") or []), f"t16: {note}"],
        "guards": {**(report.get("guards") or {}), "t16": {"kind": "inactive", "note": note}},
    }
