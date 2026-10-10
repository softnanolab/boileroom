"""Spend ledger for the Modal CI controller.

Every sandbox is reserved at its worst case *before* it is created and settled at its actual cost
afterwards. Within one controller generation the committed ledger total cannot exceed its ceiling.
Launches must be serialised by the caller (the controller runs its webhook function with
`max_containers=1` behind a lock); the reaper runs inside that same container.

The ledger prices what a sandbox can cost over its whole hard lifetime: CPU, memory, a startup
allowance, and an egress allowance. It cannot see the controller's own containers, image builds or
storage, so the controller also feeds in the measured spend of both CI apps as
`external_spend_usd`. Modal's billing report lags by an hour or more, so this is a delayed backstop,
not a strict all-in spend cap. Keep headroom for overhead and track other migration apps separately.
The egress allowance is a price estimate, not a cap; nothing here limits how much a job uploads.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

# Modal Sandbox prices, USD (https://modal.com/pricing, checked 2026-10-10).
PRICE_CPU_CORE_S = 0.00003942
PRICE_MEM_GIB_S = 0.00000667
PRICE_EGRESS_GIB = 0.04
MIN_CPU_CORES = 0.125

STARTUP_ALLOWANCE_S = 180  # image pull + runner registration are billed before the job starts
SANDBOX_GRACE_S = 300  # Modal's hard timeout is the job limit plus this, so it outlasts the supervisor's own
EGRESS_ALLOWANCE_GIB = 2.0  # artifact/cache uploads; ingress (pip, git, downloads) is free
DAY_S = 86400
ROLLUP_KEY = "rollup"  # Modal Dict entries expire after 7 idle days; old settled jobs are folded in here
MAX_CEILING_USD = 250.0  # the migration's all-in Modal budget; a Secret cannot raise the CI ceiling past it


@dataclass(frozen=True)
class Profile:
    """One sandbox shape, selected by runner label."""

    name: str
    cpu: float
    memory_gib: float
    max_seconds: int

    @property
    def hard_timeout_s(self) -> int:
        """The Modal sandbox timeout: the longest a sandbox of this profile can run."""
        return self.max_seconds + SANDBOX_GRACE_S


# Sized for what the routed jobs measure: a 1-vCPU pytest peaking near 100 MiB. 0.5 Modal cores is one vCPU; the
# memory covers the Actions runner (dotnet) and pip with room to spare. A bigger box only bills for idle cores.
PROFILES = {
    "modal-ci": Profile("modal-ci", cpu=0.5, memory_gib=2.0, max_seconds=3600),
    "modal-ci-medium": Profile("modal-ci-medium", cpu=1.0, memory_gib=8.0, max_seconds=3600),
    "modal-ci-long": Profile("modal-ci-long", cpu=0.5, memory_gib=2.0, max_seconds=210 * 60),
    "modal-ci-heavy": Profile("modal-ci-heavy", cpu=2.0, memory_gib=16.0, max_seconds=210 * 60),
}


def cost_usd(profile: Profile, seconds: float, egress_gib: float = EGRESS_ALLOWANCE_GIB) -> float:
    cores = max(profile.cpu, MIN_CPU_CORES)
    return seconds * (cores * PRICE_CPU_CORE_S + profile.memory_gib * PRICE_MEM_GIB_S) + egress_gib * PRICE_EGRESS_GIB


def worst_case_usd(profile: Profile) -> float:
    return cost_usd(profile, profile.hard_timeout_s + STARTUP_ALLOWANCE_S)


class Store(Protocol):
    """The slice of `modal.Dict` the ledger uses."""

    def get(self, key: str, /) -> Any: ...
    def put(self, key: str, value: Any, *, skip_if_exists: bool = False) -> bool: ...
    def items(self) -> Iterable[tuple[str, Any]]: ...
    def pop(self, key: str) -> Any: ...


@dataclass(frozen=True)
class Limits:
    ceiling_usd: float  # the whole CI budget, controller overhead included
    daily_cap_usd: float
    max_concurrent: int

    def __post_init__(self) -> None:
        if not 0 < self.ceiling_usd <= MAX_CEILING_USD:
            raise ValueError(f"ceiling_usd must be in (0, {MAX_CEILING_USD}], got {self.ceiling_usd}")
        if not 0 < self.daily_cap_usd <= self.ceiling_usd:
            raise ValueError(f"daily_cap_usd must be in (0, ceiling_usd], got {self.daily_cap_usd}")
        if self.max_concurrent < 1:
            raise ValueError(f"max_concurrent must be at least 1, got {self.max_concurrent}")


class Ledger:
    def __init__(self, store: Store, limits: Limits, clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.limits = limits
        self.clock = clock

    def records(self) -> list[dict[str, Any]]:
        return [v for k, v in self.store.items() if k.startswith("job:") and isinstance(v, dict)]

    def totals(self, now: float | None = None) -> dict[str, float]:
        now = self.clock() if now is None else now
        settled = (self.store.get(ROLLUP_KEY) or {}).get("usd", 0.0)
        committed = today = 0.0
        active = 0
        for rec in self.records():
            prior = rec.get("prior_usd", 0.0)  # what this job's earlier, already settled tries cost
            settled += prior
            if rec["state"] == "settled":
                settled += rec["actual_usd"]
                spent_or_reserved = prior + rec["actual_usd"]
            else:
                committed += rec["reserved_usd"]
                active += 1
                spent_or_reserved = prior + rec["reserved_usd"]
            if now - rec["created"] < DAY_S:
                today += spent_or_reserved
        return {"settled": settled, "committed": committed, "active": active, "today": today}

    def reserve(
        self, job_id: int, profile: Profile, meta: dict[str, Any], external_spend_usd: float = 0.0
    ) -> str | None:
        """Reserve the worst case for `job_id`. Returns a denial reason, or `None` on success."""
        now = self.clock()
        key = f"job:{job_id}"
        existing = self.store.get(key)
        if existing is not None and existing["state"] != "settled":
            return "already_launched"
        t = self.totals(now)
        need = worst_case_usd(profile)
        spent = max(t["settled"], external_spend_usd)
        if spent + t["committed"] + need > self.limits.ceiling_usd:
            return "budget_exhausted"
        if t["today"] + need > self.limits.daily_cap_usd:
            return "daily_cap"
        if t["active"] >= self.limits.max_concurrent:
            return "capacity"
        record = {
            **meta,
            "job_id": job_id,
            "state": "reserved",
            "profile": profile.name,
            "reserved_usd": need,
            "actual_usd": 0.0,
            "created": now,
            "tries": (existing or {}).get("tries", 0) + 1,
            # A retry replaces the record, so what the earlier tries cost is carried along.
            "prior_usd": (existing or {}).get("prior_usd", 0.0) + (existing or {}).get("actual_usd", 0.0),
        }
        if existing is None:
            # Atomic claim: a second controller container (e.g. during a redeploy) loses the race.
            if not self.store.put(key, record, skip_if_exists=True):
                return "already_launched"
        else:
            self.store.put(key, record)
        return None

    def update(self, job_id: int, **fields: Any) -> None:
        key = f"job:{job_id}"
        rec = self.store.get(key)
        if rec is not None and rec["state"] != "settled":
            self.store.put(key, {**rec, **fields})

    def settle(self, job_id: int, seconds: float) -> float:
        """Replace the reservation with the observed cost (capped at the reservation's window)."""
        key = f"job:{job_id}"
        rec = self.store.get(key)
        if rec is None or rec["state"] == "settled":
            return 0.0
        profile = PROFILES[rec["profile"]]
        # No sandbox time means nothing was sent anywhere, so no egress allowance either.
        actual = min(cost_usd(profile, seconds, EGRESS_ALLOWANCE_GIB if seconds > 0 else 0.0), rec["reserved_usd"])
        self.store.put(key, {**rec, "state": "settled", "actual_usd": actual, "settled": self.clock()})
        return actual

    def fold(self, older_than_s: float = DAY_S) -> int:
        """Fold settled records older than `older_than_s` into the roll-up. Returns how many."""
        now = self.clock()
        rollup = self.store.get(ROLLUP_KEY)
        folded = 0
        for key, rec in list(self.store.items()):
            if key.startswith("job:") and rec["state"] == "settled" and now - rec["settled"] > older_than_s:
                rollup = rollup or {"usd": 0.0, "jobs": 0}
                rollup = {
                    "usd": rollup["usd"] + rec["actual_usd"] + rec.get("prior_usd", 0.0),
                    "jobs": rollup["jobs"] + 1,
                }
                self.store.put(ROLLUP_KEY, rollup)  # persist before popping: a crash double-counts, never loses
                self.store.pop(key)
                folded += 1
        if rollup is not None and not folded:
            self.store.put(ROLLUP_KEY, rollup)  # rewritten every sweep so an idle week cannot expire the total
        return folded
