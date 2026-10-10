"""The reservation ledger: the controller's hard ceiling on Modal spend."""

import pytest

from infra.modal_ci import ledger as L

PROFILE = L.PROFILES["modal-ci"]


class FakeStore(dict):
    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def make(ceiling=10.0, daily=None, concurrent=4):
    clock = Clock()
    store = FakeStore()
    daily = min(10.0, ceiling) if daily is None else daily
    return L.Ledger(store, L.Limits(ceiling, daily, concurrent), clock), store, clock


def test_worst_case_covers_cpu_memory_startup_and_egress() -> None:
    expected = (PROFILE.hard_timeout_s + L.STARTUP_ALLOWANCE_S) * (
        PROFILE.cpu * L.PRICE_CPU_CORE_S + PROFILE.memory_gib * L.PRICE_MEM_GIB_S
    ) + L.EGRESS_ALLOWANCE_GIB * L.PRICE_EGRESS_GIB
    assert L.worst_case_usd(PROFILE) == pytest.approx(expected)
    assert 0.1 < expected < 0.5  # sanity: about twenty cents for an hour on half a core


def test_reserve_is_idempotent_per_job() -> None:
    ledger, _, _ = make()
    assert ledger.reserve(1, PROFILE, {"repo": "r"}) is None
    assert ledger.reserve(1, PROFILE, {"repo": "r"}) == "already_launched"
    assert ledger.totals()["active"] == 1


def test_ceiling_counts_reserved_and_settled_spend() -> None:
    worst = L.worst_case_usd(PROFILE)
    ledger, _, _ = make(ceiling=2.5 * worst)
    assert ledger.reserve(1, PROFILE, {}) is None
    assert ledger.reserve(2, PROFILE, {}) is None
    assert ledger.reserve(3, PROFILE, {}) == "budget_exhausted"  # 3 reservations exceed 2.5x
    ledger.settle(1, 60)  # finished quickly: most of its reservation comes back
    assert ledger.reserve(3, PROFILE, {}) is None


def test_measured_external_spend_can_close_the_budget_before_the_ledger_does() -> None:
    ledger, _, _ = make(ceiling=5.0)
    assert ledger.reserve(1, PROFILE, {}, external_spend_usd=4.9) == "budget_exhausted"
    assert ledger.reserve(1, PROFILE, {}, external_spend_usd=0.0) is None


def test_concurrency_cap() -> None:
    ledger, _, _ = make(concurrent=2)
    assert ledger.reserve(1, PROFILE, {}) is None
    assert ledger.reserve(2, PROFILE, {}) is None
    assert ledger.reserve(3, PROFILE, {}) == "capacity"
    ledger.settle(2, 10)
    assert ledger.reserve(3, PROFILE, {}) is None


def test_daily_cap_rolls_over() -> None:
    worst = L.worst_case_usd(PROFILE)
    ledger, _, clock = make(daily=1.5 * worst)
    assert ledger.reserve(1, PROFILE, {}) is None
    ledger.settle(1, PROFILE.max_seconds)
    assert ledger.reserve(2, PROFILE, {}) == "daily_cap"
    clock.now += L.DAY_S + 1
    assert ledger.reserve(2, PROFILE, {}) is None


def test_settle_never_exceeds_the_reservation_and_is_idempotent() -> None:
    ledger, store, _ = make()
    ledger.reserve(1, PROFILE, {})
    reserved = store["job:1"]["reserved_usd"]
    assert ledger.settle(1, 10 * PROFILE.max_seconds) == pytest.approx(reserved)
    assert ledger.settle(1, 5) == 0.0
    assert ledger.totals()["active"] == 0
    assert ledger.totals()["settled"] == pytest.approx(reserved)


def test_a_settled_job_may_be_relaunched_and_counts_its_tries() -> None:
    ledger, store, _ = make()
    ledger.reserve(1, PROFILE, {})
    ledger.settle(1, 30)
    assert ledger.reserve(1, PROFILE, {}) is None
    assert store["job:1"]["tries"] == 2


def test_fold_keeps_totals_while_dropping_old_records() -> None:
    ledger, store, clock = make()
    ledger.reserve(1, PROFILE, {})
    spent = ledger.settle(1, 600)
    ledger.reserve(2, PROFILE, {})
    clock.now += 2 * L.DAY_S
    assert ledger.fold() == 1
    assert "job:1" not in store and "job:2" in store
    assert ledger.totals()["settled"] == pytest.approx(spent)


def test_reservation_covers_the_sandbox_hard_timeout() -> None:
    """A sandbox can run until Modal's timeout; the reservation must cover that, not just the job limit."""
    hard = PROFILE.hard_timeout_s
    assert hard > PROFILE.max_seconds
    assert L.worst_case_usd(PROFILE) >= L.cost_usd(PROFILE, hard + L.STARTUP_ALLOWANCE_S)


def test_first_claim_of_a_job_is_atomic() -> None:
    """If another controller container claims the job between our read and write, we must not launch too."""
    ledger, store, _ = make()
    put = store.put

    def racing_put(key, value, *, skip_if_exists=False):
        store.pop(key, None)
        put(key, {**value, "state": "reserved", "reserved_usd": 1.0})  # the other container got there first
        return put(key, value, skip_if_exists=skip_if_exists)

    store.put = racing_put
    assert ledger.reserve(1, PROFILE, {}) == "already_launched"


@pytest.mark.parametrize(
    ("ceiling", "daily", "concurrent"),
    [
        (L.MAX_CEILING_USD + 1, 10.0, 4),  # a Secret cannot lift the CI ceiling past the all-in budget
        (0.0, 0.0, 4),
        (-5.0, 1.0, 4),
        (10.0, 20.0, 4),  # a daily cap above the ceiling is meaningless
        (10.0, 0.0, 4),
        (10.0, 5.0, 0),
    ],
)
def test_limits_refuse_nonsense(ceiling, daily, concurrent) -> None:
    with pytest.raises(ValueError):
        L.Limits(ceiling, daily, concurrent)


def test_limits_accept_the_budget_itself() -> None:
    L.Limits(L.MAX_CEILING_USD, 10.0, 4)


def test_a_retry_carries_the_spend_of_its_earlier_tries() -> None:
    ledger, store, clock = make()
    assert ledger.reserve(1, PROFILE, {}) is None
    clock.now += 600
    first = ledger.settle(1, 600)
    assert first > 0
    assert ledger.reserve(1, PROFILE, {}) is None  # the retry replaces the record...
    assert ledger.totals()["settled"] == pytest.approx(first)  # ...but not what the first try cost
    second = ledger.settle(1, 300)
    assert ledger.totals()["settled"] == pytest.approx(first + second)
    assert store["job:1"]["tries"] == 2


def test_a_retry_counts_toward_the_daily_cap() -> None:
    worst = L.worst_case_usd(PROFILE)
    ledger, _, _ = make(ceiling=10.0, daily=worst * 1.6)
    assert ledger.reserve(1, PROFILE, {}) is None
    first = ledger.settle(1, 3600)
    assert ledger.reserve(1, PROFILE, {}) == "daily_cap"  # first try's spend + a fresh worst case
    assert first > 0


def test_folding_keeps_the_prior_spend_and_rewrites_the_rollup_when_idle() -> None:
    ledger, store, clock = make()
    ledger.reserve(1, PROFILE, {})
    first = ledger.settle(1, 600)
    ledger.reserve(1, PROFILE, {})
    second = ledger.settle(1, 600)
    clock.now += L.DAY_S + 1
    assert ledger.fold() == 1 and "job:1" not in store
    assert store[L.ROLLUP_KEY]["usd"] == pytest.approx(first + second)
    puts: list[str] = []
    original_put = store.put

    def recording_put(key, value, **kw):
        puts.append(key)
        return original_put(key, value, **kw)

    store.put = recording_put
    assert ledger.fold() == 0
    assert puts == [L.ROLLUP_KEY]  # an idle sweep still rewrites it, so Dict expiry cannot erase the total
