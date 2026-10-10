"""The Modal-facing pieces of the controller, against stand-ins for the Modal SDK."""

import datetime as dt
from types import SimpleNamespace

import pytest

from infra.modal_ci import controller
from infra.modal_ci.core import SpendUnknown
from infra.modal_ci.ledger import PROFILES, worst_case_usd


class Clock:
    now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def test_sandboxes_are_created_with_hard_cpu_and_memory_limits(monkeypatch) -> None:
    """A bare number is only a request; a job that bursts past it would bill more than the ledger reserved."""
    seen = {}

    def create(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(object_id="sb-1", set_tags=lambda tags: None)

    monkeypatch.setattr(controller.modal.Sandbox, "create", create)
    sandboxes = object.__new__(controller.ModalSandboxes)
    sandboxes.app = SimpleNamespace(app_id="ap-1")
    profile = PROFILES["modal-ci"]
    assert sandboxes.create(profile=profile, name="n", env={}, tags={}) == "sb-1"
    assert seen["cpu"] == (profile.cpu, profile.cpu)
    assert seen["memory"] == (int(profile.memory_gib * 1024),) * 2
    assert seen["timeout"] == profile.hard_timeout_s


def test_the_reservation_prices_exactly_what_the_limits_allow() -> None:
    """The ledger's worst case is computed from the profile's cores and GiB: those must be the limits."""
    profile = PROFILES["modal-ci"]
    assert worst_case_usd(profile) > 0
    assert profile.cpu == 0.5 and profile.memory_gib == 2.0


def report(monkeypatch, rows=None, error=None):
    calls = []

    def fake(start, resolution):
        calls.append((start, resolution))
        if error:
            raise error
        return rows

    monkeypatch.setattr(controller.modal.billing, "workspace_billing_report", fake)
    return calls


ROWS = [
    {"description": controller.CONTROLLER_APP, "cost": "1.5"},
    {"description": controller.RUNNER_APP, "cost": "2.25"},
    {"description": "some-other-app", "cost": "100"},
]


def test_billing_probe_sums_only_the_two_ci_apps_and_caches(monkeypatch) -> None:
    calls = report(monkeypatch, ROWS)
    clock = Clock()
    probe = controller.BillingProbe("2026-10-10T00:00:00", clock)
    assert probe() == pytest.approx(3.75)
    clock.now += 60
    assert probe() == pytest.approx(3.75)
    assert len(calls) == 1
    clock.now += controller.BillingProbe.REFRESH_S
    probe()
    assert len(calls) == 2


def test_billing_probe_fails_closed_when_it_has_never_succeeded(monkeypatch) -> None:
    report(monkeypatch, error=RuntimeError("no billing access"))
    probe = controller.BillingProbe("2026-10-10T00:00:00", Clock())
    with pytest.raises(SpendUnknown):
        probe()


@pytest.mark.parametrize(
    "since, expected",
    [
        ("2026-10-10T00:00:00", "2026-10-10T00:00:00+00:00"),
        ("2026-10-10T00:00:00Z", "2026-10-10T00:00:00+00:00"),
        ("2026-10-10T00:00:00+02:00", "2026-10-09T22:00:00+00:00"),
        ("2026-10-10T00:00:00-04:00", "2026-10-10T04:00:00+00:00"),
    ],
)
def test_billing_probe_queries_the_actual_budget_start_instant(monkeypatch, since, expected) -> None:
    calls = report(monkeypatch, ROWS)
    controller.BillingProbe(since, Clock())()
    assert calls == [(dt.datetime.fromisoformat(expected), "h")]


def test_billing_probe_tolerates_a_short_outage_but_not_a_long_one(monkeypatch) -> None:
    clock = Clock()
    report(monkeypatch, ROWS)
    probe = controller.BillingProbe("2026-10-10T00:00:00", clock)
    assert probe() == pytest.approx(3.75)

    report(monkeypatch, error=RuntimeError("billing down"))
    clock.now += 3600
    assert probe() == pytest.approx(3.75)  # last good reading, an hour old
    clock.now += controller.BillingProbe.MAX_AGE_S
    with pytest.raises(SpendUnknown):
        probe()  # too stale to reserve against

    report(monkeypatch, ROWS)
    clock.now += controller.BillingProbe.REFRESH_S
    assert probe() == pytest.approx(3.75)  # recovers by itself
