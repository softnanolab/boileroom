"""Modal app: GitHub `workflow_job` webhook -> ephemeral CPU sandbox running one Actions job.

Deploy with `modal deploy -m infra.modal_ci.controller` (see `infra/modal_ci/README.md`).

Credential boundary: the GitHub App key, installation tokens and webhook secret exist only in the
`softnanolab-ci-controller` Modal Secret, mounted on the controller. Sandboxes live in a separate
app, run the controller-built image, and receive one JIT runner configuration and one binding.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import time
from collections.abc import Callable, Mapping

import modal
import modal.billing

from infra.modal_ci import policy
from infra.modal_ci.asgi import make_app
from infra.modal_ci.core import Core, SandboxState, SpendUnknown
from infra.modal_ci.github_api import GitHubApp
from infra.modal_ci.image import SOURCE_IGNORE, runner_image
from infra.modal_ci.ledger import PROFILES, Ledger, Limits, Profile

CONTROLLER_APP = "softnanolab-ci-controller"
RUNNER_APP = "softnanolab-ci-runners"
SECRET_NAME = "softnanolab-ci-controller"
# Non-secret scope, listed after the credentials so its keys win: the repositories the controller serves.
SCOPE_SECRET_NAME = "softnanolab-ci-scope"
LEDGER_DICT = "softnanolab-ci-ledger"
DELIVERY_DICT = "softnanolab-ci-deliveries"

app = modal.App(CONTROLLER_APP)
controller_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("PyJWT[crypto]==2.*")
    # `runner_image()` runs at import and reads the sandbox's files to reproduce the prebuilt image's hash.
    .add_local_python_source("infra", ignore=SOURCE_IGNORE)
)
sandbox_image = runner_image()  # built ahead of time by `python -m infra.modal_ci.build_image`, then reused by hash


class ModalSandboxes:
    def __init__(self) -> None:
        self.app = modal.App.lookup(RUNNER_APP, create_if_missing=True)

    def create(self, *, profile: Profile, name: str, env: Mapping[str, str], tags: Mapping[str, str]) -> str:
        sb = modal.Sandbox.create(
            "/usr/bin/python3",
            "/opt/ci/supervisor.py",
            app=self.app,
            image=sandbox_image,
            env=dict(env),
            # (request, limit): a bare number is only the request, and a job could burst past it into a bill
            # the ledger never priced. Equal limits make the reservation a true upper bound (OOM, not overage).
            cpu=(profile.cpu, profile.cpu),
            memory=(int(profile.memory_gib * 1024),) * 2,
            timeout=profile.hard_timeout_s,
            name=name,
        )
        try:
            sb.set_tags(dict(tags))  # operator convenience only; never worth losing a launched sandbox over
        except Exception:
            logging.getLogger("modal_ci").exception("could not tag sandbox %s", sb.object_id)
        return sb.object_id

    def terminate(self, sandbox_id: str) -> None:
        modal.Sandbox.from_id(sandbox_id).terminate()

    def running(self, sandbox_id: str) -> bool:
        return modal.Sandbox.from_id(sandbox_id).poll() is None

    def states(self) -> dict[str, SandboxState]:
        selftests = {sb.object_id for sb in modal.Sandbox.list(app_id=self.app.app_id, tags={"purpose": "selftest"})}
        return {
            sb.object_id: SandboxState(
                sb.object_id, finished=sb.poll() is not None, ephemeral=sb.object_id in selftests
            )
            for sb in modal.Sandbox.list(app_id=self.app.app_id)
        }


class BillingProbe:
    """Measured spend of both CI apps since `CI_BUDGET_START`, cached; a delayed cross-check on the ledger.

    Includes the controller's own cost, which the ledger cannot see, so `CI_CEILING_USD` is the whole
    CI budget. The report trails real time by an hour or more (and omits the current partial hour),
    so the ledger's reservations, not this probe, are the real-time guarantee.

    Fails closed: with no successful reading in the last `MAX_AGE_S`, the spend is unknown and
    nothing launches (`SpendUnknown`), rather than reserving against a stale or zero history.
    """

    REFRESH_S = 300
    MAX_AGE_S = 6 * 3600

    def __init__(self, since: str, clock: Callable[[], float] = time.time) -> None:
        self.since = dt.datetime.fromisoformat(since).replace(tzinfo=dt.UTC)
        self.clock = clock
        self.value = 0.0
        self.read_at: float | None = None  # when `value` was last read successfully
        self.tried_at = 0.0

    def __call__(self) -> float:
        now = self.clock()
        if self.read_at is None or now - self.tried_at >= self.REFRESH_S:
            self.tried_at = now
            try:
                rows = modal.billing.workspace_billing_report(start=self.since, resolution="h")
                self.value = sum(float(r["cost"]) for r in rows if r["description"] in (CONTROLLER_APP, RUNNER_APP))
                self.read_at = now
            except Exception:
                logging.getLogger("modal_ci").exception("billing report unavailable")
        if self.read_at is None or now - self.read_at > self.MAX_AGE_S:
            raise SpendUnknown("no recent reading of measured spend")
        return self.value


def build_core() -> Core:
    env = os.environ
    cfg = policy.Config(
        repos=frozenset(env["CI_REPOS"].split(",")),
        installation_id=int(env["CI_INSTALLATION_ID"]),
        active_until=policy.parse_deadline(env["CI_ACTIVE_UNTIL"]),
    )
    ledger = Ledger(
        modal.Dict.from_name(LEDGER_DICT, create_if_missing=True),
        Limits(
            ceiling_usd=float(env["CI_CEILING_USD"]),
            daily_cap_usd=float(env["CI_DAILY_CAP_USD"]),
            max_concurrent=int(env["CI_MAX_CONCURRENT"]),
        ),
    )
    return Core(
        cfg=cfg,
        webhook_secret=env["CI_WEBHOOK_SECRET"].encode(),
        github=GitHubApp(int(env["CI_APP_ID"]), env["CI_APP_PRIVATE_KEY"], cfg.installation_id),
        ledger=ledger,
        sandboxes=ModalSandboxes(),
        deliveries=modal.Dict.from_name(DELIVERY_DICT, create_if_missing=True),
        external_spend=BillingProbe(env["CI_BUDGET_START"]),
    )


@app.cls(
    image=controller_image,
    secrets=[modal.Secret.from_name(SECRET_NAME), modal.Secret.from_name(SCOPE_SECRET_NAME)],
    # One container: ledger writes are serialised by Core.lock, which is what makes the
    # reserve-then-launch sequence safe without Dict transactions.
    max_containers=1,
    # Shorter than the reaper's period, so the container is not kept warm (and billed) around the clock.
    scaledown_window=60,
)
@modal.concurrent(max_inputs=8)
class Controller:
    @modal.enter()
    def start(self) -> None:
        self.core = build_core()

    def _github_app(self) -> GitHubApp:
        assert isinstance(self.core.github, GitHubApp)
        return self.core.github

    @modal.asgi_app(label="softnanolab-ci")
    def web(self):  # noqa: ANN201
        return make_app(self.core)

    @modal.method()
    def selftest(self) -> dict[str, object]:
        """Operator check, callable only through an authenticated Modal client (not over HTTP).

        Starts a sandbox with an invalid binding, which the supervisor must refuse (exit 2) without
        starting a runner, and reports whether creation, exit and the billing probe work from here.
        """
        started = time.time()
        sandbox_id = self.core.sandboxes.create(
            profile=PROFILES["modal-ci"],
            name=f"selftest-{int(started)}",
            env={"CI_BINDING": "{}", "CI_JIT": "selftest"},
            tags={"purpose": "selftest"},
        )
        sandbox = modal.Sandbox.from_id(sandbox_id)
        exit_code = None
        while exit_code is None and time.time() - started < 180:
            exit_code = sandbox.poll()
            time.sleep(1)
        if exit_code is None:
            self.core.sandboxes.terminate(sandbox_id)
        try:
            spend: float | None = self.core.external_spend()
        except SpendUnknown:
            spend = None  # launches would be refused: that is what the operator needs to see
        return {
            "sandbox": sandbox_id,
            "exit_code": exit_code,
            "seconds": round(time.time() - started, 1),
            "stdout": sandbox.stdout.read(),
            "external_spend_usd": spend,
        }

    @modal.method()
    def webhook_status(self) -> dict[str, object]:
        """Operator check: the URL GitHub delivers to and how the controller answered the latest deliveries."""
        keys = ("id", "event", "action", "status", "status_code", "redelivery", "delivered_at")
        github = self._github_app()
        return {
            "repos": sorted(self.core.cfg.repos),
            "active_until": dt.datetime.fromtimestamp(self.core.cfg.active_until, dt.UTC).isoformat(),
            "expired": self.core.expired(),
            "url": github.hook_config().get("url"),
            "deliveries": [{k: d.get(k) for k in keys} for d in github.hook_deliveries()],
        }

    @modal.method()
    def redeliver(self, delivery_id: int) -> None:
        """Operator check: ask GitHub to send a past delivery again (the controller must drop a replay)."""
        self._github_app().redeliver(delivery_id)

    @modal.method()
    def reconcile(self) -> dict[str, int]:
        return self.core.reconcile()


@app.function(image=controller_image, schedule=modal.Period(minutes=5), scaledown_window=2)
def reaper() -> None:
    print(Controller().reconcile.remote())
