# Modal CPU runners for GitHub Actions

GitHub Actions jobs for `softnanolab/bakeoff` can run on ephemeral
[Modal Sandboxes](https://modal.com/docs/guide/sandboxes) instead of GitHub-hosted runners. Checks, logs and branch
protection stay on GitHub; only the compute moves. The design follows
[modal-projects/runner-modal](https://github.com/modal-projects/runner-modal) but does **not** copy its credential
mounting, where the registration credential can be read by workflow code.

## Status (2026-10-10): installed, disabled, not activated

- **Nothing is activated.** No repository sets `MODAL_CI_UNPRIVILEGED` or `MODAL_CI_PRIVILEGED`; every workflow runs on
  GitHub-hosted runners, exactly as before.
- **Scope is Bakeoff only.** The GitHub App (`softnanolab-modal-ci`) is installed on `softnanolab/bakeoff` and nowhere
  else. **Boileroom stays on GitHub-hosted runners**: this change contains controller code only, no Boileroom workflow is
  routed here (the code lives in this repository because it was built alongside the Boileroom CI work).
- **The controller app is stopped** (`modal app stop softnanolab-ci-controller`); the Modal Secrets, Dicts and the GitHub
  App are intact for later use. Webhook deliveries to it fail while it is stopped, which is harmless with the variable
  unset (no job asks for a self-hosted runner).
- **This is not a production-ready claim.** The pieces below were exercised live on throwaway branches, but the
  [live validation](#live-validation-run-before-activating-a-repository) is incomplete (see its status column), the
  cold-start limitation is unresolved, and the cost case is marginal (see [Cost evidence](#cost-evidence)).
- **Rollout recommendation: do not activate** unless the cost comparison is rerun against the then-current GitHub billing
  situation and the open items are closed. Activation needs a person to set a repository variable; nothing here does it.

## How a job flows

```
GitHub workflow_job (queued) ──webhook──▶ controller (Modal, max 1 container, holds ALL credentials)
                                           1. verify HMAC over the raw body, claim the delivery id (replay = no-op)
                                           2. re-fetch the job and run from the GitHub API; check repo, event, SHA, attempt,
                                              label, "not a fork", status queued
                                           3. reserve the worst-case cost in the ledger
                                           4. mint a single-job JIT runner config (token scoped to ONE repo)
                                           5. start a Sandbox running supervisor.py with {binding, JIT config} and nothing else
                                                       │
Sandbox (root only for ~1 s)                           ▼
   supervisor.py ─ writes /run/ci/binding.json (root-owned) ─ drops to user `runner` (uid 1001, no groups, new session)
   runner ── GitHub assigns a job ── ACTIONS_RUNNER_HOOK_JOB_STARTED = job-started.sh → guard.py
                                       job matches the binding? run the workflow : kill every process of the runner user
workflow_job (completed) ──webhook──▶ controller ends the sandbox, deletes the runner, settles the cost
reaper (every 5 min) ─ ends sandboxes past their deadline or unknown to the ledger, settles finished ones,
                       relaunches queued jobs whose delivery was lost
```

Workflows opt in per job with a per-job label that names the exact job (Bakeoff's staged, inactive edit):

```yaml
runs-on: ${{ (vars.MODAL_CI_UNPRIVILEGED == 'true' && <same-repo condition>) && fromJSON(format('["self-hosted","modal-ci","job-{0}-{1}-<job id>"]', github.run_id, github.run_attempt)) || 'ubuntu-24.04' }}
```

The routing lives in Bakeoff's workflows (`validate.yml`, `pr-title.yml`, `no-forks.yml`) together with Bakeoff's
`tests/test_workflow_routing.py`, which enforces that shape for every routed job (label ends in the job id, kill switch
and hosted fallback present, fork PRs excluded, no dangerous triggers, no matrix, no `sudo`/Docker/ARM).

Routed in Bakeoff: `public-prescreen-contract`, `unit-tests`, `validate`, `title`, `same-repo`. **`browser-tests` stays
GitHub-hosted** (see [Cost evidence](#cost-evidence)).

## Threat model and where each control lives

| Threat | Control | Enforced in | Tested by |
|---|---|---|---|
| Forged or replayed webhook | HMAC-SHA256 over the raw body, constant-time; delivery id claimed atomically before any side effect | `policy.verify_signature`, `core.handle_webhook` | `test_policy`, `test_core`, `test_asgi` |
| Webhook claims something false | Job and run are re-read from the GitHub API and must agree on repo, run id, attempt, SHA, labels, status | `policy.check_job/check_run` | `test_core` |
| Fork PR or `pull_request_target` gets a runner | Only `push`, `pull_request`, `workflow_dispatch`, `schedule` from the same repository; never `pull_request_target`, `issue_comment`, `workflow_run` | `policy`, workflow condition, guard | `test_policy`, `test_core`, Bakeoff `test_workflow_routing` |
| **A mismatched job takes another job's idle runner** (GitHub matches labels as a *subset*, so `[self-hosted, modal-ci]` matches any of our runners) | Pre-job hook compares repo, run id, attempt, job key, runner name, event and SHA (for PRs: head SHA and `fork is False` from the event payload) with the binding; **any** mismatch or error kills every process of the runner user, so even `always()` steps cannot run | `sandbox/guard.py`, `job-started.sh` | `test_guard`; real Sandboxes (pilot); live negative control (passed, below) |
| Workflow code reads controller secrets | App key, installation tokens, webhook secret and Modal credentials exist only in the controller. The sandbox gets one JIT config and the binding, and its environment is an explicit allowlist | `core._launch`, `supervisor.runner_env` | `test_core`, `test_supervisor`, pilot |
| Workflow code tampers with the guard or binding | Modal ignores the Dockerfile `USER`, so the supervisor starts as root and drops to `runner` explicitly; guard, hook and binding are root-owned and not writable | `sandbox/supervisor.py`, `image.py` | pilot (live Sandbox checks write, `sudo`, `/proc/1/environ`) |
| Runner outlives its job or escapes its process group | Supervisor ends the sandbox when the runner exits, the job never arrives, or the deadline passes; `pkill -u runner`; guard refuses PID/PGID 1 and kills with `kill(-1)` first; the hook's `EXIT` trap kills on *any* exit that is not an explicit allow | `supervisor.py`, `guard.py`, `job-started.sh` | `test_supervisor`, `test_guard`, `test_hook`, pilot (`setsid` escapee killed) |
| Workflow overrides the hook or its environment (`env:` on a job or workflow, `BASH_ENV`, `GITHUB_*`) | The guard trusts only `/run/ci/binding.json` and the runner-written event payload; the runner sets the hook path and `GITHUB_*` identity itself; `env:` is applied to steps, which run after the hook. Needs a live check, see item 8 below | `guard.py`, `job-started.sh` | `test_guard`, `test_hook`; live validation 8 (not run) |
| Setuid binaries inside the sandbox | The root supervisor clears every setuid/setgid bit and refuses to start the runner if any remains (stripping at image build does not persist: a mode-only change to a base-layer file was lost, verified in a real Sandbox); `sudo` is not installed; the runner user has no groups | `supervisor.strip_setuid` | `test_supervisor`, pilot |
| Runaway spend | Reserve-before-launch ledger, concurrency and daily caps, whole-CI ceiling cross-checked against Modal's billing report. The ceiling cannot be configured above **$250** (`ledger.MAX_CEILING_USD`) and a retried job carries its earlier tries' spend (`prior_usd`) | `ledger.py`, `controller.BillingProbe` | `test_ledger`, `test_core` |
| Billing report unavailable or stale | The probe fails **closed**: with no successful reading in the last 6 h nothing launches (`503 spend unknown`) instead of assuming zero | `controller.BillingProbe`, `core._launch` | `test_controller`, `test_core` |
| A job bursts past the CPU/memory it was priced for | Sandbox CPU and memory are `(request, limit)` with equal values, so the reservation is a true upper bound (OOM instead of overage) | `controller.ModalSandboxes.create` | `test_controller` |
| Orphaned sandboxes or registrations | Reaper ends unreferenced sandboxes, past-deadline sandboxes, deletes runners, gives back stuck reservations; a sandbox is settled as gone only after a 120 s grace and a direct `running()` lookup | `core.reconcile` | `test_core` |
| Wider GitHub token than needed | Tokens are requested for one repository with one permission set and refused if GitHub returns anything wider; GitHub API calls never follow redirects (the bearer token cannot leak to another host) and any non-2xx is an error | `github_api` | `test_github_api` |
| Probing the webhook for valid runs | Runs rejected permanently (fork, wrong event, SHA mismatch) are remembered, so repeated deliveries cost no API calls | `core._check_run` | `test_core` |
| Spend after the planned end of the credits | `CI_ACTIVE_UNTIL` (see [Expiry](#expiry-is-a-spend-stop-not-a-fallback)) refuses every launch and recovery at or after the deadline | `core.expired`, `policy.parse_deadline` | `test_core`, `test_policy` |

### Accepted risks

- **A correctly bound job runs arbitrary code as `runner`** inside its own sandbox. That is the point of the sandbox, which
  holds no credential beyond that job's own `GITHUB_TOKEN`.
- **Egress** is priced into the reservation (2 GiB per job) but not capped; heavy artifact traffic shows up in the measured
  spend only after the billing lag. Real egress was not measured.
- **Administration: write** is the permission GitHub requires for repository-level JIT runners. It would let a stolen App
  key change repository settings. The key lives only in the controller Secret. Organization-level JIT runners (a runner
  group restricted to the allowed repositories, *Self-hosted runners: write* at organization scope) would avoid the
  repository permission, but widen the install to the organization, so they were **not** used without separate approval.
- **The JIT configuration is readable by workflow code through `/proc`.** It is not in the steps' environment (checked live:
  no `ACTIONS_RUNNER_INPUT_JITCONFIG` variable), but the runner's own processes (3 observed) keep it in their initial
  environment, and steps run as the same user, so `/proc/<pid>/environ` shows it. What that is worth to an attacker: it
  registers and authenticates *that one runner* (an ephemeral JIT runner GitHub deletes after its single job), it cannot
  register another runner, and it is useless once the job ends. The code able to read it is the job's own, same-repository
  workflow code, which already runs arbitrary commands on the runner. A shared registration token (the `runner-modal`
  design) would be a reusable credential; this is not. Closing the `/proc` path would mean running the steps as a different user or PID
  namespace from the listener; that is not attempted here.
- **A recovery relaunch (a second try for a job still queued after a lost delivery or a guard kill) can leave the first
  try's runner registration behind** until the reaper deletes it; the first try's spend is carried into the retry's ledger
  entry. Accepted: no registration remained after the live tests (0 runners listed at the end).
- No `NPROC` limit or `no_new_privs` is applied in the sandbox: fork bombs end at the hard CPU/memory limits and the job
  timeout, and nothing is setuid when the runner starts.
- The supervisor does not log why the guard denied a job; the denial is visible as the runner dying (the job fails with
  "runner lost communication").
- **Cold start can exceed GitHub's 10 s webhook timeout.** A first delivery after the controller scaled to zero can take
  longer than GitHub waits; GitHub then records a 500 and does not retry, yet the controller still launched the sandbox
  (observed: ledger entry created 16:56:28.7, sandbox launched 16:56:33.6, job ran). A delivery that is genuinely lost is
  recovered by the reaper (every 5 minutes), so a job can start minutes late. Not fixed: keeping a warm container would add
  a recurring cost.

### Settings recommended outside this change (not applied)

- **Bakeoff**: set the default workflow token permissions to read-only.
- **Modal**: set a workspace spend limit as the last-resort backstop under the controller's own ledger.
- **GitHub billing**: none changed. The hosted-runner spending limit is $0, so hosted jobs are currently *blocked*, not
  billed; see [Cost evidence](#cost-evidence).

## Budget

Everything the CI controller spends is bounded by one number, `CI_CEILING_USD` (default **$30**, daily `CI_DAILY_CAP_USD`
**$10**, at most `CI_MAX_CONCURRENT` **4** sandboxes). It sits inside the migration's $250 all-in budget together with the
earlier RoseTTAFold 3 testing; the workspace-wide figure is tracked separately.

- **Profile**: one profile, `modal-ci`, **0.5 CPU core / 2 GiB** (`ledger.PROFILES`). It is priced at Modal's published
  sandbox rates (`ledger.PRICE_*`): about $0.0000331 per second, about **$0.002 per minute**. A larger profile is not
  cheaper than GitHub's $0.006 per minute: 2 cores / 4 GiB is about $0.0063 per minute.
- **Per job** the ledger reserves the worst case *before* anything starts: 0.5 CPU + 2 GiB for the sandbox's hard timeout
  (3 600 s job limit + 300 s) + 180 s of startup + 2 GiB of egress ≈ **$0.215**. The reservation is replaced by the observed
  cost when the job ends, and given back if the launch fails. The 2 GiB egress allowance ($0.08) is a flat, conservative
  reservation, not a measurement.
- **A launch is refused** (`budget_exhausted`, `daily_cap`, `capacity`) if `max(settled, measured) + reserved + this job`
  would exceed the ceiling or a cap. At most 4 sandboxes can be live, so the real-time exposure is ≈ $0.86.
- **Overhead the ledger cannot see** (controller container, image builds, storage, network) is covered by `measured`: the
  controller reads Modal's billing report for the two CI apps (`softnanolab-ci-controller`, `softnanolab-ci-runners`)
  since `CI_BUDGET_START` and uses `max(settled, measured)`, so the ceiling is the *whole* CI budget, overhead included.
  The report lags real time by an hour or more and omits the current partial hour: it is an after-the-fact backstop, and
  the reservations are the real-time guarantee.
- **Controller floor**: the controller container scales down after 60 s and the reaper wakes it every 5 minutes, so an
  idle deployment costs roughly **$0.04/day** (an analytical figure from Modal's Function rates for 0.125 core / 128 MiB;
  not yet reconciled against a quiet day of the billing report). Stopping the app removes it.
- **Not enforced**: egress above the 2 GiB allowance per job. Artifact-heavy workflows would show up in `measured` only
  after the billing lag.

### Expiry is a spend stop, not a fallback

`CI_ACTIVE_UNTIL` (ISO-8601 with a time zone; currently `2026-10-31T00:00:00+00:00`, matching the end of October on which
most of the Modal credits expire) makes the controller refuse every launch and every recovery at or after that instant.
It stops *spend*. It does **not** send jobs back to GitHub-hosted runners: GitHub chooses the runner when the job is
queued, from `runs-on`, and the workflow expression language has no current-time function. A job already routed to
`[self-hosted, modal-ci, …]` simply waits with no runner (GitHub fails it after about 24 hours).

So **the only safe fallback is unsetting the repository variable** (`MODAL_CI_UNPRIVILEGED`) on or before the deadline.
Cancelling queued runs from the controller would need an extra App permission, which was deliberately not added. If nobody
will unset the variable in time, do not set it.

## Cost evidence

Estimates from measured job durations and published rates; **none of these figures is an invoice**. Hosted jobs are
billed per job rounded up to the next minute at $0.006 (Linux, 2 cores). Modal figures are sandbox seconds ×
$0.0000331/s, controller and egress excluded unless stated.

Bakeoff jobs over the observed window (2026-10-05 to 2026-10-10, 5.1 days, 356 hosted jobs; `browser-tests` stays hosted):

| Job | Runs | Hosted mean | Hosted rounded min | Hosted at $0.006/min | Modal job / sandbox time | Modal estimate per run |
|---|---|---|---|---|---|---|
| `validate` | 75 | 15.7 s | 75 | $0.45 | 41 s / 48.5 s | $0.0016 |
| `unit-tests` | 74 | 27.4 s | 74 | $0.44 | 85 s / 100 s | $0.0033 |
| `public-prescreen-contract` | 10 | 50.2 s | 13 | $0.08 | 163 s / 172 s | $0.0057 |
| `title`, `same-repo` | 73, 50 | 4–6 s | 125 | $0.75 | not measured (assumed ≈ 25 s) | ≈ $0.0008 |
| routable subtotal | | | 287 | **$1.72** (≈ $0.34/day) | | ≈ $0.52 |
| `browser-tests` (stays hosted) | 74 | 156 s | 230 | $1.38 | see below | |

- Routed jobs run about 3× slower on Modal at 0.5 core (the runner downloads Python and starts cold), but cost about
  a third of the rounded hosted minutes. On the observed volume that is about **$0.10/day of Modal compute against
  $0.34/day hosted**, before the controller floor (≈ $0.04/day) and unmeasured egress: a saving of roughly **$0.20/day at
  the paid rate**, i.e. cents per day. Break-even against the controller floor is about two five-job pushes per day. The
  volume is contest-driven and expected to fall after the challenge closes on 2026-10-12.
- **`browser-tests` does not pay off on Modal.** The real six-file browser run (`BAKEOFF_BROWSER=1`) takes 6–9 minutes
  hosted; on Modal at 0.5 core / 2 GiB its pytest step had run for over 12 minutes without finishing when the run was
  cancelled at 14 minutes total (≈ $0.028 spent, not yet in the billing report). It stays hosted. Baking Playwright's OS
  libraries into the image (done: no `sudo` is needed) made it runnable unprivileged, but not competitive.
- **Measured benchmark spend** (billing report through the 16:00 UTC hour on 2026-10-10, setup and testing mixed, *not* the
  production profile): runners $0.1257, controller $0.0144, pilot $0.0041; total ≈ **$0.144**. Later hours had not been
  reported when this was written.
- **Why the saving is not a verdict.** GitHub's included minutes for SoftNanoLab are exhausted (2,000 / 2,000) until the
  allowance resets on 1 November, and the $0 spending limit makes hosted jobs *fail to start* rather than bill. The cash
  cost of hosted CI is therefore $0 and Modal cash spend is $0 too (it consumes credits that expire on 31 October and
  30 November). What routing to Modal would buy right now is CI that runs at all, not money; the paid-rate saving above
  applies only if hosted minutes are paid for. No permanent saving is claimed.

## Operating it

All commands run from the repository root with `uv run --frozen` and the Modal CLI authenticated for the workspace that
hosts the controller.

```bash
# 1. one-time: build the sandbox image so jobs do not wait for it (re-run after editing image.py or sandbox/)
uv run --frozen python -m infra.modal_ci.build_image

# 2. one-time: create the GitHub App; credentials go straight into the Modal Secret, never to the terminal
uv run --frozen --with "PyJWT[crypto]==2.*" python -m infra.modal_ci.create_app \
    --webhook-url https://<workspace>--softnanolab-ci.modal.run/github \
    --confirm-webhook-url https://<workspace>--softnanolab-ci.modal.run/github \
    --active-until 2026-10-31T00:00:00+00:00

# 3. deploy (the controller reads the Secrets at start; redeploy after any Secret change)
modal deploy -m infra.modal_ci.controller

# 4. operator self-test: starts a sandbox with an invalid binding; the supervisor must refuse it (exit 2)
python - <<'PY'
import modal
print(modal.Cls.from_name("softnanolab-ci-controller", "Controller")().selftest.remote())
PY
```

The App is owned by SoftNanoLab, private, installed **only** on `bakeoff` (`--repos` defaults to it), with repository
*Administration: write* (just-in-time runner lifecycle), *Actions, Pull requests, Metadata: read*, and the `workflow_job`
webhook. `create_app` refuses an installation that has other repositories, other permissions or other events.

Secret `softnanolab-ci-controller` keys: `CI_APP_ID`, `CI_APP_PRIVATE_KEY`, `CI_WEBHOOK_SECRET`, `CI_INSTALLATION_ID`,
`CI_REPOS`, `CI_CEILING_USD`, `CI_DAILY_CAP_USD`, `CI_MAX_CONCURRENT`, `CI_BUDGET_START`, `CI_ACTIVE_UNTIL`
(`test_create_app` fails if the controller reads a key the helper does not write).

The live deployment additionally mounts a second, non-secret Secret, `softnanolab-ci-scope` (`CI_REPOS=softnanolab/bakeoff`,
`CI_ACTIVE_UNTIL=2026-10-31T00:00:00+00:00`), listed after the credentials so its values win. It narrowed the scope to
Bakeoff after Boileroom was removed from the installation, without touching the credentials Secret (whose `CI_REPOS` may
still name both repositories).

### Activation (not done; needs a person)

1. Close the open items in the validation table below and re-run the cost comparison against the current GitHub billing.
2. Redeploy the controller (`modal deploy -m infra.modal_ci.controller`); it is stopped.
3. Set `MODAL_CI_UNPRIVILEGED=true` (Settings → Secrets and variables → Actions → Variables) on `softnanolab/bakeoff`. The
   next run routes its same-repo `push`/`pull_request`/`schedule` jobs to Modal; fork PRs stay hosted.
4. Put an operator reminder on **unsetting the variable before `CI_ACTIVE_UNTIL`** (see above).
5. Workflows that use secrets (publishing, evaluation) need the separate `MODAL_CI_PRIVILEGED` variable and a separate
   review; they are not routed. Docker, ARM and integration-test jobs stay hosted.

**Kill switch / rollback:** delete (or set to anything but `true`) the variable. New runs go back to GitHub-hosted
immediately; jobs already queued for Modal are finished by the reaper or can be re-run. To stop the controller itself:
`modal app stop softnanolab-ci-controller` (and uninstall the App for a hard stop).

**Diagnosing a job that never started:** `modal app logs softnanolab-ci-controller` shows each delivery with its reason
(`ignored delivery=… reason=…`, `not launched … reason=budget_exhausted`, `launched …`). The GitHub App's *Advanced* tab
lists deliveries and allows a redelivery.

## Live validation (run before activating a repository)

Each item needs a real GitHub run. Status as of 2026-10-10, on Bakeoff and (before it was removed from the installation)
Boileroom:

| # | Check | Status |
|---|---|---|
| 1 | **Startup and teardown:** a routed job starts, passes, the sandbox ends, and the runner registration disappears | passed |
| 2 | **Cancellation and timeout:** cancel a running job and run one past `timeout-minutes`: the sandbox must end and the ledger entry settle | passed (also recovery relaunch, tries = 2) |
| 3 | **Privilege boundary from inside a job:** `id`, `env`, writes to `/opt/ci/*` and `/run/ci/*`, `/proc/1/environ`, `sudo`; no `CI_*`/`MODAL_*`/JIT variable in the steps' environment (the JIT config is still readable under `/proc`, see accepted risks) | passed (uid 1001, no sudo, no setuid) |
| 4 | **Negative control (mismatched job):** while a bound runner is idle, queue a job asking only for `[self-hosted, modal-ci]` with an `if: always()` marker step; the runner must be killed before any step runs | passed on both repositories (marker never appeared) |
| 5 | **Negative control (public fork):** the same with a pull request from a fork of the public repository | **not run** (no fork available; covered by `test_policy`/`test_core`/`test_guard` only) |
| 6 | **Replay and signature:** redeliver a delivery (no second sandbox); wrong signature gets 401 | signature 401 passed; **replay not run** |
| 7 | **Budget:** with a tiny `CI_CEILING_USD` the next job is refused with `budget_exhausted` and nothing starts | **not run** (unit-tested only) |
| 8 | **Environment overrides cannot defeat the guard:** `env:` at workflow, job and step level for the hook, `BASH_ENV`, `GITHUB_*`, `RUNNER_NAME` | **not run** |

Items 4, 5 and 8 are **must-pass**: do not activate a public repository until they have passed against a real runner.
Items 5 and 8 have not, so Bakeoff must not be activated on the strength of this document.

Record the run URLs and outcomes in the pull request that activates a repository.

## What stays on GitHub-hosted runners

Everything in Boileroom; in Bakeoff `browser-tests`, the publish and evaluation workflows, and anything needing root,
Docker or ARM; integration tests that fold proteins on Modal GPUs (they already spend Modal money and use secrets); and any
workflow triggered by `pull_request_target`, `workflow_run` or `issue_comment`.

## Files

| File | Role |
|---|---|
| `policy.py` | Pure admission rules (signature, payload, API cross-checks, activation deadline) |
| `ledger.py` | Reserve/settle cost ledger and limits |
| `core.py` | Webhook handling, launch, teardown, reaper sweep (Modal-independent, fully tested with fakes) |
| `github_api.py` | GitHub App client: scoped tokens, JIT config, run and job queries |
| `asgi.py` | Stdlib ASGI front door (`POST /github`, `GET /health`) |
| `controller.py` | Modal app: web endpoint, reaper cron, sandbox launcher, billing probe, `selftest`, `webhook_status` |
| `image.py`, `build_image.py` | The sandbox image (Ubuntu 24.04, pinned runner, pinned Node.js, Chromium's OS libraries, root-owned guard files) |
| `sandbox/` | `supervisor.py` (root, drops privileges), `job-started.sh` + `guard.py` (pre-job policy) |
| `create_app.py` | One-time GitHub App creation and installation check |
