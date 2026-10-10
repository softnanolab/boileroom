# Private Bakeoff activation checks

These are operator-run checks, not a deployment or activation script. They spend real resources when dispatched.
Keep production routing unset, use a disposable `ci-live/validation` branch in **Bakeoff only**, and reconcile the
$30 CI / $10 daily / four-reservation limits and the separate $250 migration budget first. Save run/job IDs, runner IDs,
timestamps, ledger states and billing evidence, but never print tokens, JIT configurations, or secret values.

Copy `live-validation.yml` from this directory to that branch's `.github/workflows/modal-live-validation.yml` and push
only after the operator has deployed the reviewed controller. The template lives outside `.github` here deliberately.
It requests one positive bound runner; the unmatched job can steal that runner but cannot cause another allocation.
No production variable is needed for these explicit test labels.

## Hook overrides and mismatched-job control

1. With `intruder` disabled, run `bound`. Its workflow, job and step environments try to override the hook and identity.
   A successful `BOUND_MARKER` confirms the administrator hook still admits its actual binding despite those overrides.
   Confirm the hook ran in GitHub logs, and sandbox/registration cleanup completed.
2. Enable `intruder` in the template and run again. Scheduling order is nondeterministic: a completed `bound` job alone
   does not prove rejection. Obtain a case where GitHub assigns the bound runner to `intruder`, then verify the guard
   kills the runner, **no intruder step executes**, and `INTRUDER_MUST_NOT_RUN` never appears, including its `always()` step.
   The controller may recover the rightful bound job, up to three tries. Stop after a small budgeted number of attempts
   if scheduling does not exercise the case; report it unverified, not passed.
3. Cancel any remaining queued test jobs. Verify zero test sandboxes and runner registrations remain and every test
   ledger record is settled. Do not remove unrelated runners.

The template's step uses Python, avoiding shell startup from the intentionally hostile `BASH_ENV`. It prints no env
dump. Repeat after relevant runner-version/guard changes. Public-fork validation remains required before any future
public rollout; Bakeoff currently is private with forks disabled.

## Replay

Record the queued delivery ID of a completed positive job and its ledger `tries`, runner registrations, and sandbox count.
Use the existing authenticated `Controller.redeliver(delivery_id)` operator method or the GitHub App's Advanced page.
Confirm the same delivery is acknowledged as duplicate and no reservation, JIT registration, sandbox or retry is created.
A new delivery for an already completed job should instead be rejected by the GitHub job-status check; record separately.

## Tiny-budget refusal

Drain and stop the controller before changing configuration. Use a temporary scope overlay containing a deliberately
tiny positive CI ceiling and daily cap below one reservation, retaining Bakeoff-only scope, expiry and concurrency.
Preserve the original overlay values for exact restoration. Deploy, push one positive test job, and verify
`budget_exhausted`, no new runner registration, no sandbox, and no launch reservation. Cancel the waiting job.
Stop, restore the original overlay, and redeploy. Do not alter credentials or increase any budget. The controller and
reaper themselves still cost money during this test; refusing a sandbox does not prove a workspace spending cap.

## Cold start and missed delivery

For a cold request, wait until Modal reports no active controller container, then queue one positive job. Record webhook
status, queue-to-start time, and cleanup. A webhook timeout is not necessarily a lost job: verify its actual outcome.

For a deliberately missed request, stop the drained controller while routing remains unset, then queue one positive
explicitly labelled test job. Confirm GitHub could not deliver its queued event. Redeploy the same reviewed controller
without redelivering that event. Its five-minute sweep must discover and run the queued job. Record recovery timing,
`tries`, and cleanup. Do not lengthen the sweep or skip empty-ledger scans as part of this test.

## Cost and cutoff record

Benchmark the exact production commands and commit, keeping test coverage equal to hosted CI. Price the complete
sandbox lifetime, creation through termination, and include retries. Reconcile attributable controller/reaper, build,
storage and egress costs when billing catches up. The ledger's flat $0.08 egress allowance is not measured egress.

Calculate daily cost at a realistic run frequency versus the paid GitHub cost of the same completed jobs, rounded per
job. Record included-minute and credit-expiry assumptions. A positive margin after overhead is required before enabling
production routing. Keep browser, privileged and Boileroom jobs on their existing runners.

Before activation, arrange an operator cutoff before `CI_ACTIVE_UNTIL`: unset Bakeoff routing, drain or cancel already
routed jobs, verify cleanup, and stop the controller. Also use this sequence for redeployment to avoid two controller
generations racing the shared ledger. A deadline refusal alone strands queued jobs and leaves recurring overhead.
