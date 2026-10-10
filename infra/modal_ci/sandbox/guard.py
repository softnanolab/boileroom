"""Job-started hook policy. Runs inside a sandbox as the unprivileged `runner` user.

A runner is registered with the labels `modal-ci` and `job-<run>-<attempt>-<key>`, but GitHub
matches a job to a runner when the job's labels are a *subset* of the runner's. A job that asks
only for `[self-hosted, modal-ci]` -- including one from a fork PR, whose workflow file the PR
controls -- would therefore be accepted by any idle runner. This hook is the check that makes the
per-job label binding real: it runs before any workflow step, compares the job that arrived with
the one the controller bound to this sandbox, and kills the runner on any doubt.

Killing is the only safe response. A non-zero hook exit fails one step, but later `always()` steps
still execute; with every runner process gone nothing can. The default is therefore deny: any
exception, missing variable or surprising value ends in `terminate()`.

Stdlib only; it must keep working with a hostile environment, so nothing here imports from the
workspace or trusts a search path.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
from collections.abc import Mapping
from typing import Any

BINDING_PATH = "/run/ci/binding.json"
RUNNER_PGID_PATH = "/run/ci/runner.pgid"
ALLOWED_EVENTS = frozenset({"push", "pull_request", "workflow_dispatch", "schedule"})
BINDING_KEYS = ("repo", "run_id", "run_attempt", "job_key", "head_sha", "runner_name")


def _env(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if not value:
        raise ValueError(f"{name} is not set")
    return value


def check(binding: Mapping[str, Any], env: Mapping[str, str], event: Mapping[str, Any] | None) -> str | None:
    """Return a denial reason, or `None` when the arriving job is the one bound to this sandbox."""
    if any(not binding.get(k) for k in BINDING_KEYS):
        return "binding_incomplete"
    if _env(env, "GITHUB_REPOSITORY") != binding["repo"]:
        return "repo_mismatch"
    if _env(env, "GITHUB_RUN_ID") != str(binding["run_id"]):
        return "run_id_mismatch"
    if _env(env, "GITHUB_RUN_ATTEMPT") != str(binding["run_attempt"]):
        return "run_attempt_mismatch"
    if _env(env, "GITHUB_JOB") != binding["job_key"]:
        return "job_key_mismatch"
    if _env(env, "RUNNER_NAME") != binding["runner_name"]:
        return "runner_name_mismatch"

    name = _env(env, "GITHUB_EVENT_NAME")
    if name not in ALLOWED_EVENTS:
        return f"event_not_allowed:{name}"
    if name == "pull_request":
        # GITHUB_SHA is a synthetic merge commit here; the head commit and its repository come
        # from the event payload, which GitHub (not the PR author) writes.
        head = ((event or {}).get("pull_request") or {}).get("head") or {}
        if head.get("sha") != binding["head_sha"]:
            return "head_sha_mismatch"
        head_repo = head.get("repo") or {}
        if head_repo.get("full_name") != binding["repo"] or head_repo.get("fork") is not False:
            return "fork_or_unknown_head_repo"
    elif _env(env, "GITHUB_SHA") != binding["head_sha"]:
        return "sha_mismatch"
    return None


def decide(env: Mapping[str, str], binding_path: str = BINDING_PATH) -> str | None:
    """`check` plus its inputs; any failure to obtain an input is a denial, never an allow."""
    try:
        with open(binding_path, encoding="utf-8") as f:
            binding = json.load(f)
        if not isinstance(binding, dict):
            return "binding_not_an_object"
        event = None
        if env.get("GITHUB_EVENT_NAME") == "pull_request":
            with open(_env(env, "GITHUB_EVENT_PATH"), encoding="utf-8") as f:
                event = json.load(f)
        return check(binding, env, event)
    except BaseException as e:  # noqa: BLE001 -- fail closed on literally anything
        return f"guard_error:{type(e).__name__}"


def terminate(pgid_path: str = RUNNER_PGID_PATH) -> None:
    """Kill every process this user owns, then the runner's process group.

    `kill(-1)` reaches processes that left the group (setsid, double-fork); on Linux it skips
    the caller, so the group kill below is the guard's own last act. The root supervisor is not
    signalable by this user and tears the sandbox down when the runner exits.
    """
    with contextlib.suppress(OSError):
        os.kill(-1, signal.SIGKILL)
    try:
        with open(pgid_path, encoding="utf-8") as f:
            pgid = int(f.read().strip())
        # Never signal init: a bad value must not turn into "kill everything".
        if pgid > 1:
            os.killpg(pgid, signal.SIGKILL)
    except (OSError, ValueError):
        pass


def main() -> int:
    reason = decide(os.environ)
    if reason is None:
        print("ALLOW", flush=True)
        return 0
    print(f"DENY {reason}", flush=True)
    terminate()
    return 1


if __name__ == "__main__":
    sys.exit(main())
