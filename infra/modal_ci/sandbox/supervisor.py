"""Sandbox entrypoint. Runs as root (Modal ignores the Dockerfile `USER`), then drops privileges.

The controller hands this process two environment variables and nothing else:

- `CI_BINDING`: JSON naming the one job this sandbox exists for (see `guard.py`).
- `CI_JIT`: the single-job JIT runner configuration.

The supervisor writes the binding where only root can change it, starts the Actions runner as the
unprivileged `runner` user in its own session, and ends the sandbox (and every process in it) as
soon as the runner exits, the job never shows up, or the deadline passes. Controller state never
reaches the runner's environment beyond the JIT configuration the runner itself needs.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import pwd
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RUN_DIR = Path("/run/ci")
RUNNER_DIR = "/home/runner/actions-runner"
HOOK = "/opt/ci/job-started.sh"
GUARD = "/opt/ci/guard.py"
RUNNER_USER = "runner"
BINDING_KEYS = ("repo", "job_id", "run_id", "run_attempt", "job_key", "head_sha", "runner_name")
POLL_S = 1.0


def log(msg: str) -> None:
    print(f"ci-supervisor: {msg}", flush=True)


def validate_binding(raw: str) -> dict[str, Any]:
    binding = json.loads(raw)
    if not isinstance(binding, dict) or any(not binding.get(k) for k in BINDING_KEYS):
        raise ValueError("incomplete binding")
    return binding


def runner_env(jit: str) -> dict[str, str]:
    """Explicit allowlist: the runner sees these variables and no others."""
    return {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": f"/home/{RUNNER_USER}",
        "USER": RUNNER_USER,
        "LANG": "C.UTF-8",
        "RUNNER_TOOL_CACHE": "/opt/hostedtoolcache",
        "AGENT_TOOLSDIRECTORY": "/opt/hostedtoolcache",
        "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT": "1",
        "ACTIONS_RUNNER_HOOK_JOB_STARTED": HOOK,
        "ACTIONS_RUNNER_INPUT_JITCONFIG": jit,
    }


def worker_running() -> bool:
    """True once the runner has accepted a job (a `Runner.Worker` process exists)."""
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if (entry / "comm").read_text().strip() == "Runner.Worker":
                    return True
            except OSError:
                continue
    return False


def runner_processes() -> list[str]:
    """`comm(pid)` of live processes still owned by the runner user (what outlived the runner). Zombies don't count."""
    uid = pwd.getpwnam(RUNNER_USER).pw_uid
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                fields = dict(line.split(":", 1) for line in (entry / "status").read_text().splitlines() if ":" in line)
            except OSError:
                continue
            if int(fields["Uid"].split()[0]) == uid and not fields["State"].strip().startswith("Z"):
                found.append(f"{fields['Name'].strip()}({entry.name})")
    return found


def kill_runner_user() -> None:
    subprocess.run(["pkill", "-KILL", "-u", RUNNER_USER], check=False)


def prepare(binding_raw: str) -> dict[str, Any]:
    binding = validate_binding(binding_raw)
    for path, mode in ((HOOK, os.X_OK), (GUARD, os.R_OK), (f"{RUNNER_DIR}/run.sh", os.X_OK)):
        if not os.access(path, mode):
            raise FileNotFoundError(path)
    RUN_DIR.mkdir(mode=0o755, exist_ok=True)
    os.chmod(RUN_DIR, 0o755)
    binding_path = RUN_DIR / "binding.json"
    binding_path.write_text(json.dumps(binding))
    os.chmod(binding_path, 0o644)  # runner can read it, only root can change it
    return binding


_CHMOD = "/bin/chmod"


def strip_setuid(root: str = "/") -> list[str]:
    """Clear every setuid/setgid bit under `root`; return the files that still have one.

    Raises if the verification scan itself fails, so a scan that could not look never reads as "none left".

    Done here rather than in the image: a mode-only change to a file inherited from a base layer does
    not survive the image build (a real Sandbox built that way still had all 14). The runner user has
    no use for `su`, `mount`, `passwd` and friends, so none of them is left to abuse.
    """
    find = ["/usr/bin/find", root, "-xdev", "-type", "f", "(", "-perm", "-4000", "-o", "-perm", "-2000", ")"]
    subprocess.run([*find, "-exec", _CHMOD, "a-s", "{}", "+"], check=False, capture_output=True)
    scan = subprocess.run(find, check=False, capture_output=True, text=True)
    if scan.returncode != 0:  # an unreadable tree is not an empty one
        raise RuntimeError(f"setuid scan failed rc={scan.returncode}: {scan.stderr.strip()[:200]}")
    return scan.stdout.splitlines()


def main(env: Mapping[str, str] | None = None) -> int:
    env = dict(os.environ if env is None else env)
    max_seconds = int(env.get("CI_MAX_SECONDS", "3600"))
    start_timeout = int(env.get("CI_START_TIMEOUT_SECONDS", "300"))
    try:
        binding = prepare(env["CI_BINDING"])
        jit = env["CI_JIT"]
        remaining = strip_setuid()
    except Exception as e:  # noqa: BLE001 -- never start a runner on a bad handoff
        log(f"refusing to start: {type(e).__name__}: {e}")
        return 2
    if remaining:
        log(f"refusing to start: setuid/setgid files remain: {remaining[:5]}")
        return 2

    # `runner_env` is an allowlist, so the controller's variables stay in this root process only.
    proc = subprocess.Popen(
        [f"{RUNNER_DIR}/run.sh"],
        cwd=RUNNER_DIR,
        env=runner_env(jit),
        user=RUNNER_USER,
        group=RUNNER_USER,
        extra_groups=[],
        start_new_session=True,  # own session and process group: killable as one unit
    )
    pgid = os.getpgid(proc.pid)
    if pgid <= 1:
        proc.kill()
        log("refusing to continue: runner process group is not killable")
        return 2
    pgid_path = RUN_DIR / "runner.pgid"
    pgid_path.write_text(str(pgid))
    os.chmod(pgid_path, 0o644)  # the guard (runner user) reads it; do not depend on this process's umask
    log(f"runner started pid={proc.pid} pgid={pgid} job={binding['job_id']} runner={binding['runner_name']}")

    started = time.monotonic()
    saw_job = False
    reason = "runner_exited"
    while proc.poll() is None:
        elapsed = time.monotonic() - started
        saw_job = saw_job or worker_running()
        if not saw_job and elapsed > start_timeout:
            reason = "job_never_started"
            break
        if elapsed > max_seconds:
            reason = "deadline"
            break
        time.sleep(POLL_S)

    if proc.poll() is None:
        os.killpg(pgid, signal.SIGKILL)
    rc = proc.wait()
    leftover = runner_processes()  # anything here outlived the runner
    kill_runner_user()
    log(
        f"done reason={reason} rc={rc} job_seen={saw_job} leftover={leftover} elapsed={time.monotonic() - started:.0f}s"
    )
    return 0 if reason == "runner_exited" else 1


if __name__ == "__main__":
    sys.exit(main())
