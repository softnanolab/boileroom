#!/bin/bash
# ACTIONS_RUNNER_HOOK_JOB_STARTED. Fail closed: only an explicit "ALLOW" from the guard lets the
# job proceed. A denial, a crash, a missing interpreter, even a `bash -e` abort: all end by killing
# every process this user owns, because a failing hook alone would still let `always()` steps run.
# The EXIT trap is what makes that hold when the shell stops before reaching the lines below.
trap 'kill -s KILL -- -1; kill -s KILL $$' EXIT  # -1: every other process of this user; the caller is skipped
verdict=$(/usr/bin/python3 -I /opt/ci/guard.py 2>/dev/null </dev/null) || verdict=""
if [ "$verdict" = "ALLOW" ]; then
    trap - EXIT
    exit 0
fi
echo "ci-guard: ${verdict:-no verdict from guard}" >&2
exit 1
