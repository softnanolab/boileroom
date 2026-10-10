"""GitHub App client for the Modal CI controller.

This module runs only in the controller. The App private key and every installation token live
here and never reach a sandbox: a sandbox gets one single-job JIT runner configuration and nothing
else. Tokens are minted per repository with the narrowest permissions the call needs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from infra.modal_ci.policy import ALLOWED_EVENTS

API = "https://api.github.com"
Http = Callable[[str, str, Mapping[str, str], bytes | None], tuple[int, Any]]
Signer = Callable[[Mapping[str, Any], str], str]

READ_PERMISSIONS = {"actions": "read", "metadata": "read"}
RUNNER_PERMISSIONS = {"administration": "write"}
SWEEP_RUN_STATUSES = ("queued", "in_progress")  # a run is `in_progress` once any one of its jobs starts
RUNS_PER_PAGE = 50
RUN_PAGES = 2  # newest 100 live runs per status; the sweep is a safety net, not the primary path
ATTEMPTS = 3


class GitHubError(RuntimeError):
    def __init__(self, status: int, what: str) -> None:
        super().__init__(f"{what}: HTTP {status}")
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """urllib would replay the `Authorization` header on a redirect; an App token must go nowhere else."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def urllib_http(method: str, url: str, headers: Mapping[str, str], body: bytes | None) -> tuple[int, Any]:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    try:
        with _OPENER.open(req, timeout=30) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw) if raw else None
        except ValueError:
            return e.code, None


def rs256_jwt(claims: Mapping[str, Any], private_key_pem: str) -> str:
    import jwt  # PyJWT[crypto]; only the controller image has it

    return jwt.encode(dict(claims), private_key_pem, algorithm="RS256")


def _could_be_admitted(repo: str, run: Mapping[str, Any]) -> bool:
    """Cheap pre-filter on the run list so the sweep does not list jobs of runs `policy.check_run` rejects anyway."""
    head = run.get("head_repository") or {}
    return run.get("event") in ALLOWED_EVENTS and head.get("full_name") == repo and not head.get("fork")


@dataclass
class _Token:
    value: str
    expires: float


class GitHubApp:
    def __init__(
        self,
        app_id: int,
        private_key_pem: str,
        installation_id: int,
        *,
        http: Http = urllib_http,
        sign: Signer = rs256_jwt,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.app_id = app_id
        self.private_key_pem = private_key_pem
        self.installation_id = installation_id
        self.http = http
        self.sign = sign
        self.clock = clock
        self._tokens: dict[tuple[str, str], _Token] = {}

    def _request(self, method: str, path: str, token: str, body: Mapping[str, Any] | None = None, *, what: str) -> Any:
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "softnanolab-modal-ci",
        }
        data = json.dumps(body).encode() if body is not None else None
        status: int = 0
        payload: Any = None
        for attempt in range(ATTEMPTS):
            if attempt:
                time.sleep(2 ** (attempt - 1))
            try:
                status, payload = self.http(method, f"{API}{path}", headers, data)
            except OSError:  # timeouts, DNS, resets: same treatment as a 5xx
                status, payload = 0, None
                continue
            if status < 500 and status != 429:
                break
        if not 200 <= status < 300:  # redirects included: they are never followed, so never a success
            raise GitHubError(status, what)
        return payload

    def _app_jwt(self) -> str:
        now = int(self.clock())
        return self.sign({"iat": now - 60, "exp": now + 9 * 60, "iss": str(self.app_id)}, self.private_key_pem)

    def token(self, repo: str, permissions: Mapping[str, str]) -> str:
        """Installation token limited to `repo` and `permissions` (cached until near expiry)."""
        cache_key = (repo, json.dumps(permissions, sort_keys=True))
        cached = self._tokens.get(cache_key)
        if cached and cached.expires - self.clock() > 120:
            return cached.value
        payload = self._request(
            "POST",
            f"/app/installations/{self.installation_id}/access_tokens",
            self._app_jwt(),
            {"repositories": [repo.split("/", 1)[1]], "permissions": dict(permissions)},
            what="create installation token",
        )
        # GitHub echoes back what it granted (adding read-only `metadata` itself); refuse a token
        # wider than requested, in repositories or in permission level.
        granted = dict(payload.get("permissions") or {})
        if "metadata" not in permissions and granted.get("metadata") == "read":
            del granted["metadata"]
        if granted != dict(permissions) or [r["full_name"] for r in payload.get("repositories") or []] != [repo]:
            raise GitHubError(0, "installation token broader than requested")
        # `expires_at` is an ISO timestamp; a conservative fixed lifetime avoids parsing it.
        tok = _Token(payload["token"], self.clock() + 50 * 60)
        self._tokens[cache_key] = tok
        return tok.value

    def hook_config(self) -> Mapping[str, Any]:
        """Where GitHub sends deliveries (the secret comes back masked). Operator checks only."""
        return self._request("GET", "/app/hook/config", self._app_jwt(), what="get hook config")

    def hook_deliveries(self, per_page: int = 20) -> list[Mapping[str, Any]]:
        return self._request(
            "GET", f"/app/hook/deliveries?per_page={per_page}", self._app_jwt(), what="list hook deliveries"
        )

    def redeliver(self, delivery_id: int) -> None:
        self._request("POST", f"/app/hook/deliveries/{delivery_id}/attempts", self._app_jwt(), what="redeliver")

    def get_job(self, repo: str, job_id: int) -> Mapping[str, Any]:
        return self._request(
            "GET", f"/repos/{repo}/actions/jobs/{job_id}", self.token(repo, READ_PERMISSIONS), what="get job"
        )

    def get_run(self, repo: str, run_id: int) -> Mapping[str, Any]:
        return self._request(
            "GET", f"/repos/{repo}/actions/runs/{run_id}", self.token(repo, READ_PERMISSIONS), what="get run"
        )

    def queued_jobs(self, repo: str) -> list[Mapping[str, Any]]:
        """Queued jobs of live runs (used to recover lost `queued` deliveries).

        A run is `in_progress` as soon as one sibling job starts, so queued jobs hide there too.
        """
        token = self.token(repo, READ_PERMISSIONS)
        run_ids: list[int] = []
        for status in SWEEP_RUN_STATUSES:
            for page in range(1, RUN_PAGES + 1):
                runs = self._request(
                    "GET",
                    f"/repos/{repo}/actions/runs?status={status}&per_page={RUNS_PER_PAGE}&page={page}",
                    token,
                    what="list runs",
                )
                batch = runs.get("workflow_runs", [])
                run_ids.extend(r["id"] for r in batch if r["id"] not in run_ids and _could_be_admitted(repo, r))
                if len(batch) < RUNS_PER_PAGE:
                    break
        out: list[Mapping[str, Any]] = []
        for run_id in run_ids:
            jobs = self._request(
                "GET", f"/repos/{repo}/actions/runs/{run_id}/jobs?filter=latest&per_page=100", token, what="list jobs"
            )
            out.extend(j for j in jobs.get("jobs", []) if j.get("status") == "queued")
        return out

    def generate_jit(self, repo: str, name: str, labels: list[str]) -> tuple[str, int]:
        """Mint a single-job JIT runner configuration. Returns `(encoded_jit_config, runner_id)`."""
        payload = self._request(
            "POST",
            f"/repos/{repo}/actions/runners/generate-jitconfig",
            self.token(repo, RUNNER_PERMISSIONS),
            {"name": name, "runner_group_id": 1, "labels": labels, "work_folder": "_work"},
            what="generate jit config",
        )
        return payload["encoded_jit_config"], payload["runner"]["id"]

    def delete_runner(self, repo: str, runner_id: int) -> None:
        try:
            self._request(
                "DELETE",
                f"/repos/{repo}/actions/runners/{runner_id}",
                self.token(repo, RUNNER_PERMISSIONS),
                what="delete runner",
            )
        except GitHubError as e:
            if e.status != 404:  # already gone: JIT runners deregister themselves after one job
                raise
