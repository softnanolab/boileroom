"""One-time setup of the SoftNanoLab CI GitHub App, using GitHub's manifest flow.

Run on a trusted workstation from the repository root, signed in to GitHub in a browser as an owner
of the organisation, with the Modal CLI authenticated for the workspace that hosts the controller::

    uv run --frozen --with "PyJWT[crypto]==2.*" python -m infra.modal_ci.create_app \\
        --webhook-url https://<workspace>--softnanolab-ci.modal.run/github \\
        --confirm-webhook-url https://<workspace>--softnanolab-ci.modal.run/github

What it does, in order:

1. Serves a one-shot page on 127.0.0.1 that posts the App manifest to GitHub. You approve it in the
   browser (GitHub may ask you to re-authenticate).
2. Exchanges GitHub's single-use code for the App's credentials and checks the App is exactly what
   the manifest asked for. The private key and webhook secret go straight into the Modal Secret
   `softnanolab-ci-controller`; they are never printed, logged or written to disk.
3. Waits for you to install the App on the allowed repositories (and only those), verifies that
   installation, and records its id in the same Secret.

Afterwards redeploy the controller (`modal deploy -m infra.modal_ci.controller`) so it picks the
Secret up. Interrupted before step 2 finishes? Nothing was stored: delete the App in the
organisation settings and run this again.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hmac
import html
import http.server
import json
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from infra.modal_ci.github_api import API, Http, Signer, urllib_http
from infra.modal_ci.policy import parse_deadline

SECRET_NAME = "softnanolab-ci-controller"
DEFAULT_ORG = "softnanolab"
DEFAULT_REPOS = ("softnanolab/bakeoff",)
APP_PERMISSIONS = {"administration": "write", "actions": "read", "pull_requests": "read", "metadata": "read"}
APP_EVENTS = ["workflow_job"]
WAIT_S = 15 * 60


class SetupError(RuntimeError):
    """Something is not as it must be; the message names it and never contains a credential."""


@dataclass
class Credentials:
    app_id: int
    slug: str
    private_key: str = field(repr=False)
    webhook_secret: str = field(repr=False)


# -- manifest -------------------------------------------------------------------------------------


def build_manifest(*, name: str, org: str, webhook_url: str, redirect_url: str) -> dict[str, Any]:
    return {
        "name": name,
        "url": f"https://github.com/{org}",
        "hook_attributes": {"url": webhook_url, "active": True},
        "redirect_url": redirect_url,
        "public": False,
        "default_permissions": dict(APP_PERMISSIONS),
        "default_events": list(APP_EVENTS),
    }


def form_page(manifest: Mapping[str, Any], state: str, org: str) -> str:
    action = f"https://github.com/organizations/{org}/settings/apps/new?state={urllib.parse.quote(state)}"
    return (
        "<!doctype html><meta charset=utf-8><title>Create the CI GitHub App</title>"
        f'<form method="post" action="{html.escape(action)}">'
        f'<input type="hidden" name="manifest" value="{html.escape(json.dumps(manifest))}">'
        "<p>This will create a private GitHub App owned by "
        f"<b>{html.escape(org)}</b> with the permissions and webhook shown on the next page.</p>"
        '<button type="submit">Continue to GitHub</button></form>'
    )


def exchange_code(code: str, http: Http = urllib_http) -> dict[str, Any]:
    status, payload = http(
        "POST",
        f"{API}/app-manifests/{urllib.parse.quote(code, safe='')}/conversions",
        {"Accept": "application/vnd.github+json", "User-Agent": "softnanolab-modal-ci-setup"},
        b"",
    )
    if status != 201 or not isinstance(payload, dict):
        raise SetupError(f"GitHub did not accept the manifest code (HTTP {status})")
    return payload


def credentials_from(payload: Mapping[str, Any], *, org: str) -> Credentials:
    """Check GitHub created the App we asked for, then pull out the parts the controller needs."""
    if (payload.get("owner") or {}).get("login", "").lower() != org.lower():
        raise SetupError(f"the App is owned by {(payload.get('owner') or {}).get('login')!r}, not {org!r}")
    if payload.get("permissions") != APP_PERMISSIONS:
        raise SetupError(f"the App has unexpected permissions: {payload.get('permissions')}")
    if sorted(payload.get("events") or []) != sorted(APP_EVENTS):
        raise SetupError(f"the App subscribes to unexpected events: {payload.get('events')}")
    for key in ("id", "slug", "pem", "webhook_secret"):
        if not payload.get(key):
            raise SetupError(f"GitHub's response lacks {key!r}")
    return Credentials(int(payload["id"]), str(payload["slug"]), str(payload["pem"]), str(payload["webhook_secret"]))


# -- local callback server -------------------------------------------------------------------------


class Callback:
    """The state machine behind the local server, kept free of sockets so it can be tested."""

    def __init__(self, *, manifest: Mapping[str, Any], state: str, org: str) -> None:
        self.manifest = manifest
        self.state = state
        self.org = org
        self.code: str | None = None

    def handle(self, path: str) -> tuple[int, str]:
        url = urllib.parse.urlsplit(path)
        if url.path == "/":
            return 200, form_page(self.manifest, self.state, self.org)
        if url.path == "/callback":
            query = urllib.parse.parse_qs(url.query)
            got_state = (query.get("state") or [""])[0]
            code = (query.get("code") or [""])[0]
            if not code or not hmac.compare_digest(got_state.encode(), self.state.encode()):
                return 400, "unexpected callback; start again"
            if self.code is not None:
                return 409, "already used"
            self.code = code
            return 200, "GitHub App created. Return to the terminal: next, install it on the repositories."
        return 404, "not found"


def serve_until_code(callback: Callback, server: http.server.HTTPServer, timeout_s: float = WAIT_S) -> str:
    server.timeout = 1
    deadline = time.monotonic() + timeout_s
    while callback.code is None:
        if time.monotonic() > deadline:
            raise SetupError("timed out waiting for the browser step")
        server.handle_request()
    return callback.code


def make_server(callback: Callback) -> http.server.HTTPServer:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 -- http.server's API
            status, body = callback.handle(self.path)
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 -- keep codes/states out of logs
            pass

    return http.server.HTTPServer(("127.0.0.1", 0), Handler)


# -- installation ---------------------------------------------------------------------------------


class AppApi:
    """The few App-authenticated GitHub calls setup needs."""

    def __init__(self, creds: Credentials, *, http: Http = urllib_http, sign: Signer | None = None) -> None:
        self.creds = creds
        self.http = http
        if sign is None:
            from infra.modal_ci.github_api import rs256_jwt

            sign = rs256_jwt
        self.sign = sign

    def _call(self, method: str, path: str, bearer: str, body: Mapping[str, Any] | None = None) -> Any:
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {bearer}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "softnanolab-modal-ci-setup",
        }
        status, payload = self.http(method, f"{API}{path}", headers, json.dumps(body).encode() if body else None)
        if status >= 400:
            raise SetupError(f"{method} {path}: HTTP {status}")
        return payload

    def _jwt(self) -> str:
        now = int(time.time())
        return self.sign({"iat": now - 60, "exp": now + 9 * 60, "iss": str(self.creds.app_id)}, self.creds.private_key)

    def installations(self) -> list[dict[str, Any]]:
        return list(self._call("GET", "/app/installations?per_page=100", self._jwt()))

    def installation_repos(self, installation_id: int) -> set[str]:
        token = self._call(
            "POST",
            f"/app/installations/{installation_id}/access_tokens",
            self._jwt(),
            {"permissions": {"metadata": "read"}},
        )["token"]
        repos = self._call("GET", "/installation/repositories?per_page=100", token)
        return {r["full_name"] for r in repos["repositories"]}


def installation_problems(
    installation: Mapping[str, Any], repos: set[str], *, org: str, app_id: int, allowed: Sequence[str]
) -> list[str]:
    """Everything wrong with an installation; empty means it is exactly what the controller expects."""
    problems = []
    if installation.get("app_id") != app_id:
        problems.append("installation belongs to another App")
    if (installation.get("account") or {}).get("login", "").lower() != org.lower():
        problems.append(f"installed on {(installation.get('account') or {}).get('login')!r}, not {org!r}")
    if installation.get("repository_selection") != "selected":
        problems.append("installed on all repositories; choose 'Only select repositories'")
    if installation.get("permissions") != APP_PERMISSIONS:
        problems.append(f"installation permissions differ: {installation.get('permissions')}")
    if sorted(installation.get("events") or []) != sorted(APP_EVENTS):
        problems.append(f"installation events differ: {installation.get('events')}")
    if repos != set(allowed):
        extra, missing = sorted(repos - set(allowed)), sorted(set(allowed) - repos)
        problems.append(f"repositories differ (unexpected: {extra or 'none'}; missing: {missing or 'none'})")
    return problems


def wait_for_installation(
    api: AppApi,
    *,
    org: str,
    allowed: Sequence[str],
    say: Callable[[str], None],
    timeout_s: float = WAIT_S,
    interval_s: float = 5.0,
) -> int:
    deadline = time.monotonic() + timeout_s
    last: list[str] | None = None
    while time.monotonic() < deadline:
        for installation in api.installations():
            try:
                repos = api.installation_repos(installation["id"])
            except SetupError as e:
                problems = [str(e)]
            else:
                problems = installation_problems(installation, repos, org=org, app_id=api.creds.app_id, allowed=allowed)
            if not problems:
                return int(installation["id"])
            if problems != last:
                say("installation not acceptable yet: " + "; ".join(problems))
                last = problems
        time.sleep(interval_s)
    raise SetupError("timed out waiting for a correct installation")


# -- Modal secret ---------------------------------------------------------------------------------


def secret_values(
    creds: Credentials,
    *,
    installation_id: int,
    repos: Sequence[str],
    ceiling_usd: float,
    daily_cap_usd: float,
    max_concurrent: int,
    budget_start: str,
    active_until: str,
) -> dict[str, str]:
    return {
        "CI_APP_ID": str(creds.app_id),
        "CI_APP_PRIVATE_KEY": creds.private_key,
        "CI_WEBHOOK_SECRET": creds.webhook_secret,
        "CI_INSTALLATION_ID": str(installation_id),
        "CI_REPOS": ",".join(repos),
        "CI_CEILING_USD": f"{ceiling_usd:g}",
        "CI_DAILY_CAP_USD": f"{daily_cap_usd:g}",
        "CI_MAX_CONCURRENT": str(max_concurrent),
        "CI_BUDGET_START": budget_start,
        "CI_ACTIVE_UNTIL": active_until,
    }


def write_modal_secret(values: Mapping[str, str]) -> None:
    import modal

    try:
        modal.Secret.objects.create(SECRET_NAME, dict(values))
    except modal.exception.AlreadyExistsError:
        # One server-side update preserves the existing credentials if the
        # request fails; deleting first would destroy their only saved copy.
        modal.Secret.from_name(SECRET_NAME).update(dict(values))


# -- command line ---------------------------------------------------------------------------------


def check_endpoint(webhook_url: str, http: Callable[[str], tuple[int, str]] | None = None) -> None:
    """The URL must be an https `/github` endpoint whose `/health` answers `ok`."""
    parts = urllib.parse.urlsplit(webhook_url)
    if parts.scheme != "https" or not parts.netloc or parts.path != "/github" or parts.query or parts.fragment:
        raise SetupError("webhook URL must look like https://<host>/github")
    health = f"https://{parts.netloc}/health"

    def fetch(url: str) -> tuple[int, str]:
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                return resp.status, resp.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, ""

    status, body = (http or fetch)(health)
    if (status, body) != (200, "ok"):
        raise SetupError(f"{health} did not answer 'ok' (HTTP {status}); is the controller deployed?")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    p.add_argument("--webhook-url", required=True)
    p.add_argument("--confirm-webhook-url", required=True, help="repeat the URL: the explicit confirmation")
    p.add_argument("--org", default=DEFAULT_ORG)
    p.add_argument("--repos", default=",".join(DEFAULT_REPOS), help="comma-separated owner/name list")
    p.add_argument("--app-name", default="softnanolab-modal-ci")
    p.add_argument("--ceiling-usd", type=float, default=30.0, help="whole-CI Modal budget, controller included")
    p.add_argument("--daily-cap-usd", type=float, default=10.0)
    p.add_argument("--max-concurrent", type=int, default=4)
    p.add_argument(
        "--active-until",
        required=True,
        help="ISO-8601 instant with a time zone (e.g. 2026-10-31T00:00:00+00:00) after which nothing launches",
    )
    p.add_argument("--budget-start", default=dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT00:00:00"))
    p.add_argument("--no-browser", action="store_true", help="print the URL instead of opening a browser")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    say = lambda msg: print(msg, flush=True)  # noqa: E731
    repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    try:
        if args.webhook_url != args.confirm_webhook_url:
            raise SetupError("--webhook-url and --confirm-webhook-url differ")
        if any(not r.startswith(f"{args.org}/") for r in repos):
            raise SetupError(f"all repositories must belong to {args.org}")
        try:
            parse_deadline(args.active_until)
        except ValueError as e:
            raise SetupError(f"--active-until: {e}") from e
        check_endpoint(args.webhook_url)
        say(f"webhook endpoint confirmed and answering: {args.webhook_url}")

        state = secrets.token_urlsafe(24)
        callback = Callback(manifest={}, state=state, org=args.org)
        server = make_server(callback)
        redirect = f"http://127.0.0.1:{server.server_port}/callback"
        callback.manifest = build_manifest(
            name=args.app_name, org=args.org, webhook_url=args.webhook_url, redirect_url=redirect
        )
        page = f"http://127.0.0.1:{server.server_port}/"
        say(f"open {page} in the browser where you are signed in to GitHub as an owner of {args.org}")
        if not args.no_browser:
            threading.Thread(target=webbrowser.open, args=(page,), daemon=True).start()
        code = serve_until_code(callback, server)
        server.server_close()

        creds = credentials_from(exchange_code(code), org=args.org)
        values = {
            "budget_start": args.budget_start,
            "active_until": args.active_until,
            "ceiling_usd": args.ceiling_usd,
            "daily_cap_usd": args.daily_cap_usd,
            "max_concurrent": args.max_concurrent,
        }
        write_modal_secret(secret_values(creds, installation_id=0, repos=repos, **values))  # id 0 admits nothing
        say(f"App {creds.slug!r} (id {creds.app_id}) created; credentials stored in Modal secret {SECRET_NAME!r}")
        say(f"now install it: https://github.com/apps/{creds.slug}/installations/new")
        say(f"  choose {args.org}, 'Only select repositories', then exactly: {', '.join(repos)}")

        installation_id = wait_for_installation(AppApi(creds), org=args.org, allowed=repos, say=say)
        write_modal_secret(secret_values(creds, installation_id=installation_id, repos=repos, **values))
        say(f"installation {installation_id} verified and recorded")
        say("next: modal deploy -m infra.modal_ci.controller")
    except SetupError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
