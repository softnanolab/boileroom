"""The GitHub App setup helper: manifest shape, callback safety, installation checks, secret hygiene."""

import json
import re
import time

import pytest

from infra.modal_ci import create_app as C
from infra.modal_ci import policy

ORG = "softnanolab"
REPOS = ["softnanolab/boileroom", "softnanolab/bakeoff"]
PEM = "-----BEGIN RSA PRIVATE KEY-----\nSECRETKEYMATERIAL\n-----END RSA PRIVATE KEY-----"
HOOK_SECRET = "whsec_very_secret"


def conversion(**overrides):
    payload = {
        "id": 123,
        "slug": "softnanolab-modal-ci",
        "owner": {"login": "softnanolab"},
        "permissions": dict(C.APP_PERMISSIONS),
        "events": ["workflow_job"],
        "pem": PEM,
        "webhook_secret": HOOK_SECRET,
        "client_secret": "never-used",
    }
    return {**payload, **overrides}


def installation(**overrides):
    base = {
        "id": 77,
        "app_id": 123,
        "account": {"login": "softnanolab"},
        "repository_selection": "selected",
        "permissions": dict(C.APP_PERMISSIONS),
        "events": ["workflow_job"],
    }
    return {**base, **overrides}


# -- manifest -------------------------------------------------------------------------------------


def test_manifest_asks_for_exactly_the_agreed_permissions_and_event() -> None:
    m = C.build_manifest(name="n", org=ORG, webhook_url="https://x/github", redirect_url="http://127.0.0.1:1/callback")
    assert m["default_permissions"] == {
        "administration": "write",
        "actions": "read",
        "pull_requests": "read",
        "metadata": "read",
    }
    assert m["default_events"] == ["workflow_job"]
    assert m["public"] is False
    assert m["hook_attributes"] == {"url": "https://x/github", "active": True}
    assert m["redirect_url"].startswith("http://127.0.0.1:")


def test_form_posts_the_manifest_to_the_org_with_the_state_and_escapes_it() -> None:
    m = C.build_manifest(name='a"b', org=ORG, webhook_url="https://x/github", redirect_url="http://127.0.0.1:1/cb")
    page = C.form_page(m, "st&te", ORG)
    assert 'action="https://github.com/organizations/softnanolab/settings/apps/new?state=st%26te"' in page
    (value,) = re.findall(r'name="manifest" value="([^"]*)"', page)
    import html

    assert json.loads(html.unescape(value)) == m


# -- callback -------------------------------------------------------------------------------------


def test_callback_accepts_one_code_with_the_right_state_only() -> None:
    cb = C.Callback(manifest={}, state="good", org=ORG)
    assert cb.handle("/callback?code=abc&state=bad")[0] == 400 and cb.code is None
    assert cb.handle("/callback?code=abc")[0] == 400
    assert cb.handle("/callback?state=good")[0] == 400
    assert cb.handle("/callback?code=abc&state=g%C3%B6od")[0] == 400  # non-ASCII must not crash the comparison
    assert cb.handle("/callback?code=abc&state=good")[0] == 200 and cb.code == "abc"
    assert cb.handle("/callback?code=other&state=good")[0] == 409 and cb.code == "abc"


def test_unknown_paths_are_404_and_root_serves_the_form() -> None:
    cb = C.Callback(manifest={"name": "n"}, state="s", org=ORG)
    assert cb.handle("/etc/passwd")[0] == 404
    status, body = cb.handle("/")
    assert status == 200 and "<form" in body


# -- credentials ----------------------------------------------------------------------------------


def test_conversion_is_accepted_when_it_matches_and_hides_secrets_in_repr() -> None:
    creds = C.credentials_from(conversion(), org=ORG)
    assert (creds.app_id, creds.slug) == (123, "softnanolab-modal-ci")
    assert PEM not in repr(creds) and HOOK_SECRET not in repr(creds)


@pytest.mark.parametrize(
    "bad",
    [
        {"owner": {"login": "mallory"}},
        {"permissions": {**C.APP_PERMISSIONS, "contents": "write"}},
        {"permissions": {**C.APP_PERMISSIONS, "administration": "read"}},
        {"events": ["workflow_job", "push"]},
        {"pem": ""},
        {"webhook_secret": None},
    ],
)
def test_conversion_that_differs_from_the_manifest_is_refused(bad) -> None:
    with pytest.raises(C.SetupError) as err:
        C.credentials_from(conversion(**bad), org=ORG)
    assert PEM not in str(err.value) and HOOK_SECRET not in str(err.value)


def test_exchange_posts_to_the_conversion_endpoint() -> None:
    seen = []

    def http(method, url, headers, body):
        seen.append((method, url))
        return 201, conversion()

    assert C.exchange_code("a/b", http)["id"] == 123
    assert seen == [("POST", "https://api.github.com/app-manifests/a%2Fb/conversions")]
    with pytest.raises(C.SetupError):
        C.exchange_code("x", lambda *a: (404, {"message": "Not Found"}))


# -- installation ---------------------------------------------------------------------------------


def problems(inst=None, repos=None, **kw):
    return C.installation_problems(
        inst or installation(), set(REPOS if repos is None else repos), org=ORG, app_id=123, allowed=REPOS, **kw
    )


def test_exactly_the_allowed_repositories_pass() -> None:
    assert problems() == []


@pytest.mark.parametrize(
    ("inst", "repos", "fragment"),
    [
        (installation(repository_selection="all"), REPOS, "all repositories"),
        (installation(), REPOS[:1], "missing"),
        (installation(), [*REPOS, "softnanolab/other"], "unexpected"),
        (installation(app_id=9), REPOS, "another App"),
        (installation(account={"login": "mallory"}), REPOS, "not 'softnanolab'"),
        (installation(permissions={**C.APP_PERMISSIONS, "contents": "write"}), REPOS, "permissions differ"),
        (installation(events=["workflow_job", "push"]), REPOS, "events differ"),
    ],
)
def test_any_deviation_is_reported(inst, repos, fragment) -> None:
    assert any(fragment in p for p in problems(inst, repos))


def test_wait_for_installation_polls_until_the_installation_is_right(monkeypatch) -> None:
    monkeypatch.setattr(C.time, "sleep", lambda s: None)

    class Api:
        creds = C.Credentials(123, "slug", PEM, HOOK_SECRET)

        def __init__(self) -> None:
            self.calls = 0

        def installations(self):
            self.calls += 1
            return [installation()] if self.calls > 1 else []

        def installation_repos(self, installation_id):
            return set(REPOS[:1]) if self.calls == 2 else set(REPOS)

    api = Api()
    said: list[str] = []
    assert C.wait_for_installation(api, org=ORG, allowed=REPOS, say=said.append, interval_s=0) == 77  # type: ignore[arg-type]
    assert said and "missing" in said[0]


def test_wait_for_installation_times_out(monkeypatch) -> None:
    monkeypatch.setattr(C.time, "sleep", lambda s: None)

    class Api:
        creds = C.Credentials(123, "slug", PEM, HOOK_SECRET)

        def installations(self):
            return []

    with pytest.raises(C.SetupError, match="timed out"):
        C.wait_for_installation(Api(), org=ORG, allowed=REPOS, say=print, timeout_s=0.01, interval_s=0)  # type: ignore[arg-type]


def test_app_api_scopes_its_installation_token_and_uses_a_jwt_for_app_calls() -> None:
    calls = []

    def http(method, url, headers, body):
        calls.append((method, url.removeprefix(C.API), headers["Authorization"], json.loads(body) if body else None))
        if url.endswith("/access_tokens"):
            return 201, {"token": "ghs_x"}
        if url.endswith("/installation/repositories?per_page=100"):
            return 200, {"repositories": [{"full_name": r} for r in REPOS]}
        return 200, [installation()]

    api = C.AppApi(
        C.Credentials(123, "slug", PEM, HOOK_SECRET), http=http, sign=lambda claims, key: f"jwt:{claims['iss']}"
    )
    assert api.installations() == [installation()]
    assert api.installation_repos(77) == set(REPOS)
    assert calls[0][2] == "Bearer jwt:123"
    assert calls[1][3] == {"permissions": {"metadata": "read"}}  # read-only token for the check
    assert calls[2][2] == "Bearer ghs_x"


# -- secret ---------------------------------------------------------------------------------------


def test_secret_values_match_what_the_controller_reads() -> None:
    values = C.secret_values(
        C.credentials_from(conversion(), org=ORG),
        installation_id=77,
        repos=REPOS,
        ceiling_usd=30,
        daily_cap_usd=10,
        max_concurrent=4,
        budget_start="2026-10-10T00:00:00",
        active_until="2026-10-31T00:00:00+00:00",
    )
    assert values == {
        "CI_APP_ID": "123",
        "CI_APP_PRIVATE_KEY": PEM,
        "CI_WEBHOOK_SECRET": HOOK_SECRET,
        "CI_INSTALLATION_ID": "77",
        "CI_REPOS": "softnanolab/boileroom,softnanolab/bakeoff",
        "CI_CEILING_USD": "30",
        "CI_DAILY_CAP_USD": "10",
        "CI_MAX_CONCURRENT": "4",
        "CI_BUDGET_START": "2026-10-10T00:00:00",
        "CI_ACTIVE_UNTIL": "2026-10-31T00:00:00+00:00",
    }
    # and the policy accepts what the controller builds from them
    cfg = policy.Config(
        repos=frozenset(values["CI_REPOS"].split(",")),
        installation_id=int(values["CI_INSTALLATION_ID"]),
        active_until=policy.parse_deadline(values["CI_ACTIVE_UNTIL"]),
    )
    assert cfg.repos == frozenset(REPOS)


def test_controller_reads_only_keys_this_helper_writes() -> None:
    """Keep the Secret's key set and `controller.build_core` in step."""
    from pathlib import Path

    source = (Path(C.__file__).parent / "controller.py").read_text()
    wanted = set(re.findall(r'env\["(CI_[A-Z_]+)"\]', source))
    values = C.secret_values(
        C.Credentials(1, "s", "k", "w"),
        installation_id=1,
        repos=REPOS,
        ceiling_usd=1,
        daily_cap_usd=1,
        max_concurrent=1,
        budget_start="x",
        active_until="x",
    )
    assert wanted == set(values)


# -- command line ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    ["http://x.modal.run/github", "https://x.modal.run/other", "https://x.modal.run/github?a=1", "https:///github"],
)
def test_webhook_url_must_be_an_https_github_endpoint(url) -> None:
    with pytest.raises(C.SetupError):
        C.check_endpoint(url, http=lambda u: (200, "ok"))


def test_endpoint_must_answer_health_ok() -> None:
    C.check_endpoint("https://x.modal.run/github", http=lambda u: (200, "ok"))
    with pytest.raises(C.SetupError, match="deployed"):
        C.check_endpoint("https://x.modal.run/github", http=lambda u: (404, ""))


def test_main_refuses_when_the_confirmation_differs_and_prints_no_secrets(capsys) -> None:
    code = C.main(
        [
            "--webhook-url",
            "https://a.modal.run/github",
            "--confirm-webhook-url",
            "https://b.modal.run/github",
            "--active-until", "2026-10-31T00:00:00+00:00",
        ]
    )
    out = capsys.readouterr()
    assert code == 1 and "differ" in out.err
    assert PEM not in out.out + out.err


def test_main_refuses_repositories_outside_the_org(capsys) -> None:
    code = C.main(
        [
            "--webhook-url",
            "https://a.modal.run/github",
            "--confirm-webhook-url",
            "https://a.modal.run/github",
            "--repos",
            "evil/repo",
            "--active-until", "2026-10-31T00:00:00+00:00",
        ]
    )
    assert code == 1 and "must belong" in capsys.readouterr().err


@pytest.mark.parametrize("deadline", ["2026-10-31T00:00:00", "tomorrow", ""])
def test_main_refuses_a_deadline_without_a_time_zone(deadline, capsys) -> None:
    url = "https://a.modal.run/github"
    code = C.main(["--webhook-url", url, "--confirm-webhook-url", url, "--active-until", deadline])
    assert code == 1 and "--active-until" in capsys.readouterr().err


def test_main_requires_a_deadline() -> None:
    url = "https://a.modal.run/github"
    with pytest.raises(SystemExit):
        C.main(["--webhook-url", url, "--confirm-webhook-url", url])


def test_full_flow_stores_credentials_before_install_and_never_prints_them(monkeypatch, capsys) -> None:
    """Drives the real local server like a browser would, with GitHub, Modal and the browser faked."""
    import html
    import threading
    import urllib.request

    written: list[dict[str, str]] = []
    opened: list[str] = []

    class Api:
        def __init__(self, creds) -> None:
            self.creds = creds

        def installations(self):
            return [installation()]

        def installation_repos(self, installation_id):
            return set(REPOS)

    monkeypatch.setattr(C, "check_endpoint", lambda url: None)
    monkeypatch.setattr(C, "exchange_code", lambda code: conversion())
    monkeypatch.setattr(C, "AppApi", Api)
    monkeypatch.setattr(C, "write_modal_secret", lambda values: written.append(dict(values)))
    monkeypatch.setattr(C.webbrowser, "open", lambda url: opened.append(url))

    def browser() -> None:
        while not opened:
            time.sleep(0.01)
        page = urllib.request.urlopen(opened[0]).read().decode()  # noqa: S310 -- local server
        action = html.unescape(re.search(r'action="([^"]+)"', page).group(1))  # type: ignore[union-attr]
        state = urllib.parse.parse_qs(urllib.parse.urlsplit(action).query)["state"][0]
        manifest = json.loads(html.unescape(re.search(r'name="manifest" value="([^"]*)"', page).group(1)))  # type: ignore[union-attr]
        urllib.request.urlopen(f"{manifest['redirect_url']}?code=abc&state={state}")  # noqa: S310

    t = threading.Thread(target=browser)
    t.start()
    url = "https://a.modal.run/github"
    assert C.main(["--webhook-url", url, "--confirm-webhook-url", url, "--active-until", "2026-10-31T00:00:00+00:00"]) == 0
    t.join()
    out = capsys.readouterr()
    assert [w["CI_INSTALLATION_ID"] for w in written] == ["0", "77"]  # usable by nothing until verified
    assert written[1]["CI_APP_PRIVATE_KEY"] == PEM
    assert PEM not in out.out + out.err and HOOK_SECRET not in out.out + out.err
