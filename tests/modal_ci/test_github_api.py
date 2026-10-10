"""The GitHub App client against a fake HTTP layer: token scoping, retries, pagination, job listing."""

import json

import pytest

from infra.modal_ci import github_api as G

REPO = "softnanolab/boileroom"


class Http:
    """Scripted responses keyed by `METHOD path-prefix`; records every request."""

    def __init__(self, routes=None) -> None:
        self.routes = routes or {}
        self.requests: list[tuple[str, str, dict, dict | None]] = []

    def __call__(self, method, url, headers, body):
        path = url.removeprefix(G.API)
        self.requests.append((method, path, dict(headers), json.loads(body) if body else None))
        for (m, prefix), response in self.routes.items():
            if m == method and path.startswith(prefix):
                result = response.pop(0) if isinstance(response, list) else response
                if isinstance(result, Exception):
                    raise result
                return result
        raise AssertionError(f"unexpected request {method} {path}")


def token_reply(permissions, repos=(REPO,)):
    return 201, {
        "token": "ghs_installation",
        "permissions": permissions,
        "repositories": [{"full_name": r} for r in repos],
    }


def run(run_id, *, event="push", head=REPO, fork=False):
    return {"id": run_id, "event": event, "head_repository": {"full_name": head, "fork": fork}}


def make(routes, **kw):
    http = Http(routes)
    app = G.GitHubApp(
        1, "pem", 42, http=http, sign=lambda claims, key: f"jwt:{claims['iss']}", clock=lambda: 1000.0, **kw
    )
    return app, http


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(G.time, "sleep", lambda s: None)


def test_tokens_are_requested_for_one_repo_with_only_the_needed_permissions() -> None:
    app, http = make(
        {("POST", "/app/installations/42/access_tokens"): token_reply({"administration": "write", "metadata": "read"})}
    )
    assert app.token(REPO, G.RUNNER_PERMISSIONS) == "ghs_installation"
    method, path, headers, body = http.requests[0]
    assert body == {"repositories": ["boileroom"], "permissions": {"administration": "write"}}
    assert headers["Authorization"] == "Bearer jwt:1"


def test_token_is_cached_until_close_to_expiry() -> None:
    app, http = make({("POST", "/app/installations/42/access_tokens"): token_reply({"administration": "write"})})
    app.token(REPO, G.RUNNER_PERMISSIONS)
    app.token(REPO, G.RUNNER_PERMISSIONS)
    assert len(http.requests) == 1


@pytest.mark.parametrize(
    "reply",
    [
        token_reply({"administration": "write", "contents": "write"}),  # extra permission
        token_reply({"administration": "read"}),  # wrong level
        token_reply({"administration": "write"}, repos=(REPO, "softnanolab/bakeoff")),  # wider than one repo
        token_reply({"administration": "write"}, repos=("softnanolab/bakeoff",)),  # different repo
    ],
)
def test_a_token_wider_or_different_than_requested_is_refused(reply) -> None:
    app, _ = make({("POST", "/app/installations/42/access_tokens"): reply})
    with pytest.raises(G.GitHubError, match="broader"):
        app.token(REPO, G.RUNNER_PERMISSIONS)


def test_server_errors_are_retried_then_raised_without_a_trailing_sleep(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(G.time, "sleep", sleeps.append)
    routes = {
        ("POST", "/app/installations/42/access_tokens"): token_reply({"actions": "read", "metadata": "read"}),
        ("GET", "/repos/softnanolab/boileroom/actions/jobs/7"): [(502, None), (502, None), (502, None)],
    }
    app, http = make(routes)
    with pytest.raises(G.GitHubError, match="HTTP 502"):
        app.get_job(REPO, 7)
    assert len([r for r in http.requests if r[0] == "GET"]) == G.ATTEMPTS
    assert len(sleeps) == G.ATTEMPTS - 1


def test_network_errors_are_retried_like_server_errors() -> None:
    routes = {
        ("POST", "/app/installations/42/access_tokens"): token_reply({"actions": "read", "metadata": "read"}),
        ("GET", "/repos/softnanolab/boileroom/actions/jobs/7"): [TimeoutError("slow"), (200, {"id": 7})],
    }
    app, _ = make(routes)
    assert app.get_job(REPO, 7) == {"id": 7}


def test_client_errors_are_not_retried() -> None:
    routes = {
        ("POST", "/app/installations/42/access_tokens"): token_reply({"actions": "read", "metadata": "read"}),
        ("GET", "/repos/softnanolab/boileroom/actions/jobs/7"): [(404, {"message": "Not Found"})],
    }
    app, http = make(routes)
    with pytest.raises(G.GitHubError) as err:
        app.get_job(REPO, 7)
    assert err.value.status == 404 and len([r for r in http.requests if r[0] == "GET"]) == 1


def test_generate_jit_sends_the_labels_and_returns_config_and_runner_id() -> None:
    routes = {
        ("POST", "/app/installations/42/access_tokens"): token_reply({"administration": "write"}),
        ("POST", "/repos/softnanolab/boileroom/actions/runners/generate-jitconfig"): (
            201,
            {"encoded_jit_config": "JIT", "runner": {"id": 9}},
        ),
    }
    app, http = make(routes)
    assert app.generate_jit(REPO, "modal-7-abc", ["self-hosted", "modal-ci"]) == ("JIT", 9)
    body = http.requests[-1][3]
    assert body["labels"] == ["self-hosted", "modal-ci"] and body["name"] == "modal-7-abc"


def test_deleting_a_runner_that_is_already_gone_is_fine_but_other_errors_are_not() -> None:
    routes = {
        ("POST", "/app/installations/42/access_tokens"): token_reply({"administration": "write"}),
        ("DELETE", "/repos/softnanolab/boileroom/actions/runners/9"): [(404, None), (403, None)],
    }
    app, _ = make(routes)
    app.delete_runner(REPO, 9)
    with pytest.raises(G.GitHubError):
        app.delete_runner(REPO, 9)


def test_queued_jobs_look_in_in_progress_runs_too_and_keep_only_queued_jobs() -> None:
    def runs(*ids):
        return 200, {"workflow_runs": [run(i) for i in ids]}

    routes = {
        ("POST", "/app/installations/42/access_tokens"): token_reply({"actions": "read", "metadata": "read"}),
        ("GET", "/repos/softnanolab/boileroom/actions/runs?status=queued"): runs(100),
        ("GET", "/repos/softnanolab/boileroom/actions/runs?status=in_progress"): runs(100, 101),
        ("GET", "/repos/softnanolab/boileroom/actions/runs/100/jobs"): (
            200,
            {"jobs": [{"id": 1, "status": "queued"}, {"id": 2, "status": "in_progress"}]},
        ),
        ("GET", "/repos/softnanolab/boileroom/actions/runs/101/jobs"): (200, {"jobs": [{"id": 3, "status": "queued"}]}),
    }
    app, http = make(routes)
    assert [j["id"] for j in app.queued_jobs(REPO)] == [1, 3]
    assert (
        len([r for r in http.requests if r[1].endswith("/jobs?filter=latest&per_page=100") and "/100/" in r[1]]) == 1
    )  # run 100 listed once


def test_queued_jobs_follow_run_pages() -> None:
    full = (200, {"workflow_runs": [run(i) for i in range(G.RUNS_PER_PAGE)]})
    tail = (200, {"workflow_runs": [run(1000)]})
    routes = {
        ("POST", "/app/installations/42/access_tokens"): token_reply({"actions": "read", "metadata": "read"}),
        ("GET", "/repos/softnanolab/boileroom/actions/runs?status=queued"): [full, tail],
        ("GET", "/repos/softnanolab/boileroom/actions/runs?status=in_progress"): (200, {"workflow_runs": []}),
        ("GET", "/repos/softnanolab/boileroom/actions/runs/"): (200, {"jobs": []}),
    }
    app, http = make(routes)
    app.queued_jobs(REPO)
    job_listings = [r for r in http.requests if "/jobs?" in r[1]]
    assert len(job_listings) == G.RUNS_PER_PAGE + 1


@pytest.mark.parametrize(
    "unwanted",
    [
        run(5, head="someone/boileroom", fork=True),  # a fork PR
        run(5, event="pull_request_target"),
        run(5, event="workflow_run"),
        {"id": 5},  # fields missing: not admissible
        {**run(5), "head_repository": None},
    ],
)
def test_queued_jobs_skips_runs_the_policy_would_reject_anyway(unwanted) -> None:
    routes = {
        ("POST", "/app/installations/42/access_tokens"): token_reply({"actions": "read", "metadata": "read"}),
        ("GET", "/repos/softnanolab/boileroom/actions/runs?status=queued"): (
            200,
            {"workflow_runs": [unwanted, run(6)]},
        ),
        ("GET", "/repos/softnanolab/boileroom/actions/runs?status=in_progress"): (200, {"workflow_runs": []}),
        ("GET", "/repos/softnanolab/boileroom/actions/runs/6/jobs"): (200, {"jobs": [{"id": 9, "status": "queued"}]}),
    }
    app, http = make(routes)
    assert [j["id"] for j in app.queued_jobs(REPO)] == [9]
    assert not [r for r in http.requests if "/runs/5/" in r[1]]


@pytest.mark.parametrize("status", [301, 302, 307])
def test_a_redirect_is_an_error_not_a_success(status) -> None:
    app, _ = make({("GET", "/repos/softnanolab/boileroom/actions/jobs/7"): (status, None)})
    app._tokens[(REPO, json.dumps(G.READ_PERMISSIONS, sort_keys=True))] = G._Token("ghs", 10**9)
    with pytest.raises(G.GitHubError) as e:
        app.get_job(REPO, 7)
    assert e.value.status == status


def test_the_real_http_layer_never_follows_a_redirect() -> None:
    """urllib would resend `Authorization` to the redirect target."""
    handler = G._NoRedirect()
    result = handler.redirect_request(None, None, 302, "Found", {}, "https://evil.example/")  # type: ignore[func-returns-value]
    assert result is None


def test_operator_calls_authenticate_as_the_app_and_never_as_an_installation() -> None:
    app, http = make(
        {
            ("GET", "/app/hook/config"): (200, {"url": "https://example.test/github", "secret": "********"}),
            ("GET", "/app/hook/deliveries"): (200, [{"id": 7}]),
            ("POST", "/app/hook/deliveries/7/attempts"): (202, None),
        }
    )
    assert app.hook_config()["url"] == "https://example.test/github"
    assert app.hook_deliveries(5) == [{"id": 7}]
    app.redeliver(7)
    assert [(m, p) for m, p, _, _ in http.requests] == [
        ("GET", "/app/hook/config"),
        ("GET", "/app/hook/deliveries?per_page=5"),
        ("POST", "/app/hook/deliveries/7/attempts"),
    ]
    assert {h["Authorization"] for _, _, h, _ in http.requests} == {"Bearer jwt:1"}
