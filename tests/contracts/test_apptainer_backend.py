"""Contract tests for the Apptainer backend's image cache, digest records, startup, refusals and provenance env.

Nothing here runs Apptainer or contacts a registry: processes, pulls and HTTP are faked.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from boileroom.backend import apptainer
from boileroom.backend.apptainer import (
    DEFAULT_STARTUP_TIMEOUT,
    STARTUP_TIMEOUT_ENV,
    ApptainerBackend,
    _ApptainerModelProxy,
    _digest_record_path,
    _get_cached_sif_path,
    _pull_pinned_image,
    _resolve_registry_digest,
    _resolve_startup_timeout,
)
from boileroom.optimization import REFUSAL_PROCESS_EXIT_CODE, OptimizationUnavailableError
from boileroom.provenance import IMAGE_REF_ENV

DIGEST = "sha256:" + "ab" * 32
OTHER_DIGEST = "sha256:" + "cd" * 32


# --- .sif cache names ---------------------------------------------------------------------------------------------


def test_tagged_reference_keeps_its_cache_name(tmp_path: Path) -> None:
    """Existing caches of tagged images stay valid."""
    path = _get_cached_sif_path("docker://docker.io/jakublala/boileroom-chai:0.3.0", tmp_path)
    assert path == tmp_path / "images" / "jakublala-boileroom-chai_0.3.0.sif"


def test_digest_reference_is_cached_under_its_digest(tmp_path: Path) -> None:
    path = _get_cached_sif_path(f"docker://docker.io/jakublala/boileroom-esmfold2-kit@{DIGEST}", tmp_path)
    assert path == tmp_path / "images" / f"jakublala-boileroom-esmfold2-kit_sha256-{'ab' * 32}.sif"


def test_different_digests_never_share_a_cache_entry(tmp_path: Path) -> None:
    """A digest, and the difference between a digest and a tag, is part of the name."""
    first = _get_cached_sif_path(f"docker://docker.io/jakublala/boileroom-esmfold2-kit@{DIGEST}", tmp_path)
    second = _get_cached_sif_path(f"docker://docker.io/jakublala/boileroom-esmfold2-kit@{OTHER_DIGEST}", tmp_path)
    tagged = _get_cached_sif_path("docker://docker.io/jakublala/boileroom-esmfold2-kit:sha-dc652b0", tmp_path)
    assert len({first, second, tagged}) == 3


def test_different_repositories_never_share_a_cache_entry(tmp_path: Path) -> None:
    """The repository namespace is part of the name, so a fork's image never reuses the upstream .sif."""
    upstream = _get_cached_sif_path("docker://docker.io/jakublala/boileroom-chai:0.3.0", tmp_path)
    fork = _get_cached_sif_path("docker://docker.io/phauglin/boileroom-chai:0.3.0", tmp_path)
    assert upstream != fork
    assert fork.name == "phauglin-boileroom-chai_0.3.0.sif"


def test_a_registry_other_than_docker_hub_is_part_of_the_name(tmp_path: Path) -> None:
    """The old name dropped the registry, so the same repository and tag on another registry reused the Hub .sif."""
    hub = _get_cached_sif_path("docker://docker.io/jakublala/boileroom-chai:0.3.0", tmp_path)
    ghcr = _get_cached_sif_path("docker://ghcr.io/jakublala/boileroom-chai:0.3.0", tmp_path)
    local = _get_cached_sif_path("docker://localhost:5000/jakublala/boileroom-chai:0.3.0", tmp_path)
    assert ghcr.name == "ghcr.io-jakublala-boileroom-chai_0.3.0.sif"
    assert local.name == "localhost_5000-jakublala-boileroom-chai_0.3.0.sif"
    assert len({hub, ghcr, local}) == 3


# --- startup timeout ----------------------------------------------------------------------------------------------


def test_startup_timeout_defaults_to_a_long_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(STARTUP_TIMEOUT_ENV, raising=False)
    assert _resolve_startup_timeout(None) == DEFAULT_STARTUP_TIMEOUT
    assert DEFAULT_STARTUP_TIMEOUT >= 1800


def test_startup_timeout_reads_the_env_and_the_argument_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STARTUP_TIMEOUT_ENV, " 3600 ")
    assert _resolve_startup_timeout(None) == 3600.0
    assert _resolve_startup_timeout(12.5) == 12.5


@pytest.mark.parametrize("raw", ["soon", "0", "-5", "inf", "nan"])
def test_invalid_startup_timeout_env_is_rejected(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(STARTUP_TIMEOUT_ENV, raw)
    with pytest.raises(ValueError, match="seconds"):
        _resolve_startup_timeout(None)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_startup_timeout_argument_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="positive, finite"):
        _resolve_startup_timeout(value)


# --- fake process and backend -------------------------------------------------------------------------------------


class _FakeProcess:
    """A server process that never exits by itself, optionally ignoring SIGTERM."""

    def __init__(self, ignores_sigterm: bool = False, exit_code: int | None = None) -> None:
        self.ignores_sigterm = ignores_sigterm
        self.returncode = exit_code
        self.events: list[str] = []

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.events.append("terminate")
        if not self.ignores_sigterm:
            self.returncode = -15

    def kill(self) -> None:
        self.events.append("kill")
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        self.events.append(f"wait({timeout})")
        if self.returncode is None:
            raise subprocess.TimeoutExpired("apptainer", timeout or 0)
        return self.returncode


def _backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_uri: str, startup_timeout: float | None = None
) -> ApptainerBackend:
    monkeypatch.setattr(apptainer, "_is_tool_available", lambda name: True)
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    return ApptainerBackend(
        "boileroom.models.esmfold2.core.ESMFold2Core",
        image_uri,
        config={},
        device="cpu",
        startup_timeout=startup_timeout,
    )


def _never_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(apptainer.httpx, "get", refuse)
    monkeypatch.setattr(apptainer.time, "sleep", lambda seconds: None)


def test_constructor_validates_the_startup_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(STARTUP_TIMEOUT_ENV, "7200")
    assert _backend(monkeypatch, tmp_path, "docker://docker.io/jakublala/boileroom-chai:0.3.0")._startup_timeout == 7200
    with pytest.raises(ValueError):
        _backend(monkeypatch, tmp_path, "docker://docker.io/jakublala/boileroom-chai:0.3.0", startup_timeout=0)


def test_health_timeout_quotes_the_log_tail_and_the_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backend = _backend(monkeypatch, tmp_path, "docker://docker.io/jakublala/boileroom-chai:0.3.0")
    backend._base_url = "http://127.0.0.1:1"
    backend._process = _FakeProcess()  # type: ignore[assignment]
    backend._log_file_path = tmp_path / "server.log"
    backend._log_file_path.write_text("".join(f"line {index}\n" for index in range(80)) + "Downloading weights 41%\n")
    _never_healthy(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        backend._wait_for_health_check(timeout=0.05)

    message = str(excinfo.value)
    assert "did not become ready within 0.05 seconds" in message
    assert STARTUP_TIMEOUT_ENV in message
    assert "Downloading weights 41%" in message
    assert "line 79" in message
    assert "line 10\n" not in message  # only the tail


def test_dead_server_quotes_the_log_tail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backend = _backend(monkeypatch, tmp_path, "docker://docker.io/jakublala/boileroom-chai:0.3.0")
    backend._base_url = "http://127.0.0.1:1"
    backend._process = _FakeProcess(exit_code=1)  # type: ignore[assignment]
    backend._log_file_path = tmp_path / "server.log"
    backend._log_file_path.write_text("ImportError: libcuda.so.1\n")
    _never_healthy(monkeypatch)

    with pytest.raises(RuntimeError, match=r"Server process died(.|\n)*ImportError: libcuda.so.1"):
        backend._wait_for_health_check(timeout=60)


@pytest.mark.parametrize("ignores_sigterm", [False, True])
def test_stop_process_terminates_then_kills(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ignores_sigterm: bool
) -> None:
    backend = _backend(monkeypatch, tmp_path, "docker://docker.io/jakublala/boileroom-chai:0.3.0")
    process = _FakeProcess(ignores_sigterm=ignores_sigterm)
    backend._process = process  # type: ignore[assignment]

    backend._stop_process()

    expected = ["terminate", "wait(10)"] + (["kill", "wait(None)"] if ignores_sigterm else [])
    assert process.events == expected
    assert backend._process is None


def _fake_startup(monkeypatch: pytest.MonkeyPatch, process: _FakeProcess) -> list[list[str]]:
    """Patch everything startup() touches outside the process: cache, architecture, port and Popen."""
    commands: list[list[str]] = []

    def popen(cmd: list[str], **kwargs: Any) -> _FakeProcess:
        commands.append(cmd)
        return process

    monkeypatch.setattr(apptainer, "_is_image_cached", lambda path: True)
    monkeypatch.setattr(apptainer, "_get_image_architecture", lambda path: None)
    monkeypatch.setattr(apptainer, "_find_available_port", lambda: 8123)
    monkeypatch.setattr(apptainer.subprocess, "Popen", popen)
    return commands


def test_startup_timeout_stops_the_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A server that never answers used to be left running (holding the GPU) after the timeout error."""
    backend = _backend(monkeypatch, tmp_path, "docker://docker.io/jakublala/boileroom-chai:0.3.0", startup_timeout=0.05)
    process = _FakeProcess(ignores_sigterm=True)
    _fake_startup(monkeypatch, process)
    _never_healthy(monkeypatch)

    with pytest.raises(RuntimeError, match="did not become ready within 0.05 seconds"):
        backend.startup()

    assert process.events == ["terminate", "wait(10)", "kill", "wait(None)"]
    assert backend._process is None
    assert backend._client is None


@pytest.mark.parametrize(
    "image_uri",
    [
        "docker://docker.io/jakublala/boileroom-chai:0.3.0",
        f"docker://docker.io/jakublala/boileroom-protenix-kit@{DIGEST}",
    ],
)
def test_startup_passes_the_image_reference_to_the_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_uri: str
) -> None:
    backend = _backend(monkeypatch, tmp_path, image_uri)
    commands = _fake_startup(monkeypatch, _FakeProcess())
    monkeypatch.setattr(ApptainerBackend, "_wait_for_health_check", lambda self, timeout: None)

    backend.startup()
    try:
        (cmd,) = commands
        env = [cmd[index + 1] for index, arg in enumerate(cmd) if arg == "--env"]
        assert f"{IMAGE_REF_ENV}={image_uri.removeprefix('docker://')}" in env
        assert str(backend._sif_path) in cmd
    finally:
        backend.shutdown()


def _image_ref(cmd: list[str]) -> str:
    (value,) = [
        cmd[index + 1] for index, arg in enumerate(cmd) if arg == "--env" and cmd[index + 1].startswith(IMAGE_REF_ENV)
    ]
    return value.split("=", 1)[1]


def _no_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("a cached image must not contact the registry")

    monkeypatch.setattr(apptainer, "_resolve_registry_digest", fail)


# --- registry digests ---------------------------------------------------------------------------------------------

TAGGED = "docker://docker.io/jakublala/boileroom-chai:0.3.0"
CHALLENGE = 'Bearer realm="https://auth.docker.io/token",service="registry.docker.io",scope="repository:x:pull"'


def _registry(requests: list[httpx.Request], manifest: Any) -> httpx.Client:
    """A Docker Hub stand-in: anonymous HEADs get a Bearer challenge, the token endpoint hands out ``t0k``."""

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "auth.docker.io":
            return httpx.Response(200, json={"token": "t0k"})
        if request.headers.get("authorization") != "Bearer t0k":
            return httpx.Response(401, headers={"www-authenticate": CHALLENGE})
        return manifest(request) if callable(manifest) else manifest

    return httpx.Client(transport=httpx.MockTransport(handle))


def test_a_tag_resolves_to_its_index_digest_with_an_anonymous_token() -> None:
    requests: list[httpx.Request] = []
    client = _registry(requests, httpx.Response(200, headers={"docker-content-digest": DIGEST}))

    assert _resolve_registry_digest(TAGGED, client) == DIGEST

    unauthenticated, token, manifest = requests
    assert manifest.method == "HEAD"
    assert str(manifest.url) == "https://registry-1.docker.io/v2/jakublala/boileroom-chai/manifests/0.3.0"
    assert "application/vnd.oci.image.index.v1+json" in manifest.headers["accept"]
    assert "application/vnd.docker.distribution.manifest.list.v2+json" in manifest.headers["accept"]
    assert token.url.params["service"] == "registry.docker.io"
    assert token.url.params["scope"] == "repository:x:pull"
    assert "authorization" not in unauthenticated.headers


def test_an_official_image_resolves_under_library() -> None:
    requests: list[httpx.Request] = []
    client = _registry(requests, httpx.Response(200, headers={"docker-content-digest": DIGEST}))

    assert _resolve_registry_digest("docker://ubuntu:22.04", client) == DIGEST
    assert requests[-1].url.path == "/v2/library/ubuntu/manifests/22.04"


def test_a_digest_reference_needs_no_registry() -> None:
    requests: list[httpx.Request] = []
    client = _registry(requests, httpx.Response(500))

    assert _resolve_registry_digest(f"docker://docker.io/jakublala/x@{DIGEST}", client) == DIGEST
    assert requests == []


def _offline(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("no network", request=request)


@pytest.mark.parametrize(
    "manifest",
    [
        httpx.Response(404, json={"errors": [{"code": "MANIFEST_UNKNOWN"}]}),
        httpx.Response(200),
        httpx.Response(200, headers={"docker-content-digest": "sha512:abc"}),
        _offline,
    ],
    ids=["unknown-tag", "no-digest-header", "not-sha256", "offline"],
)
def test_an_unresolvable_tag_gives_none(manifest: Any) -> None:
    assert _resolve_registry_digest(TAGGED, _registry([], manifest)) is None


def test_a_reference_without_a_tag_is_not_guessed() -> None:
    requests: list[httpx.Request] = []
    assert (
        _resolve_registry_digest("docker://docker.io/jakublala/boileroom-chai", _registry(requests, _offline)) is None
    )
    assert requests == []


# --- pulls and digest records -------------------------------------------------------------------------------------


def _fake_pull(monkeypatch: pytest.MonkeyPatch, resolved: str | None) -> list[str]:
    pulled: list[str] = []

    def pull(image_uri: str, sif_path: Path, log_file: Path | None = None) -> None:
        pulled.append(image_uri)
        sif_path.parent.mkdir(parents=True, exist_ok=True)
        sif_path.write_bytes(b"SIF")

    monkeypatch.setattr(apptainer, "_pull_image", pull)
    monkeypatch.setattr(apptainer, "_resolve_registry_digest", lambda image_uri: resolved)
    return pulled


def test_a_tag_is_pulled_at_its_resolved_digest_and_the_digest_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pulling the resolved digest, not the tag, makes the record match the content even if the tag moves meanwhile."""
    pulled = _fake_pull(monkeypatch, DIGEST)
    sif = _get_cached_sif_path(TAGGED, tmp_path)

    _pull_pinned_image(TAGGED, sif)

    assert pulled == [f"docker://docker.io/jakublala/boileroom-chai@{DIGEST}"]
    assert _digest_record_path(sif).read_text().strip() == DIGEST
    assert sif.name == "jakublala-boileroom-chai_0.3.0.sif"  # the cache name stays the tag's


def test_an_unresolved_tag_is_pulled_by_tag_and_a_stale_record_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pulled = _fake_pull(monkeypatch, None)
    sif = _get_cached_sif_path(TAGGED, tmp_path)
    sif.parent.mkdir(parents=True)
    _digest_record_path(sif).write_text(f"{OTHER_DIGEST}\n")

    _pull_pinned_image(TAGGED, sif)

    assert pulled == [TAGGED]
    assert not _digest_record_path(sif).exists()


def test_a_failed_pull_leaves_no_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail(image_uri: str, sif_path: Path, log_file: Path | None = None) -> None:
        raise RuntimeError("Failed to pull Apptainer image")

    monkeypatch.setattr(apptainer, "_pull_image", fail)
    monkeypatch.setattr(apptainer, "_resolve_registry_digest", lambda image_uri: DIGEST)
    sif = _get_cached_sif_path(TAGGED, tmp_path)
    sif.parent.mkdir(parents=True)
    _digest_record_path(sif).write_text(f"{OTHER_DIGEST}\n")

    with pytest.raises(RuntimeError, match="Failed to pull"):
        _pull_pinned_image(TAGGED, sif)
    assert not _digest_record_path(sif).exists()


def test_a_digest_reference_is_pulled_as_is_without_a_record(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pulled = _fake_pull(monkeypatch, None)
    _no_registry(monkeypatch)
    image_uri = f"docker://docker.io/jakublala/boileroom-protenix-kit@{DIGEST}"
    sif = _get_cached_sif_path(image_uri, tmp_path)

    _pull_pinned_image(image_uri, sif)

    assert pulled == [image_uri]
    assert not _digest_record_path(sif).exists()


def test_startup_reports_the_digest_a_fresh_pull_resolved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """BOILEROOM_IMAGE_REF used to name the tag only, which says nothing once the tag is pushed again."""
    backend = _backend(monkeypatch, tmp_path, TAGGED)
    commands = _fake_startup(monkeypatch, _FakeProcess())
    monkeypatch.setattr(apptainer, "_is_image_cached", lambda path: path.exists())
    pulled = _fake_pull(monkeypatch, DIGEST)
    monkeypatch.setattr(ApptainerBackend, "_wait_for_health_check", lambda self, timeout: None)

    backend.startup()
    try:
        assert pulled == [f"docker://docker.io/jakublala/boileroom-chai@{DIGEST}"]
        assert _image_ref(commands[0]) == f"docker.io/jakublala/boileroom-chai:0.3.0@{DIGEST}"
    finally:
        backend.shutdown()


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (f"{DIGEST}\n", f"docker.io/jakublala/boileroom-chai:0.3.0@{DIGEST}"),
        (None, "docker.io/jakublala/boileroom-chai:0.3.0"),
        ("garbage\n", "docker.io/jakublala/boileroom-chai:0.3.0"),
    ],
    ids=["recorded", "pulled-before-records", "corrupt-record"],
)
def test_startup_from_the_cache_reports_the_recorded_digest_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, record: str | None, expected: str
) -> None:
    backend = _backend(monkeypatch, tmp_path, TAGGED)
    commands = _fake_startup(monkeypatch, _FakeProcess())
    _no_registry(monkeypatch)
    monkeypatch.setattr(apptainer, "_pull_image", lambda *args, **kwargs: pytest.fail("a cached image is not pulled"))
    if record is not None:
        backend._sif_path.parent.mkdir(parents=True)
        _digest_record_path(backend._sif_path).write_text(record)
    monkeypatch.setattr(ApptainerBackend, "_wait_for_health_check", lambda self, timeout: None)

    backend.startup()
    try:
        assert _image_ref(commands[0]) == expected
    finally:
        backend.shutdown()


# --- refusals -----------------------------------------------------------------------------------------------------


def test_a_server_that_exits_with_the_refusal_code_raises_a_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A refused optimization mode used to surface as a generic RuntimeError, unlike on Modal."""
    backend = _backend(monkeypatch, tmp_path, TAGGED)
    process = _FakeProcess(exit_code=REFUSAL_PROCESS_EXIT_CODE)
    _fake_startup(monkeypatch, process)
    _never_healthy(monkeypatch)
    log = tmp_path / "models" / "logs"

    with pytest.raises(OptimizationUnavailableError, match="exit code 3") as excinfo:
        backend.startup()

    assert "refused" in str(excinfo.value)
    assert backend._process is None
    assert list(log.glob("apptainer_*.log"))


def test_a_server_that_exits_otherwise_is_not_a_refusal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    backend = _backend(monkeypatch, tmp_path, TAGGED)
    backend._base_url = "http://127.0.0.1:1"
    backend._process = _FakeProcess(exit_code=1)  # type: ignore[assignment]
    _never_healthy(monkeypatch)

    with pytest.raises(RuntimeError, match="exit code 1") as excinfo:
        backend._wait_for_health_check(timeout=60)
    assert not isinstance(excinfo.value, OptimizationUnavailableError)


def _proxy(response: httpx.Response, tmp_path: Path) -> _ApptainerModelProxy:
    client = httpx.Client(base_url="http://server", transport=httpx.MockTransport(lambda request: response))
    return _ApptainerModelProxy(client, transport_secret="secret", log_file_path=tmp_path / "server.log")


@pytest.mark.parametrize("method", ["fold", "embed"])
def test_a_refused_call_raises_a_refusal(tmp_path: Path, method: str) -> None:
    body = {"detail": "Folding failed: no kit on this GPU", "error_type": "OptimizationUnavailableError"}
    proxy = _proxy(httpx.Response(500, json=body), tmp_path)

    with pytest.raises(OptimizationUnavailableError, match="no kit on this GPU") as excinfo:
        getattr(proxy, method)("MKV")
    assert str(tmp_path / "server.log") in str(excinfo.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, json={"detail": "Folding failed: CUDA out of memory", "error_type": "OutOfMemoryError"}),
        httpx.Response(500, text="Internal Server Error"),
    ],
    ids=["other-error", "not-json"],
)
def test_other_server_errors_stay_runtime_errors(tmp_path: Path, response: httpx.Response) -> None:
    with pytest.raises(RuntimeError, match="Internal server error") as excinfo:
        _proxy(response, tmp_path).fold("MKV")
    assert not isinstance(excinfo.value, OptimizationUnavailableError)


def test_client_errors_are_raised_as_http_errors(tmp_path: Path) -> None:
    with pytest.raises(httpx.HTTPStatusError):
        _proxy(httpx.Response(422, json={"detail": "bad request"}), tmp_path).fold("MKV")
