"""Contract tests for Docker Hub cleanup policy planning."""

import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error

import pytest
from click.testing import CliRunner

from boileroom.images.metadata import KIT_IMAGE_DIGEST_HISTORY, KIT_IMAGE_DIGESTS, KIT_IMAGE_SPECS
from scripts.images import cleanup_dockerhub_tags

NOW = datetime(2026, 4, 26, tzinfo=UTC)
OLD = datetime(2026, 1, 1, tzinfo=UTC)


def test_plan_tag_retention_keeps_latest_three_alpha_versions() -> None:
    """Only the newest three alpha versions should be retained."""
    tags = [
        cleanup_dockerhub_tags.TagInfo("0.3.1-alpha.1", datetime(2026, 4, 1, tzinfo=UTC)),
        cleanup_dockerhub_tags.TagInfo("cuda12.6-0.3.1-alpha.1", datetime(2026, 4, 1, tzinfo=UTC)),
        cleanup_dockerhub_tags.TagInfo("0.3.1-alpha.2", datetime(2026, 4, 8, tzinfo=UTC)),
        cleanup_dockerhub_tags.TagInfo("cuda12.6-0.3.1-alpha.2", datetime(2026, 4, 8, tzinfo=UTC)),
        cleanup_dockerhub_tags.TagInfo("0.3.1-alpha.3", datetime(2026, 4, 15, tzinfo=UTC)),
        cleanup_dockerhub_tags.TagInfo("cuda12.6-0.3.1-alpha.3", datetime(2026, 4, 15, tzinfo=UTC)),
        cleanup_dockerhub_tags.TagInfo("0.3.1-alpha.4", datetime(2026, 4, 22, tzinfo=UTC)),
        cleanup_dockerhub_tags.TagInfo("cuda12.6-0.3.1-alpha.4", datetime(2026, 4, 22, tzinfo=UTC)),
    ]

    plan = cleanup_dockerhub_tags.plan_tag_retention(
        tags,
        keep_alpha=3,
        sha_max_age_days=7,
        now=datetime(2026, 4, 26, tzinfo=UTC),
    )

    assert "0.3.1-alpha.1" in plan.delete_tags
    assert "cuda12.6-0.3.1-alpha.1" in plan.delete_tags
    assert "0.3.1-alpha.4" in plan.keep_tags
    assert "cuda12.6-0.3.1-alpha.2" in plan.keep_tags


def test_plan_tag_retention_keeps_stable_and_buildcache_and_prunes_old_sha() -> None:
    """Stable/buildcache tags stay; only stale sha tags are deleted."""
    tags = [
        cleanup_dockerhub_tags.TagInfo("0.3.0", datetime(2026, 1, 1, tzinfo=UTC)),
        cleanup_dockerhub_tags.TagInfo("cuda12.6-0.3.0", datetime(2026, 1, 1, tzinfo=UTC)),
        cleanup_dockerhub_tags.TagInfo("buildcache-cuda12.6", datetime(2026, 4, 24, tzinfo=UTC)),
        cleanup_dockerhub_tags.TagInfo("sha-aaaaaaaaaaaa", datetime(2026, 4, 10, tzinfo=UTC)),
        cleanup_dockerhub_tags.TagInfo("cuda12.6-sha-aaaaaaaaaaaa", datetime(2026, 4, 10, tzinfo=UTC)),
        cleanup_dockerhub_tags.TagInfo("sha-bbbbbbbbbbbb", datetime(2026, 4, 24, tzinfo=UTC)),
    ]

    plan = cleanup_dockerhub_tags.plan_tag_retention(
        tags,
        keep_alpha=3,
        sha_max_age_days=7,
        now=datetime(2026, 4, 26, tzinfo=UTC),
    )

    assert "sha-aaaaaaaaaaaa" in plan.delete_tags
    assert "cuda12.6-sha-aaaaaaaaaaaa" in plan.delete_tags
    assert "sha-bbbbbbbbbbbb" in plan.keep_tags
    assert "0.3.0" in plan.keep_tags
    assert "cuda12.6-0.3.0" in plan.keep_tags
    assert "buildcache-cuda12.6" in plan.keep_tags


def test_plan_tag_retention_keeps_any_tag_on_a_protected_digest() -> None:
    """The kit images are pulled by digest, so the only tag holding that digest must survive the stale-sha rule."""
    pinned = "sha256:" + "1" * 64
    tags = [
        cleanup_dockerhub_tags.TagInfo("sha-dc652b0", OLD, digests=("sha256:" + "9" * 64, pinned)),
        cleanup_dockerhub_tags.TagInfo("sha-0000000", OLD, digests=("sha256:" + "2" * 64,)),
        cleanup_dockerhub_tags.TagInfo("0.3.1-alpha.1", OLD, digests=(pinned,)),
    ]

    plan = cleanup_dockerhub_tags.plan_tag_retention(
        tags, keep_alpha=0, sha_max_age_days=7, now=NOW, protected_digests=(pinned,)
    )

    assert set(plan.keep_tags) == {"sha-dc652b0", "0.3.1-alpha.1"}
    assert list(plan.delete_tags) == ["sha-0000000"]

    unprotected = cleanup_dockerhub_tags.plan_tag_retention(tags, keep_alpha=0, sha_max_age_days=7, now=NOW)
    assert "sha-dc652b0" in unprotected.delete_tags


def test_runtime_image_names_include_the_kit_images() -> None:
    names = cleanup_dockerhub_tags.runtime_image_names()
    assert {spec.image_name for spec in KIT_IMAGE_SPECS} <= set(names)
    assert len(names) == len(set(names))


def test_list_repository_tags_reads_the_index_and_platform_digests(monkeypatch: pytest.MonkeyPatch) -> None:
    pages: dict[str, dict[str, Any]] = {
        "first": {
            "results": [
                {
                    "name": "sha-dc652b0",
                    "last_updated": "2026-10-03T12:00:00.000000Z",
                    "digest": "sha256:index",
                    "images": [{"digest": "sha256:amd64"}, {"digest": None}],
                },
                {"name": "buildcache", "last_updated": None, "images": None},
            ],
            "next": "second",
        },
        "second": {"results": [{"name": "0.3.0", "digest": "sha256:stable"}], "next": None},
    }

    def fake_request(method: str, url: str, token: str | None = None, payload: Any = None) -> dict[str, Any]:
        return pages["second" if url == "second" else "first"]

    monkeypatch.setattr(cleanup_dockerhub_tags, "dockerhub_request", fake_request)
    tags = cleanup_dockerhub_tags.list_repository_tags("jakublala", "boileroom-esmfold2-kit", "token")

    assert [(tag.name, tag.digests) for tag in tags] == [
        ("sha-dc652b0", ("sha256:index", "sha256:amd64")),
        ("buildcache", ()),
        ("0.3.0", ("sha256:stable",)),
    ]


def _http_error(code: int) -> RuntimeError:
    cause = error.HTTPError("https://hub.docker.com", code, "status", {}, None)  # type: ignore[arg-type]
    wrapped = RuntimeError(f"Docker Hub API request failed: HTTP {code}")
    wrapped.__cause__ = cause
    return wrapped


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    tags_by_image: dict[str, Any],
    deleted: list[str],
    delete_error: Exception | None = None,
    extra_args: list[str] | None = None,
) -> Any:
    def list_tags(namespace: str, image_name: str, auth_token: str) -> list[cleanup_dockerhub_tags.TagInfo]:
        # A kit repository the test does not name does not exist; one with no tags would hold no pinned digest.
        tags = tags_by_image.get(image_name, _http_error(404) if image_name in KIT_IMAGE_DIGESTS else [])
        if isinstance(tags, Exception):
            raise tags
        return tags

    def delete_tag(namespace: str, image_name: str, tag: str, auth_token: str) -> None:
        if delete_error is not None:
            raise delete_error
        deleted.append(f"{image_name}:{tag}")

    monkeypatch.setattr(cleanup_dockerhub_tags, "dockerhub_login", lambda username, token: "token")
    monkeypatch.setattr(cleanup_dockerhub_tags, "list_repository_tags", list_tags)
    monkeypatch.setattr(cleanup_dockerhub_tags, "delete_repository_tag", delete_tag)
    monkeypatch.setattr(cleanup_dockerhub_tags, "datetime", _FixedDatetime)
    return CliRunner().invoke(
        cleanup_dockerhub_tags.cli, ["--dockerhub-username", "user", "--dockerhub-token", "secret", *(extra_args or [])]
    )


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
        return NOW


def test_cli_keeps_the_pinned_kit_tag_and_prunes_other_stale_kit_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    deleted: list[str] = []
    kit_name = "boileroom-protenix-kit"
    tags = {
        kit_name: [
            cleanup_dockerhub_tags.TagInfo("sha-d3e1d62", OLD, digests=(KIT_IMAGE_DIGESTS[kit_name],)),
            *(
                cleanup_dockerhub_tags.TagInfo(f"0.4.{index}", OLD, digests=(digest,))
                for index, digest in enumerate(KIT_IMAGE_DIGEST_HISTORY[kit_name][:-1])
            ),
            cleanup_dockerhub_tags.TagInfo("sha-0000000", OLD, digests=("sha256:" + "2" * 64,)),
        ],
        "boileroom-esmfold2-kit": _http_error(404),
    }

    result = _run_cli(monkeypatch, tags, deleted)

    assert result.exit_code == 0, result.output
    assert deleted == [f"{kit_name}:sha-0000000"]
    assert "boileroom-esmfold2-kit: repository not found in jakublala, skipping" in result.output
    assert "WARNING" not in result.output


@pytest.mark.parametrize("dry_run", [False, True])
def test_cli_fails_closed_when_no_tag_holds_the_pinned_digest(monkeypatch: pytest.MonkeyPatch, dry_run: bool) -> None:
    """It used to warn, prune the kit repository anyway and exit 0, so CI never flagged an unprotected pinned image."""
    deleted: list[str] = []
    kit_name = "boileroom-protenix-kit"
    tags = {
        kit_name: [cleanup_dockerhub_tags.TagInfo("sha-0000000", OLD, digests=("sha256:" + "2" * 64,))],
        "boileroom-chai1": [cleanup_dockerhub_tags.TagInfo("sha-1111111", OLD)],
        "boileroom-esmfold2-kit": _http_error(404),
    }

    result = _run_cli(monkeypatch, tags, deleted, extra_args=["--dry-run"] if dry_run else [])

    assert result.exit_code == 1, result.output
    assert f"{kit_name}: ERROR: no tag points at its pinned digest {KIT_IMAGE_DIGESTS[kit_name]}" in result.output
    assert f"{kit_name}@{KIT_IMAGE_DIGESTS[kit_name]}" in result.output
    assert f"{kit_name}:sha-0000000" not in result.output  # not even planned for deletion
    # The other repositories are still cleaned.
    if dry_run:
        assert deleted == []
        assert "DRY RUN delete boileroom-chai1:sha-1111111" in result.output
    else:
        assert deleted == ["boileroom-chai1:sha-1111111"]


@pytest.mark.parametrize("image_name", ["boileroom-chai1", "boileroom-esmfold2-kit"])
def test_cli_fails_on_a_missing_stock_repository_or_a_kit_server_error(
    monkeypatch: pytest.MonkeyPatch, image_name: str
) -> None:
    code = 404 if image_name == "boileroom-chai1" else 500
    result = _run_cli(monkeypatch, {image_name: _http_error(code)}, [])
    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)


def test_cli_treats_a_tag_deleted_meanwhile_as_done(monkeypatch: pytest.MonkeyPatch) -> None:
    """dockerhub_request wraps HTTP errors in RuntimeError, so the old ``except HTTPError`` never matched a 404."""
    tags = {"boileroom-chai1": [cleanup_dockerhub_tags.TagInfo("sha-0000000", OLD)]}

    result = _run_cli(monkeypatch, tags, [], delete_error=_http_error(404))
    assert result.exit_code == 0, result.output
    assert "Already deleted boileroom-chai1:sha-0000000" in result.output

    result = _run_cli(monkeypatch, tags, [], delete_error=_http_error(403))
    assert result.exit_code != 0


EARLIER_PIN = "sha256:" + "4" * 64


def test_cli_keeps_an_earlier_pinned_digest_held_only_by_a_stale_sha_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """An installed older release still pulls the digest it pinned; only the current pin used to be protected."""
    kit_name = "boileroom-protenix-kit"
    monkeypatch.setattr(
        cleanup_dockerhub_tags,
        "KIT_IMAGE_DIGEST_HISTORY",
        {**KIT_IMAGE_DIGEST_HISTORY, kit_name: (EARLIER_PIN, KIT_IMAGE_DIGESTS[kit_name])},
    )
    deleted: list[str] = []
    tags = {
        kit_name: [
            cleanup_dockerhub_tags.TagInfo("0.4.1", OLD, digests=(KIT_IMAGE_DIGESTS[kit_name],)),
            cleanup_dockerhub_tags.TagInfo("sha-aaaaaaa", OLD, digests=("sha256:" + "a" * 64, EARLIER_PIN)),
            cleanup_dockerhub_tags.TagInfo("sha-0000000", OLD, digests=("sha256:" + "2" * 64,)),
        ],
    }

    result = _run_cli(monkeypatch, tags, deleted)

    assert result.exit_code == 0, result.output
    assert deleted == [f"{kit_name}:sha-0000000"]
    assert "WARNING" not in result.output


def test_cli_warns_about_an_earlier_pin_that_no_tag_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the current pin fails closed; an untagged older pin cannot be fixed by the cleanup, so it is reported."""
    kit_name = "boileroom-protenix-kit"
    monkeypatch.setattr(
        cleanup_dockerhub_tags,
        "KIT_IMAGE_DIGEST_HISTORY",
        {**KIT_IMAGE_DIGEST_HISTORY, kit_name: (EARLIER_PIN, KIT_IMAGE_DIGESTS[kit_name])},
    )
    deleted: list[str] = []
    tags = {
        kit_name: [
            cleanup_dockerhub_tags.TagInfo("0.4.1", OLD, digests=(KIT_IMAGE_DIGESTS[kit_name],)),
            cleanup_dockerhub_tags.TagInfo("sha-0000000", OLD, digests=("sha256:" + "2" * 64,)),
        ],
    }

    result = _run_cli(monkeypatch, tags, deleted)

    assert result.exit_code == 0, result.output
    assert f"{kit_name}: WARNING: no tag points at the earlier pinned digest {EARLIER_PIN}" in result.output
    assert deleted == [f"{kit_name}:sha-0000000"]


def test_protected_kit_digests_cover_the_history_and_the_current_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    for spec in KIT_IMAGE_SPECS:
        assert (
            cleanup_dockerhub_tags.protected_kit_digests(spec.image_name) == KIT_IMAGE_DIGEST_HISTORY[spec.image_name]
        )
    assert cleanup_dockerhub_tags.protected_kit_digests("boileroom-chai1") == ()
    # A current pin missing from the history is still protected.
    monkeypatch.setattr(cleanup_dockerhub_tags, "KIT_IMAGE_DIGEST_HISTORY", {"boileroom-protenix-kit": (EARLIER_PIN,)})
    assert cleanup_dockerhub_tags.protected_kit_digests("boileroom-protenix-kit") == (
        EARLIER_PIN,
        KIT_IMAGE_DIGESTS["boileroom-protenix-kit"],
    )


def test_kit_digest_history_lists_each_current_pin_once() -> None:
    assert set(KIT_IMAGE_DIGEST_HISTORY) == set(KIT_IMAGE_DIGESTS) == {spec.image_name for spec in KIT_IMAGE_SPECS}
    for image_name, digest in KIT_IMAGE_DIGESTS.items():
        history = KIT_IMAGE_DIGEST_HISTORY[image_name]
        assert history[-1] == digest
        assert len(set(history)) == len(history)
        assert all(re.fullmatch(r"sha256:[0-9a-f]{64}", entry) for entry in history)


_PIN_LINE = re.compile(r'"(?P<image>boileroom-[a-z0-9]+-kit)":\s*"(?P<digest>sha256:[0-9a-f]{64})"')


def pinned_digests_in(text: str) -> set[tuple[str, str]]:
    """Return every ``(image, digest)`` that a ``KIT_IMAGE_DIGESTS`` entry in ``text`` (a file or a diff) names."""
    return {(match["image"], match["digest"]) for match in _PIN_LINE.finditer(text)}


def test_pinned_digests_in_reads_diff_lines_but_not_history_tuples() -> None:
    old, new = "sha256:" + "1" * 64, "sha256:" + "2" * 64
    diff = f'''
-        "boileroom-protenix-kit": "{old}",
+        "boileroom-protenix-kit": "{new}",
         "boileroom-esmfold2-kit": ("{old}",),
'''
    assert pinned_digests_in(diff) == {("boileroom-protenix-kit", old), ("boileroom-protenix-kit", new)}


def test_kit_digest_history_keeps_every_digest_ever_pinned() -> None:
    """KIT_IMAGE_DIGEST_HISTORY is append-only: each pin in the git history of metadata.py must still be listed."""
    metadata_path = Path(__file__).resolve().parents[2] / "boileroom" / "images" / "metadata.py"
    pins = pinned_digests_in(metadata_path.read_text())
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    log = subprocess.run(
        ["git", "log", "-p", "--", str(metadata_path)],
        cwd=metadata_path.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if log.returncode != 0:
        pytest.skip(f"no git history: {log.stderr.strip()}")
    pins |= pinned_digests_in(log.stdout)

    missing = sorted(pin for pin in pins if pin[1] not in KIT_IMAGE_DIGEST_HISTORY.get(pin[0], ()))
    assert not missing, f"Pinned kit digests missing from KIT_IMAGE_DIGEST_HISTORY: {missing}"
