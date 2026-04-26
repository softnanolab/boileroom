"""Contract tests for Docker Hub cleanup policy planning."""

from datetime import UTC, datetime

from scripts.images import cleanup_dockerhub_tags


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
