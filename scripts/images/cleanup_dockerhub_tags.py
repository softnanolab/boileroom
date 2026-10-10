#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib import error, parse, request

import click

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boileroom.images.metadata import (  # noqa: E402
    BASE_IMAGE_SPEC,
    DEFAULT_DOCKER_REPOSITORY,
    KIT_IMAGE_DIGEST_HISTORY,
    KIT_IMAGE_DIGESTS,
    KIT_IMAGE_SPECS,
    MODEL_IMAGE_SPECS,
    normalize_docker_repository,
)
from scripts.cli_utils import CONTEXT_SETTINGS  # noqa: E402

DOCKER_HUB_API_URL = "https://hub.docker.com/v2"


@dataclass(frozen=True)
class TagInfo:
    """Tag metadata needed for retention decisions."""

    name: str
    last_updated: datetime | None
    # Content digests the tag points at: the top-level digest and, for a multi-platform index, each platform image's.
    digests: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetentionPlan:
    """Per-repository retention decision output."""

    keep_tags: tuple[str, ...]
    delete_tags: tuple[str, ...]


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse a Docker Hub timestamp string to UTC."""
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def strip_cuda_prefix(tag: str) -> str:
    """Return the unqualified tag form for policy evaluation."""
    if not tag.startswith("cuda"):
        return tag
    _, separator, suffix = tag.partition("-")
    if separator and suffix:
        return suffix
    return tag


def parse_alpha(tag: str) -> tuple[int, int, int, int] | None:
    """Parse a canonical alpha tag, for example ``0.3.1-alpha.5``."""
    prefix, separator, number = tag.partition("-alpha.")
    if separator != "-alpha.":
        return None
    parts = prefix.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts) or not number.isdigit():
        return None
    major, minor, patch = (int(part) for part in parts)
    return major, minor, patch, int(number)


def is_stable_tag(tag: str) -> bool:
    """Return whether a tag is a stable semver release."""
    normalized = tag[1:] if tag.startswith("v") else tag
    parts = normalized.split(".")
    return len(parts) == 3 and all(part.isdigit() for part in parts)


def is_sha_tag(tag: str) -> bool:
    """Return whether a tag is a temporary sha validation tag."""
    if not tag.startswith("sha-"):
        return False
    suffix = tag.removeprefix("sha-")
    return 7 <= len(suffix) <= 40 and all(char in "0123456789abcdef" for char in suffix)


def plan_tag_retention(
    tags: list[TagInfo],
    keep_alpha: int,
    sha_max_age_days: int,
    now: datetime | None = None,
    protected_digests: Collection[str] = (),
) -> RetentionPlan:
    """Compute keep/delete tags using the cleanup policy.

    A tag that points at one of ``protected_digests`` (every kit image digest a boileroom release has pinned) is always
    kept, whatever its name, so the pinned images stay tagged and Docker Hub never treats them as untagged.
    """
    current_time = (now or datetime.now(tz=UTC)).astimezone(UTC)
    cutoff = current_time - timedelta(days=sha_max_age_days)

    alpha_versions = sorted(
        {
            parsed
            for tag in tags
            if (parsed := parse_alpha(strip_cuda_prefix(tag.name))) is not None
        },
        reverse=True,
    )
    keep_alpha_versions = set(alpha_versions[:keep_alpha])
    keep: list[str] = []
    delete: list[str] = []

    for tag in tags:
        logical_tag = strip_cuda_prefix(tag.name)
        alpha_version = parse_alpha(logical_tag)
        if protected_digests and not set(tag.digests).isdisjoint(protected_digests):
            keep.append(tag.name)
            continue
        if logical_tag.startswith("buildcache-"):
            keep.append(tag.name)
            continue
        if is_stable_tag(logical_tag):
            keep.append(tag.name)
            continue
        if alpha_version is not None:
            if alpha_version in keep_alpha_versions:
                keep.append(tag.name)
            else:
                delete.append(tag.name)
            continue
        if is_sha_tag(logical_tag):
            if tag.last_updated is not None and tag.last_updated < cutoff:
                delete.append(tag.name)
            else:
                keep.append(tag.name)
            continue
        keep.append(tag.name)

    return RetentionPlan(keep_tags=tuple(sorted(keep)), delete_tags=tuple(sorted(delete)))


def normalize_namespace(docker_repository: str) -> str:
    """Return Docker Hub namespace from normalized repository string."""
    normalized = normalize_docker_repository(docker_repository)
    repository_path = normalized.removeprefix("docker.io/")
    if "/" in repository_path:
        raise ValueError("Docker Hub cleanup expects a single-segment namespace, for example docker.io/jakublala.")
    return repository_path


def dockerhub_request(
    method: str,
    url: str,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Perform a Docker Hub API request and decode JSON responses."""
    headers = {"Accept": "application/json"}
    body: bytes | None = None
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")

    req = request.Request(url, data=body, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=30) as response:
            raw = response.read()
    except (error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Docker Hub API request failed for {method} {url}: {exc}") from exc
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def dockerhub_login(username: str, token: str) -> str:
    """Return a Docker Hub access token."""
    response = dockerhub_request(
        "POST",
        f"{DOCKER_HUB_API_URL}/users/login/",
        payload={"username": username, "password": token},
    )
    if not response or "token" not in response:
        raise RuntimeError("Docker Hub login failed: missing token in response.")
    return str(response["token"])


def list_repository_tags(namespace: str, repository: str, auth_token: str) -> list[TagInfo]:
    """List all tags for a Docker Hub repository."""
    results: list[TagInfo] = []
    next_url = (
        f"{DOCKER_HUB_API_URL}/namespaces/{parse.quote(namespace)}/repositories/"
        f"{parse.quote(repository)}/tags?page_size=100"
    )
    while next_url:
        response = dockerhub_request("GET", next_url, token=auth_token)
        if response is None:
            break
        for item in response.get("results", []):
            digests = [item.get("digest")] + [image.get("digest") for image in item.get("images") or []]
            results.append(
                TagInfo(
                    name=str(item["name"]),
                    last_updated=parse_timestamp(item.get("last_updated")),
                    digests=tuple(str(digest) for digest in digests if digest),
                )
            )
        next_url = str(response["next"]) if response.get("next") else ""
    return results


def delete_repository_tag(namespace: str, repository: str, tag: str, auth_token: str) -> None:
    """Delete one Docker Hub tag."""
    url = (
        f"{DOCKER_HUB_API_URL}/namespaces/{parse.quote(namespace)}/repositories/"
        f"{parse.quote(repository)}/tags/{parse.quote(tag)}"
    )
    dockerhub_request("DELETE", url, token=auth_token)


def is_not_found(exc: BaseException) -> bool:
    """Return whether ``exc`` is a failed Docker Hub request that answered 404 (wrapped by :func:`dockerhub_request`)."""
    cause = exc.__cause__ if isinstance(exc, RuntimeError) else exc
    return isinstance(cause, error.HTTPError) and cause.code == 404


def runtime_image_names() -> tuple[str, ...]:
    """Return all boileroom runtime image repository names, the kit images included."""
    return (
        BASE_IMAGE_SPEC.image_name,
        *(spec.image_name for spec in MODEL_IMAGE_SPECS),
        *(spec.image_name for spec in KIT_IMAGE_SPECS),
    )


def protected_kit_digests(image_name: str) -> tuple[str, ...]:
    """Return every digest the cleanup must keep tagged in ``image_name``: each pin in its history and the current one.

    Parameters
    ----------
    image_name : str
        A runtime image name, for example ``boileroom-protenix-kit``.

    Returns
    -------
    tuple[str, ...]
        The historical pins in order, then the current pin if the history does not list it; empty for a stock image.
    """
    history = KIT_IMAGE_DIGEST_HISTORY.get(image_name, ())
    current = KIT_IMAGE_DIGESTS.get(image_name)
    return (*history, current) if current is not None and current not in history else tuple(history)


@click.command(context_settings=CONTEXT_SETTINGS, help="Apply retention policy to boileroom Docker Hub image tags.")
@click.option("--docker-user", default=DEFAULT_DOCKER_REPOSITORY, help="Docker Hub namespace to clean.")
@click.option(
    "--dockerhub-username",
    envvar="DOCKERHUB_USERNAME",
    required=True,
    help="Docker Hub username with permissions for the target namespace.",
)
@click.option(
    "--dockerhub-token",
    envvar="DOCKERHUB_TOKEN",
    required=True,
    help="Docker Hub token/password for the supplied username.",
)
@click.option("--keep-alpha", default=3, type=click.IntRange(min=0), show_default=True)
@click.option("--sha-max-age-days", default=7, type=click.IntRange(min=1), show_default=True)
@click.option("--dry-run", is_flag=True, help="Compute and print the cleanup plan without deleting tags.")
def cli(
    docker_user: str,
    dockerhub_username: str,
    dockerhub_token: str,
    keep_alpha: int,
    sha_max_age_days: int,
    dry_run: bool,
) -> None:
    """Run Docker Hub retention cleanup for runtime image tags.

    A kit repository in which no tag points at the digest pinned in ``KIT_IMAGE_DIGESTS`` is not pruned at all: either
    the pinned image is already untagged (and so open to Docker Hub's garbage collection) or the listing does not show
    it, and in both cases deleting tags could remove the image every installation pulls. The other repositories are
    still cleaned, then the command exits non-zero naming them, in a dry run too.

    Every digest in ``KIT_IMAGE_DIGEST_HISTORY`` is protected, not only the current pin: an installed older release
    still pulls the digest it pinned. An older pin that no tag names any more is only reported, since nothing this
    cleanup does can make it tagged again.
    """
    namespace = normalize_namespace(docker_user)
    auth_token = dockerhub_login(dockerhub_username, dockerhub_token)
    image_names = runtime_image_names()
    deleted_total = 0
    unpinned: list[str] = []

    for image_name in image_names:
        try:
            tags = list_repository_tags(namespace, image_name, auth_token)
        except RuntimeError as exc:
            # The kit images are pushed by hand, so a namespace may not have them; the stock images must exist.
            if is_not_found(exc) and image_name in KIT_IMAGE_DIGESTS:
                click.echo(f"{image_name}: repository not found in {namespace}, skipping")
                continue
            raise
        pinned_digest = KIT_IMAGE_DIGESTS.get(image_name)
        protected = protected_kit_digests(image_name)
        if pinned_digest is not None and not any(pinned_digest in tag.digests for tag in tags):
            click.echo(
                f"{image_name}: ERROR: no tag points at its pinned digest {pinned_digest}; not pruning this repository"
            )
            unpinned.append(f"{image_name}@{pinned_digest}")
            continue
        for digest in protected:
            if digest != pinned_digest and not any(digest in tag.digests for tag in tags):
                click.echo(f"{image_name}: WARNING: no tag points at the earlier pinned digest {digest}")
        plan = plan_tag_retention(
            tags, keep_alpha=keep_alpha, sha_max_age_days=sha_max_age_days, protected_digests=protected
        )
        click.echo(
            f"{image_name}: total={len(tags)} keep={len(plan.keep_tags)} delete={len(plan.delete_tags)} dry_run={dry_run}"
        )
        for tag_name in plan.delete_tags:
            if dry_run:
                click.echo(f"  DRY RUN delete {image_name}:{tag_name}")
                continue
            try:
                delete_repository_tag(namespace, image_name, tag_name, auth_token)
                click.echo(f"  Deleted {image_name}:{tag_name}")
                deleted_total += 1
            except RuntimeError as exc:
                # dockerhub_request wraps HTTP errors, so a bare HTTPError never reaches this loop.
                if is_not_found(exc):
                    click.echo(f"  Already deleted {image_name}:{tag_name}")
                    continue
                raise

    if dry_run:
        click.echo("Dry run complete.")
    else:
        click.echo(f"Cleanup complete. Deleted {deleted_total} tag(s).")
    if unpinned:
        raise click.ClickException(
            f"No tag in {namespace} points at the pinned kit image(s) {', '.join(unpinned)}, so those repositories were "
            "not pruned. Tag each pinned digest again (scripts/images/promote_image_tags.py) or update "
            "KIT_IMAGE_DIGESTS, then rerun."
        )


if __name__ == "__main__":
    cli()
