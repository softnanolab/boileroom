#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
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
) -> RetentionPlan:
    """Compute keep/delete tags using the cleanup policy."""
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
    with request.urlopen(req, timeout=30) as response:
        raw = response.read()
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
            results.append(TagInfo(name=str(item["name"]), last_updated=parse_timestamp(item.get("last_updated"))))
        next_url = str(response["next"]) if response.get("next") else ""
    return results


def delete_repository_tag(namespace: str, repository: str, tag: str, auth_token: str) -> None:
    """Delete one Docker Hub tag."""
    url = (
        f"{DOCKER_HUB_API_URL}/namespaces/{parse.quote(namespace)}/repositories/"
        f"{parse.quote(repository)}/tags/{parse.quote(tag)}"
    )
    dockerhub_request("DELETE", url, token=auth_token)


def runtime_image_names() -> tuple[str, ...]:
    """Return all boileroom runtime image repository names."""
    return (BASE_IMAGE_SPEC.image_name, *(spec.image_name for spec in MODEL_IMAGE_SPECS))


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
    """Run Docker Hub retention cleanup for runtime image tags."""
    namespace = normalize_namespace(docker_user)
    auth_token = dockerhub_login(dockerhub_username, dockerhub_token)
    image_names = runtime_image_names()
    deleted_total = 0

    for image_name in image_names:
        tags = list_repository_tags(namespace, image_name, auth_token)
        plan = plan_tag_retention(tags, keep_alpha=keep_alpha, sha_max_age_days=sha_max_age_days)
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
            except error.HTTPError as exc:
                if exc.code == 404:
                    click.echo(f"  Already deleted {image_name}:{tag_name}")
                    continue
                raise

    if dry_run:
        click.echo("Dry run complete.")
    else:
        click.echo(f"Cleanup complete. Deleted {deleted_total} tag(s).")


if __name__ == "__main__":
    cli()
