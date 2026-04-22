"""Shared Click helpers for repo-local maintenance scripts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

import click

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

F = TypeVar("F", bound=Callable[..., Any])


def none_if_empty(values: Sequence[str]) -> list[str] | None:
    """Return a list for repeated Click options, or None when omitted."""

    return list(values) if values else None


def tag_option(function: F) -> F:
    """Add the shared image tag option."""

    return cast(
        F,
        click.option(
            "--tag",
            default=None,
            help=(
                "Tag to check. Defaults to the installed boileroom version; explicit examples include "
                "0.3.0 or cuda12.6-0.3.0."
            ),
        )(function),
    )


def cuda_version_option(help_text: str) -> Callable[[F], F]:
    """Return a reusable repeatable CUDA-version Click option decorator."""

    def decorator(function: F) -> F:
        return cast(
            F,
            click.option(
                "--cuda-version",
                "cuda_versions",
                multiple=True,
                help=help_text,
            )(function),
        )

    return decorator


def all_cuda_option(help_text: str) -> Callable[[F], F]:
    """Return the shared all-CUDA flag decorator."""

    def decorator(function: F) -> F:
        return cast(F, click.option("--all-cuda", is_flag=True, help=help_text)(function))

    return decorator


def pull_option(function: F) -> F:
    """Add the shared Docker pull flag."""

    return cast(F, click.option("--pull", is_flag=True, help="Pull images before running checks.")(function))


def cleanup_option(function: F) -> F:
    """Add the shared Docker image cleanup flag."""

    return cast(
        F,
        click.option("--cleanup", is_flag=True, help="Remove each image after checking to free disk space.")(function),
    )
