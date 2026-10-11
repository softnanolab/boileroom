"""A small Dockerfile reader for the contract tests: instructions, build stages, ARG scoping and ENV resolution.

It follows Docker's own rules where the tests depend on them: a backslash at the end of a line continues the instruction
(the backslash and the newline are removed, so a ``RUN`` body is one shell command line), comment lines inside a
continued instruction are dropped, an ``ARG`` is visible only in the stage that declares it (one declared before the
first ``FROM`` serves the ``FROM`` lines only), and an ``ENV`` instruction expands ``$NAME`` / ``${NAME}`` against the
values in effect before it. Heredocs, parser directives other than comments and the JSON (exec) form are not modelled;
the Dockerfiles in this repository use none of them in the instructions the tests read.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

_VARIABLE = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))")
_FLAGS = re.compile(r"^(?:--\S+\s+)*")


@dataclass(frozen=True)
class Instruction:
    """One Dockerfile instruction.

    Attributes
    ----------
    keyword : str
        The instruction, upper case (``RUN``, ``ENV``, ...).
    value : str
        Everything after the keyword, continuation lines joined the way Docker joins them.
    stage : int
        Index of the build stage (``FROM``) it belongs to; ``-1`` before the first ``FROM``.
    line : int
        1-based line number of the instruction's first line.
    """

    keyword: str
    value: str
    stage: int
    line: int

    @property
    def body(self) -> str:
        """Return ``value`` without leading ``--flag`` options (``RUN --mount=...``, ``COPY --from=...``)."""
        return _FLAGS.sub("", self.value, count=1)

    @property
    def flags(self) -> tuple[str, ...]:
        """Return the leading ``--flag`` options of ``value``."""
        return tuple(_FLAGS.match(self.value).group(0).split())  # type: ignore[union-attr]


def expand(text: str, values: Mapping[str, str]) -> str:
    """Expand ``$NAME`` and ``${NAME}`` from ``values``; an unknown name stays as written."""

    def substitute(match: re.Match[str]) -> str:
        name = match.group("braced") or match.group("bare")
        return values.get(name, match.group(0))

    return _VARIABLE.sub(substitute, text)


class Dockerfile:
    """A parsed Dockerfile.

    Parameters
    ----------
    path : Path
        The Dockerfile to read.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.text = path.read_text(encoding="utf-8")

    @cached_property
    def instructions(self) -> tuple[Instruction, ...]:
        """Return every instruction in file order."""
        instructions: list[Instruction] = []
        lines = self.text.splitlines()
        stage = -1
        index = 0
        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            index += 1
            if not stripped or stripped.startswith("#"):
                continue
            start = index
            parts = [line]
            while parts[-1].rstrip().endswith("\\") and index < len(lines):
                parts[-1] = parts[-1].rstrip()[:-1]
                following = lines[index]
                index += 1
                if following.strip().startswith("#"):
                    parts.append("\\")  # a comment line inside the instruction is dropped; keep continuing
                    continue
                parts.append(following)
            if parts[-1].rstrip().endswith("\\"):
                parts[-1] = parts[-1].rstrip()[:-1]
            joined = "".join(part for part in parts if part != "\\").strip()
            keyword, _, value = joined.partition(" ")
            keyword = keyword.upper()
            if keyword == "FROM":
                stage += 1
            instructions.append(Instruction(keyword, value.strip(), stage, start))
        return tuple(instructions)

    @cached_property
    def comments(self) -> tuple[str, ...]:
        """Return the text of every comment line (without the ``#``)."""
        return tuple(line.strip()[1:].strip() for line in self.text.splitlines() if line.strip().startswith("#"))

    @property
    def final_stage(self) -> int:
        """Return the index of the last build stage, the one the image is."""
        return max(instruction.stage for instruction in self.instructions)

    def stage(self, stage: int | None = None) -> list[Instruction]:
        """Return the instructions of ``stage`` (default: the final stage), ``FROM`` included."""
        stage = self.final_stage if stage is None else stage
        return [instruction for instruction in self.instructions if instruction.stage == stage]

    def of(self, keyword: str, stage: int | None = None) -> list[Instruction]:
        """Return the ``keyword`` instructions of ``stage`` (default: the final stage)."""
        return [instruction for instruction in self.stage(stage) if instruction.keyword == keyword]

    def scope(
        self, until: Instruction | None = None, stage: int | None = None, build_args: Mapping[str, str] | None = None
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Return the ARG and ENV values in effect in a stage, before ``until`` or at the end of the stage.

        Parameters
        ----------
        until : Instruction | None
            Stop before this instruction; its stage is the stage read. ``None`` reads the whole stage.
        stage : int | None
            The stage when ``until`` is not given (default: the final stage).
        build_args : Mapping[str, str] | None
            ``--build-arg`` values; they replace the default of an ARG the stage declares.

        Returns
        -------
        tuple[dict[str, str], dict[str, str]]
            The ARG values and the ENV values.
        """
        build_args = build_args or {}
        stage = until.stage if until is not None else (self.final_stage if stage is None else stage)
        args: dict[str, str] = {}
        env: dict[str, str] = {}
        for instruction in self.stage(stage):
            if instruction is until:
                break
            # ENV wins over an ARG of the same name, as in Docker.
            values = {**args, **env}
            if instruction.keyword == "ARG":
                name, has_default, default = instruction.value.partition("=")
                if name in build_args:
                    args[name] = build_args[name]
                elif has_default:
                    args[name] = expand(default.strip().strip('"'), values)
            elif instruction.keyword == "ENV":
                env.update({name: expand(value, values) for name, value in _env_pairs(instruction.value)})
        return args, env

    def env(self, stage: int | None = None, build_args: Mapping[str, str] | None = None) -> dict[str, str]:
        """Return the resolved ENV of a stage (default: the final stage, the image's own environment)."""
        return self.scope(stage=stage, build_args=build_args)[1]

    def labels(self, build_args: Mapping[str, str] | None = None) -> dict[str, str]:
        """Return the resolved LABELs of the final stage."""
        labels: dict[str, str] = {}
        for instruction in self.of("LABEL"):
            args, env = self.scope(until=instruction, build_args=build_args)
            labels.update({name: expand(value, {**args, **env}) for name, value in _env_pairs(instruction.value)})
        return labels


def _env_pairs(value: str) -> list[tuple[str, str]]:
    """Split the arguments of an ``ENV`` or ``LABEL`` instruction into name/value pairs (quotes removed)."""
    tokens = shlex.split(value, posix=True)
    if len(tokens) >= 2 and "=" not in tokens[0]:  # the legacy ``ENV NAME value`` form
        return [(tokens[0], " ".join(tokens[1:]))]
    pairs = []
    for token in tokens:
        name, separator, item = token.partition("=")
        assert separator, f"ENV/LABEL argument without '=': {token!r}"
        pairs.append((name, item))
    return pairs
