from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from pathlib import Path
import subprocess
from threading import Thread
from typing import Sequence

from rich.align import Align  # type: ignore[import-not-found]
from rich.console import Console, Group  # type: ignore[import-not-found]
from rich.live import Live  # type: ignore[import-not-found]
from rich.panel import Panel  # type: ignore[import-not-found]
from rich.spinner import Spinner  # type: ignore[import-not-found]
from rich.text import Text  # type: ignore[import-not-found]


class CondaProgressTracker(AbstractContextManager["CondaProgressTracker"]):
    """Render a Rich spinner with a rolling subprocess log tail."""

    def __init__(
        self,
        logger_name: str,
        *,
        tail_size: int = 4,
        console: Console | None = None,
    ) -> None:
        self._logger_name = logger_name
        self._logger = logging.getLogger(logger_name)
        self._show_all_tail = self._logger.getEffectiveLevel() <= logging.DEBUG
        self._tail_limit = None if self._show_all_tail else tail_size
        self._tail: list[str] = []
        self._sections: list[str] = []
        self._current_stage = "Starting..."
        self._subprocess_title = "subprocess output"
        self._console = console or Console()
        self._live: Live | None = None

    def __enter__(self) -> "CondaProgressTracker":
        if self._console.is_terminal:
            self._live = Live(
                self._render(),
                console=self._console,
                refresh_per_second=12,
            )
            self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc, exc_tb)
            self._live = None

    def record_stage(self, description: str) -> None:
        """Append a new stage line to the header."""
        self._sections.append(description)
        self._current_stage = description
        self._refresh()

    def set_subprocess_title(self, title: str) -> None:
        self._subprocess_title = title
        self._refresh()

    def append_tail_line(self, line: str, *, stream_label: str | None = None) -> None:
        prefix = f"{stream_label}: " if stream_label else ""
        text = f"{prefix}{line}".rstrip()
        if not text:
            return
        self._tail.append(text)
        if self._tail_limit is not None and len(self._tail) > self._tail_limit:
            self._tail = self._tail[-self._tail_limit :]
        self._refresh()

    def run_subprocess(
        self,
        command: Sequence[str],
        *,
        stage_label: str,
        subprocess_title: str | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Execute a command while streaming the trailing output."""
        self.record_stage(stage_label)
        if subprocess_title is not None:
            self.set_subprocess_title(subprocess_title)
        else:
            self.set_subprocess_title(stage_label)

        normalized_cwd: str | None
        if isinstance(cwd, Path):
            normalized_cwd = str(cwd)
        else:
            normalized_cwd = cwd

        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=normalized_cwd,
            env=env,
        )

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        def consume_stream(pipe, collector: list[str], label: str) -> None:
            if pipe is None:
                return
            for raw_line in iter(pipe.readline, ""):
                collector.append(raw_line)
                stripped_line = raw_line.rstrip()
                if stripped_line:
                    self.append_tail_line(stripped_line, stream_label=label)
            pipe.close()

        threads: list[Thread] = []
        for label, pipe, collector in (
            ("STDOUT", process.stdout, stdout_lines),
            ("STDERR", process.stderr, stderr_lines),
        ):
            thread = Thread(target=consume_stream, args=(pipe, collector, label), daemon=True)
            thread.start()
            threads.append(thread)

        return_code = process.wait()
        for thread in threads:
            thread.join()

        completed = subprocess.CompletedProcess(
            list(command),
            return_code,
            stdout="".join(stdout_lines),
            stderr="".join(stderr_lines),
        )

        if check and completed.returncode != 0:
            raise subprocess.CalledProcessError(
                completed.returncode,
                list(command),
                output=completed.stdout,
                stderr=completed.stderr,
            )

        return completed

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Align:
        header_lines = [
            Text(f"{self._logger_name} ---- {line}", style="cyan")
            for line in self._sections[-3:]
        ]
        spinner = Spinner("dots", text=self._current_stage, style="bold green")
        tail_body = "\n".join(self._tail) if self._tail else "waiting for subprocess output..."
        tail_panel = Panel(
            tail_body,
            title=self._subprocess_title,
            border_style="magenta",
        )
        group = Group(*header_lines, spinner, tail_panel)
        panel = Panel(group, title=self._logger_name.upper(), border_style="blue")
        height = self._console.size.height if self._console.size is not None else None
        return Align(panel, vertical="bottom", height=height)

