from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager
from pathlib import Path
import subprocess
from threading import Thread
from typing import IO, Sequence

from rich.align import Align  # type: ignore[import-not-found]
from rich.console import Console, Group  # type: ignore[import-not-found]
from rich.live import Live  # type: ignore[import-not-found]
from rich.panel import Panel  # type: ignore[import-not-found]
from rich.spinner import Spinner  # type: ignore[import-not-found]
from rich.text import Text  # type: ignore[import-not-found]


def _escape_rich_markup(text: str) -> str:
    """Escape Rich markup characters in text to prevent parsing errors.

    Parameters
    ----------
    text : str
        Text that may contain Rich markup characters.

    Returns
    -------
    str
        Text with square brackets escaped to prevent Rich from interpreting them as markup.
    """
    return text.replace("[", "\\[").replace("]", "\\]")


class ProgressTracker(AbstractContextManager["ProgressTracker"]):
    """Render a Rich spinner with a rolling subprocess log tail.

    Generic progress tracker for subprocess-based operations that provides
    visual feedback with a spinner and streaming output from long-running commands.
    """

    def __init__(
        self,
        logger_name: str,
        *,
        tail_size: int = 20,
        console: Console | None = None,
        plain: bool | None = None,
        log_file_path: Path | None = None,
    ) -> None:
        self._logger_name = logger_name
        self._logger = logging.getLogger(logger_name)
        self._show_all_tail = self._logger.getEffectiveLevel() <= logging.DEBUG
        self._tail_limit = None if self._show_all_tail else tail_size
        self._tail: list[str] = []
        # Store full output history separated by stream for error reporting
        self._stdout_lines: list[str] = []
        self._stderr_lines: list[str] = []
        self._sections: list[str] = []
        self._current_stage = "Starting..."
        self._subprocess_title = "subprocess output"
        self._console = console or Console()
        self._live: Live | None = None
        # Plain mode: skip Rich Live output and log to the standard logger instead.
        if plain is None:
            env_mode = os.environ.get("BOILEROOM_PROGRESS", "").lower()
            # Treat several values as meaning \"no Rich UI\"
            plain = env_mode in {"plain", "none", "off"}
        self._plain = plain
        self._use_rich = not self._plain

        # Optional log file tee for debugging: each tail line is also written to this file.
        self._log_file_path: Path | None = log_file_path
        self._log_file_handle: IO[str] | None = None

    def __enter__(self) -> "ProgressTracker":
        if self._use_rich and self._console.is_terminal:
            self._live = Live(
                self._render(),
                console=self._console,
                refresh_per_second=12,
            )
            self._live.__enter__()
        else:
            self._live = None
            self._logger.debug(
                f"ProgressTracker for {self._logger_name} running in plain mode (no Rich Live UI)"
            )
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc, exc_tb)
            self._live = None
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.close()
            except OSError:
                # Best-effort close; failure here should not surface to caller.
                self._logger.debug("Failed to close ProgressTracker log file cleanly", exc_info=True)
            self._log_file_handle = None

    def record_stage(self, description: str) -> None:
        """Append a new stage line to the header."""
        self._sections.append(description)
        self._current_stage = description
        if self._plain:
            self._logger.info(f"{self._logger_name} ---- {description}")
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
        
        # Store full output history for error reporting
        if stream_label == "STDOUT":
            self._stdout_lines.append(line)
        elif stream_label == "STDERR":
            self._stderr_lines.append(line)

        # Optionally tee to a log file for easier debugging.
        if self._log_file_path is not None:
            if self._log_file_handle is None:
                try:
                    self._log_file_handle = self._log_file_path.open("a", encoding="utf-8")
                    self._logger.debug(f"ProgressTracker logging to file: {self._log_file_path}")
                except OSError:
                    # If we cannot open the file, disable further attempts.
                    self._logger.warning(
                        f"Failed to open ProgressTracker log file at {self._log_file_path}, disabling file logging",
                        exc_info=True,
                    )
                    self._log_file_path = None
            if self._log_file_handle is not None:
                try:
                    self._log_file_handle.write(f"{text}\n")
                    self._log_file_handle.flush()
                except OSError:
                    self._logger.debug(
                        "Error writing to ProgressTracker log file; disabling file logging",
                        exc_info=True,
                    )
                    self._log_file_path = None
                    self._log_file_handle = None

        # In plain mode, also emit tail lines to the logger at debug level.
        if self._plain:
            self._logger.debug(f"{self._logger_name} tail: {text}")

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

    def get_captured_output(self) -> tuple[str, str]:
        """Get all captured stdout and stderr output.
        
        Returns
        -------
        tuple[str, str]
            A tuple of (stdout, stderr) as strings, with newlines preserved.
        """
        stdout = "\n".join(self._stdout_lines)
        stderr = "\n".join(self._stderr_lines)
        return stdout, stderr

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    def _render(self) -> Align:
        # Escape user-provided content to prevent Rich markup parsing errors
        header_lines = [
            Text(_escape_rich_markup(f"{self._logger_name} ---- {line}"), style="cyan")
            for line in self._sections[-3:]
        ]
        spinner = Spinner("dots", text=_escape_rich_markup(self._current_stage), style="bold green")
        tail_content = "\n".join(self._tail) if self._tail else "waiting for subprocess output..."
        # Escape subprocess output to prevent Rich from interpreting it as markup
        tail_text = Text(_escape_rich_markup(tail_content))
        tail_panel = Panel(
            tail_text,
            title=_escape_rich_markup(self._subprocess_title),
            border_style="magenta",
        )
        group = Group(*header_lines, spinner, tail_panel)
        panel = Panel(group, title=_escape_rich_markup(self._logger_name.upper()), border_style="blue")
        height = self._console.size.height if self._console.size is not None else None
        return Align(panel, vertical="bottom", height=height)

