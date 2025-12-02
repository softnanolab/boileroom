"""Tests for ProgressTracker."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from boileroom.backend.progress import ProgressTracker, _escape_rich_markup


class TestEscapeRichMarkup:
    """Test the Rich markup escaping function."""

    def test_escape_brackets(self):
        """Test that square brackets are escaped."""
        text = "[/tmpdir/job/2155346.undefined]"
        escaped = _escape_rich_markup(text)
        assert escaped == "\\[/tmpdir/job/2155346.undefined\\]"
        # Verify that brackets are escaped (backslash before bracket)
        assert "\\[" in escaped
        assert "\\]" in escaped
        # Verify the original brackets are replaced with escaped versions
        assert escaped.count("[") == escaped.count("\\[")
        assert escaped.count("]") == escaped.count("\\]")

    def test_escape_multiple_brackets(self):
        """Test escaping multiple bracket pairs."""
        text = "[bold]text[/bold] and [red]more[/red]"
        escaped = _escape_rich_markup(text)
        assert escaped == "\\[bold\\]text\\[/bold\\] and \\[red\\]more\\[/red\\]"

    def test_no_brackets(self):
        """Test that text without brackets is unchanged."""
        text = "plain text without brackets"
        escaped = _escape_rich_markup(text)
        assert escaped == text

    def test_empty_string(self):
        """Test that empty string is handled."""
        assert _escape_rich_markup("") == ""


class TestProgressTrackerBasic:
    """Test basic ProgressTracker functionality."""

    def test_context_manager_entry_exit(self):
        """Test that ProgressTracker works as a context manager."""
        with ProgressTracker(logger_name="test.logger") as tracker:
            assert tracker is not None
            assert tracker._logger_name == "test.logger"

    def test_record_stage(self):
        """Test recording stages."""
        with ProgressTracker(logger_name="test.logger") as tracker:
            tracker.record_stage("Stage 1")
            assert "Stage 1" in tracker._sections
            assert tracker._current_stage == "Stage 1"

            tracker.record_stage("Stage 2")
            assert len(tracker._sections) == 2
            assert tracker._current_stage == "Stage 2"

    def test_set_subprocess_title(self):
        """Test setting subprocess title."""
        with ProgressTracker(logger_name="test.logger") as tracker:
            tracker.set_subprocess_title("my command")
            assert tracker._subprocess_title == "my command"

    def test_append_tail_line(self):
        """Test appending lines to tail."""
        with ProgressTracker(logger_name="test.logger", tail_size=4) as tracker:
            tracker.append_tail_line("line 1")
            tracker.append_tail_line("line 2")
            assert len(tracker._tail) == 2
            assert tracker._tail[0] == "line 1"
            assert tracker._tail[1] == "line 2"

    def test_append_tail_line_with_label(self):
        """Test appending lines with stream labels."""
        with ProgressTracker(logger_name="test.logger") as tracker:
            tracker.append_tail_line("output", stream_label="STDOUT")
            assert len(tracker._tail) == 1
            assert tracker._tail[0] == "STDOUT: output"

    def test_tail_size_limit(self):
        """Test that tail size is limited."""
        with ProgressTracker(logger_name="test.logger", tail_size=2) as tracker:
            tracker.append_tail_line("line 1")
            tracker.append_tail_line("line 2")
            tracker.append_tail_line("line 3")
            assert len(tracker._tail) == 2
            assert tracker._tail[0] == "line 2"
            assert tracker._tail[1] == "line 3"

    def test_tail_size_unlimited_in_debug(self):
        """Test that tail size is unlimited when logger is in DEBUG mode."""
        import logging

        logger = logging.getLogger("test.debug.logger")
        logger.setLevel(logging.DEBUG)

        with ProgressTracker(logger_name="test.debug.logger", tail_size=2) as tracker:
            for i in range(10):
                tracker.append_tail_line(f"line {i}")
            # Should not be limited in DEBUG mode
            assert len(tracker._tail) == 10


class TestProgressTrackerMarkupEscaping:
    """Test that ProgressTracker properly escapes Rich markup."""

    def test_markup_in_stage_description(self):
        """Test that stage descriptions with brackets are handled."""
        with ProgressTracker(logger_name="test.logger") as tracker:
            # This should not raise an error
            tracker.record_stage("Processing [/tmpdir/job/2155346.undefined]")
            assert "[/tmpdir/job/2155346.undefined]" in tracker._sections[0]

    def test_markup_in_subprocess_title(self):
        """Test that subprocess titles with brackets are handled."""
        with ProgressTracker(logger_name="test.logger") as tracker:
            tracker.set_subprocess_title("apptainer exec [container]")
            assert tracker._subprocess_title == "apptainer exec [container]"

    def test_markup_in_tail_output(self):
        """Test that tail output with brackets is handled."""
        with ProgressTracker(logger_name="test.logger") as tracker:
            tracker.append_tail_line("STDERR: FATAL: [/tmpdir/job/2155346.undefined]")
            assert "[/tmpdir/job/2155346.undefined]" in tracker._tail[0]

    def test_render_with_markup_characters(self):
        """Test that rendering with markup characters doesn't raise errors."""
        with ProgressTracker(logger_name="test.logger") as tracker:
            tracker.record_stage("Stage with [brackets]")
            tracker.set_subprocess_title("Command [with] [tags]")
            tracker.append_tail_line("Output [/path/to/file]")
            # Should not raise Rich markup errors
            render_result = tracker._render()
            assert render_result is not None


class TestProgressTrackerSubprocess:
    """Test ProgressTracker subprocess execution with mocks."""

    @patch("subprocess.Popen")
    def test_run_subprocess_success(self, mock_popen):
        """Test successful subprocess execution."""
        # Create mock file-like objects that support readline()
        stdout_lines = ["stdout line 1\n", "stdout line 2\n", ""]
        stderr_lines = ["stderr line 1\n", ""]
        stdout_iter = iter(stdout_lines)
        stderr_iter = iter(stderr_lines)

        mock_stdout = MagicMock()
        mock_stdout.readline = lambda: next(stdout_iter, "")
        mock_stdout.close = MagicMock()

        mock_stderr = MagicMock()
        mock_stderr.readline = lambda: next(stderr_iter, "")
        mock_stderr.close = MagicMock()

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.wait.return_value = 0
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        with ProgressTracker(logger_name="test.logger") as tracker:
            result = tracker.run_subprocess(
                ["echo", "test"],
                stage_label="Running echo",
                subprocess_title="echo command",
                check=True,
            )

        assert result.returncode == 0
        assert "stdout line 1" in result.stdout
        assert "stderr line 1" in result.stderr

    @patch("subprocess.Popen")
    def test_run_subprocess_failure(self, mock_popen):
        """Test failed subprocess execution."""
        # Create mock file-like objects
        stdout_lines = [""]
        stderr_lines = ["Error occurred\n", ""]
        stdout_iter = iter(stdout_lines)
        stderr_iter = iter(stderr_lines)

        mock_stdout = MagicMock()
        mock_stdout.readline = lambda: next(stdout_iter, "")
        mock_stdout.close = MagicMock()

        mock_stderr = MagicMock()
        mock_stderr.readline = lambda: next(stderr_iter, "")
        mock_stderr.close = MagicMock()

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.wait.return_value = 1
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        with ProgressTracker(logger_name="test.logger") as tracker:
            with pytest.raises(subprocess.CalledProcessError):
                tracker.run_subprocess(
                    ["false"],
                    stage_label="Running false",
                    check=True,
                )

    @patch("subprocess.Popen")
    def test_run_subprocess_no_check(self, mock_popen):
        """Test subprocess execution without check."""
        # Create mock file-like objects
        stdout_lines = [""]
        stderr_lines = [""]
        stdout_iter = iter(stdout_lines)
        stderr_iter = iter(stderr_lines)

        mock_stdout = MagicMock()
        mock_stdout.readline = lambda: next(stdout_iter, "")
        mock_stdout.close = MagicMock()

        mock_stderr = MagicMock()
        mock_stderr.readline = lambda: next(stderr_iter, "")
        mock_stderr.close = MagicMock()

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.wait.return_value = 1
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        with ProgressTracker(logger_name="test.logger") as tracker:
            result = tracker.run_subprocess(
                ["false"],
                stage_label="Running false",
                check=False,
            )

        assert result.returncode == 1

    @patch("subprocess.Popen")
    def test_run_subprocess_with_markup_in_output(self, mock_popen):
        """Test subprocess with markup-like output."""
        # Create mock file-like objects with problematic output
        stdout_lines = ["[/tmpdir/job/2155346.undefined]\n", ""]
        stderr_lines = ["STDERR: [bold]error[/bold]\n", ""]
        stdout_iter = iter(stdout_lines)
        stderr_iter = iter(stderr_lines)

        mock_stdout = MagicMock()
        mock_stdout.readline = lambda: next(stdout_iter, "")
        mock_stdout.close = MagicMock()

        mock_stderr = MagicMock()
        mock_stderr.readline = lambda: next(stderr_iter, "")
        mock_stderr.close = MagicMock()

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.wait.return_value = 0
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        with ProgressTracker(logger_name="test.logger") as tracker:
            # Should not raise Rich markup errors
            result = tracker.run_subprocess(
                ["test", "command"],
                stage_label="Test command",
            )

        assert result.returncode == 0
        assert "[/tmpdir/job/2155346.undefined]" in result.stdout


class TestMockCondaBackend:
    """Test ProgressTracker integration with mocked CondaBackend operations."""

    @patch("subprocess.Popen")
    def test_conda_env_create(self, mock_popen):
        """Test conda environment creation with ProgressTracker."""
        # Simulate conda env create output
        conda_output = [
            "Collecting package metadata (repodata.json): ...working... done\n",
            "Solving environment: ...working... done\n",
            "Downloading and Extracting Packages:\n",
            "Preparing transaction: ...working... done\n",
            "Verifying transaction: ...working... done\n",
            "Executing transaction: ...working... done\n",
            "",
        ]

        stdout_iter = iter(conda_output)
        stderr_iter = iter([""])

        mock_stdout = MagicMock()
        mock_stdout.readline = lambda: next(stdout_iter, "")
        mock_stdout.close = MagicMock()

        mock_stderr = MagicMock()
        mock_stderr.readline = lambda: next(stderr_iter, "")
        mock_stderr.close = MagicMock()

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.wait.return_value = 0
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        with ProgressTracker(logger_name="boileroom.backend.conda") as tracker:
            tracker.record_stage("Creating environment 'boileroom-esm'")
            result = tracker.run_subprocess(
                ["conda", "env", "create", "-f", "environment.yml", "-n", "boileroom-esm"],
                stage_label="Creating environment 'boileroom-esm'",
                subprocess_title="conda env create",
            )

        assert result.returncode == 0
        assert "Collecting package metadata" in result.stdout

    @patch("subprocess.Popen")
    def test_conda_env_update(self, mock_popen):
        """Test conda environment update with ProgressTracker."""
        # Simulate conda env update output
        conda_output = [
            "Collecting package metadata (repodata.json): ...working... done\n",
            "Solving environment: ...working... done\n",
            "Updating packages:\n",
            "  - numpy: 1.24.3-py39h1234567_0 --> 1.26.0-py39h1234567_0\n",
            "Preparing transaction: ...working... done\n",
            "Verifying transaction: ...working... done\n",
            "Executing transaction: ...working... done\n",
            "",
        ]

        stdout_iter = iter(conda_output)
        stderr_iter = iter([""])

        mock_stdout = MagicMock()
        mock_stdout.readline = lambda: next(stdout_iter, "")
        mock_stdout.close = MagicMock()

        mock_stderr = MagicMock()
        mock_stderr.readline = lambda: next(stderr_iter, "")
        mock_stderr.close = MagicMock()

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.wait.return_value = 0
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        with ProgressTracker(logger_name="boileroom.backend.conda") as tracker:
            tracker.record_stage("Updating environment 'boileroom-esm'")
            result = tracker.run_subprocess(
                ["conda", "env", "update", "-f", "environment.yml", "-n", "boileroom-esm"],
                stage_label="Updating environment 'boileroom-esm'",
                subprocess_title="conda env update",
            )

        assert result.returncode == 0
        assert "Updating packages" in result.stdout


class TestMockApptainerBackend:
    """Test ProgressTracker integration with mocked ApptainerBackend operations."""

    @patch("subprocess.Popen")
    def test_apptainer_pull(self, mock_popen):
        """Test apptainer image pull with ProgressTracker."""
        # Simulate apptainer pull output
        apptainer_output = [
            "INFO:    Converting OCI blobs to SIF format\n",
            "INFO:    Starting build...\n",
            "Getting image source signatures\n",
            "Copying blob sha256:abc123... done\n",
            "Copying blob sha256:def456... done\n",
            "Copying config sha256:789ghi... done\n",
            "Writing manifest to image destination\n",
            "Storing signatures\n",
            "INFO:    Creating SIF file...\n",
            "INFO:    Build complete: /path/to/image.sif\n",
            "",
        ]

        stdout_iter = iter(apptainer_output)
        stderr_iter = iter([""])

        mock_stdout = MagicMock()
        mock_stdout.readline = lambda: next(stdout_iter, "")
        mock_stdout.close = MagicMock()

        mock_stderr = MagicMock()
        mock_stderr.readline = lambda: next(stderr_iter, "")
        mock_stderr.close = MagicMock()

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.wait.return_value = 0
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        with ProgressTracker(logger_name="boileroom.backend.apptainer") as tracker:
            result = tracker.run_subprocess(
                ["apptainer", "pull", "--force", "image.sif", "docker://image:tag"],
                stage_label="Pulling image: docker://image:tag",
                subprocess_title="apptainer pull",
            )

        assert result.returncode == 0
        assert "Creating SIF file" in result.stdout

    @patch("subprocess.Popen")
    def test_apptainer_exec_with_markup(self, mock_popen):
        """Test apptainer exec with output containing markup-like patterns."""
        # Simulate apptainer exec output with problematic patterns
        apptainer_output = [
            "2025/12/02 16:48:50  warn\n",
            "rootless{opt/conda/pkgs/liblapack-3.11.0-1_h47877c9_openblas/lib/liblapack.so.3} ignoring\n",
            "(usually) harmless EPERM on setxattr \"user.rootlesscontainers\"\n",
            "INFO:    Creating SIF file...\n",
            "STDERR: FATAL:   \"python\": executable file not found in $PATH\n",
            "[/tmpdir/job/2155346.undefined]\n",
            "",
        ]

        stderr_output = [
            "STDERR: FATAL:   \"python\": executable file not found in $PATH\n",
            "",
        ]

        stdout_iter = iter(apptainer_output)
        stderr_iter = iter(stderr_output)

        mock_stdout = MagicMock()
        mock_stdout.readline = lambda: next(stdout_iter, "")
        mock_stdout.close = MagicMock()

        mock_stderr = MagicMock()
        mock_stderr.readline = lambda: next(stderr_iter, "")
        mock_stderr.close = MagicMock()

        mock_process = MagicMock()
        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        mock_process.wait.return_value = 1
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        with ProgressTracker(logger_name="boileroom.backend.apptainer") as tracker:
            tracker.record_stage("Starting server in container")
            tracker.set_subprocess_title("apptainer exec")
            # Should not raise Rich markup errors even with problematic output
            result = tracker.run_subprocess(
                ["apptainer", "exec", "image.sif", "python", "server.py"],
                stage_label="Starting server in container",
                subprocess_title="apptainer exec",
                check=False,
            )

        assert result.returncode == 1
        assert "[/tmpdir/job/2155346.undefined]" in result.stdout

    def test_apptainer_startup_simulation(self):
        """Test simulating apptainer startup with ProgressTracker."""
        # This test doesn't need subprocess mocking, just tests the tracker's ability
        # to handle output that would come from streaming during startup
        with ProgressTracker(logger_name="boileroom.backend.apptainer") as tracker:
            tracker.record_stage("Starting server in container")
            tracker.set_subprocess_title("apptainer exec")
            # Simulate streaming output while waiting for health check
            tracker.append_tail_line("Starting server...", stream_label="STDOUT")
            tracker.append_tail_line("Loading model...", stream_label="STDOUT")
            tracker.append_tail_line("Server ready on http://127.0.0.1:8000", stream_label="STDOUT")

        assert len(tracker._tail) == 3
        assert "Server ready" in tracker._tail[2]

