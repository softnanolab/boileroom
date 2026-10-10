"""What the controller container is given: it must be able to rebuild the sandbox image's definition."""

from pathlib import PurePosixPath

from modal.file_pattern_matcher import FilePatternMatcher
from modal.mount import _MountedPythonModule

from infra.modal_ci import image


def shipped(ignore) -> set[str]:
    entry = _MountedPythonModule("infra", "/root", ignore)
    return {PurePosixPath(remote).relative_to("/root").as_posix() for _, remote in entry.get_files_to_upload()}


def test_controller_ships_every_file_the_sandbox_image_is_built_from() -> None:
    files = shipped(FilePatternMatcher(*image.SOURCE_IGNORE))
    for path in image.SANDBOX_DIR.iterdir():
        if path.is_file():
            assert f"infra/modal_ci/sandbox/{path.name}" in files, f"{path.name} would be missing in the controller"


def test_controller_does_not_ship_bytecode() -> None:
    assert not [
        f for f in shipped(FilePatternMatcher(*image.SOURCE_IGNORE)) if "__pycache__" in f or f.endswith(".pyc")
    ]


def test_modal_default_would_have_dropped_the_shell_hook() -> None:
    """Pins the premise of `SOURCE_IGNORE`: if Modal's default ever ships non-.py files, this can go."""
    from modal.file_pattern_matcher import NON_PYTHON_FILES

    assert "infra/modal_ci/sandbox/job-started.sh" not in shipped(NON_PYTHON_FILES)
