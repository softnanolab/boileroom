"""Fast contract tests for the repo-local harness."""

from pathlib import Path

import pytest

from boileroom.images.metadata import RuntimeImageSpec
from boileroom.models.registry import ModelContract, ModelSpec
from scripts.harness import check_repo

pytestmark = pytest.mark.contract


def test_current_repo_harness_passes() -> None:
    """The checked-in repository should satisfy the harness contract."""
    issues = check_repo.run_checks(check_repo.REPO_ROOT, check_repo.load_contract())
    assert issues == []


def test_required_family_files_reports_missing_files(tmp_path: Path) -> None:
    """Missing model-family files should produce actionable harness issues."""
    family_dir = tmp_path / "boileroom/models/fake"
    family_dir.mkdir(parents=True)
    (family_dir / "types.py").write_text("from dataclasses import dataclass\n", encoding="utf-8")

    contract = {
        "model_family": {
            "required_files": ["types.py", "core.py"],
        }
    }

    issues = check_repo.check_required_family_files(tmp_path, contract, ["fake"])

    assert [issue.code for issue in issues] == ["missing-family-file"]
    assert "core.py" in issues[0].message
    assert "Add core.py" in issues[0].fix


def test_lightweight_types_rejects_forbidden_import(tmp_path: Path) -> None:
    """types.py must not import heavyweight runtime modules."""
    family_dir = tmp_path / "boileroom/models/fake"
    family_dir.mkdir(parents=True)
    (family_dir / "types.py").write_text("import torch\n", encoding="utf-8")
    contract = check_repo.load_contract()

    issues = check_repo.check_lightweight_types_imports(tmp_path, contract, ["fake"])

    assert [issue.code for issue in issues] == ["forbidden-types-import"]
    assert "torch" in issues[0].message


def test_lightweight_types_allows_type_checking_import(tmp_path: Path) -> None:
    """TYPE_CHECKING-only imports should not make output types heavyweight at runtime."""
    family_dir = tmp_path / "boileroom/models/fake"
    family_dir.mkdir(parents=True)
    (family_dir / "types.py").write_text(
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    from biotite.structure import AtomArray\n",
        encoding="utf-8",
    )
    contract = check_repo.load_contract()

    issues = check_repo.check_lightweight_types_imports(tmp_path, contract, ["fake"])

    assert issues == []


def test_lightweight_types_allows_base_protocol_import(tmp_path: Path) -> None:
    """types.py may import the lightweight Boileroom base protocols."""
    family_dir = tmp_path / "boileroom/models/fake"
    family_dir.mkdir(parents=True)
    (family_dir / "types.py").write_text("from ...base import PredictionMetadata\n", encoding="utf-8")
    contract = check_repo.load_contract()

    issues = check_repo.check_lightweight_types_imports(tmp_path, contract, ["fake"])

    assert issues == []


def test_lightweight_types_rejects_absolute_in_repo_core_import(tmp_path: Path) -> None:
    """Allowing boileroom.base must not allow arbitrary in-repo runtime modules."""
    family_dir = tmp_path / "boileroom/models/fake"
    family_dir.mkdir(parents=True)
    (family_dir / "types.py").write_text("from boileroom.models.fake.core import FakeCore\n", encoding="utf-8")
    contract = check_repo.load_contract()

    issues = check_repo.check_lightweight_types_imports(tmp_path, contract, ["fake"])

    assert [issue.code for issue in issues] == ["unexpected-types-import"]
    assert "boileroom.models.fake.core" in issues[0].message


def test_lightweight_types_rejects_relative_core_import(tmp_path: Path) -> None:
    """Relative imports should be resolved before enforcing lightweight import rules."""
    family_dir = tmp_path / "boileroom/models/fake"
    family_dir.mkdir(parents=True)
    (family_dir / "types.py").write_text("from .core import FakeCore\n", encoding="utf-8")
    contract = check_repo.load_contract()

    issues = check_repo.check_lightweight_types_imports(tmp_path, contract, ["fake"])

    assert [issue.code for issue in issues] == ["unexpected-types-import"]
    assert "boileroom.models.fake.core" in issues[0].message


def test_registry_check_reports_missing_core_class(tmp_path: Path) -> None:
    """Apptainer core paths should point at a real class without importing the core module."""
    core_dir = tmp_path / "boileroom/models/fake"
    core_dir.mkdir(parents=True)
    (core_dir / "core.py").write_text("class OtherCore:\n    pass\n", encoding="utf-8")
    model_spec = ModelSpec(
        key="fake",
        public_name="Fake",
        family="fake",
        wrapper_class_path="boileroom.models.fake.fake.Fake",
        modal_class_path="boileroom.models.fake.fake.ModalFake",
        apptainer_core_class_path="boileroom.models.fake.core.FakeCore",
        apptainer_image_name="boileroom-fake",
        supported_backends=("modal", "apptainer"),
        contract=ModelContract(
            task_method="fold",
            task_kind="structure",
            static_config_keys=frozenset(),
            minimal_output_fields=("metadata", "atom_array"),
        ),
    )
    image_spec = RuntimeImageSpec(
        key="fake",
        image_name="boileroom-fake",
        dockerfile_relative_path="pyproject.toml",
        context_relative_path=".",
    )

    issues = check_repo.check_registry_image_consistency(tmp_path, [model_spec], [image_spec])

    assert any(issue.code == "missing-apptainer-core-class" for issue in issues)


def test_registry_check_enforces_required_backends(tmp_path: Path) -> None:
    """The YAML contract should drive required backend policy."""
    family_dir = tmp_path / "boileroom/models/fake"
    family_dir.mkdir(parents=True)
    (family_dir / "core.py").write_text("class FakeCore:\n    pass\n", encoding="utf-8")
    model_spec = ModelSpec(
        key="fake",
        public_name="Fake",
        family="fake",
        wrapper_class_path="boileroom.models.fake.fake.Fake",
        modal_class_path="boileroom.models.fake.fake.ModalFake",
        apptainer_core_class_path=None,
        apptainer_image_name="boileroom-fake",
        supported_backends=("modal",),
        contract=ModelContract(
            task_method="fold",
            task_kind="structure",
            static_config_keys=frozenset(),
            minimal_output_fields=("metadata", "atom_array"),
        ),
    )
    image_spec = RuntimeImageSpec(
        key="fake",
        image_name="boileroom-fake",
        dockerfile_relative_path="pyproject.toml",
        context_relative_path=".",
    )
    contract = {"registry": {"required_backends": ["modal", "apptainer"]}}

    issues = check_repo.check_registry_image_consistency(tmp_path, [model_spec], [image_spec], contract)

    assert any(issue.code == "missing-required-backend" for issue in issues)
