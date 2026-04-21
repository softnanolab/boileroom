#!/usr/bin/env python3
"""Validate Boileroom's repo-local implementation harness."""

from __future__ import annotations

import ast
import importlib
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import click
import yaml

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boileroom.images.import_checks import iter_image_targets  # noqa: E402
from boileroom.images.metadata import MODEL_IMAGE_SPECS, RuntimeImageSpec  # noqa: E402
from boileroom.models.registry import MODEL_SPECS, ModelSpec, resolve_object  # noqa: E402
from scripts.cli_utils import CONTEXT_SETTINGS  # noqa: E402

DEFAULT_CONTRACT_PATH = REPO_ROOT / "harness/model_family_contract.yaml"


@dataclass(frozen=True)
class CheckIssue:
    """Actionable harness validation failure."""

    code: str
    subject: str
    message: str
    fix: str
    path: str | None = None


@dataclass(frozen=True)
class ImportReference:
    """Import found in a lightweight types module."""

    module: str
    line_no: int

    @property
    def root(self) -> str:
        """Return the top-level import root."""

        return self.module.split(".", 1)[0]


def _issue(code: str, subject: str, message: str, fix: str, path: Path | str | None = None) -> CheckIssue:
    """Create a normalized issue."""

    return CheckIssue(code=code, subject=subject, message=message, fix=fix, path=str(path) if path else None)


def load_contract(path: Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Load the harness YAML contract."""

    with path.open(encoding="utf-8") as handle:
        contract = yaml.safe_load(handle) or {}
    if not isinstance(contract, dict):
        raise ValueError(f"Harness contract must be a mapping: {path}")
    return contract


def _family_dirs(repo_root: Path) -> set[str]:
    """Return model family directories present in the repository."""

    models_root = repo_root / "boileroom/models"
    return {
        path.name
        for path in models_root.iterdir()
        if path.is_dir() and not path.name.startswith("_") and path.name != "__pycache__"
    }


def _required_files(contract: dict[str, Any]) -> list[str]:
    """Return required model-family files from the contract."""

    raw_files = contract.get("model_family", {}).get("required_files", [])
    return [str(path) for path in raw_files]


def _type_import_rules(contract: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    """Return allowed and forbidden import rules for lightweight types modules."""

    raw_rules = contract.get("model_family", {}).get("lightweight_types", {})
    allowed_roots = {str(value) for value in raw_rules.get("allowed_import_roots", [])}
    allowed_modules = {str(value) for value in raw_rules.get("allowed_import_modules", [])}
    forbidden = {str(value) for value in raw_rules.get("forbidden_import_roots", [])}
    return allowed_roots, allowed_modules, forbidden


def _required_backends(contract: dict[str, Any]) -> set[str]:
    """Return required model backends from the contract."""

    return {str(value) for value in contract.get("registry", {}).get("required_backends", [])}


def check_required_family_files(repo_root: Path, contract: dict[str, Any], family_names: Iterable[str]) -> list[CheckIssue]:
    """Validate required files for every model family."""

    issues: list[CheckIssue] = []
    required_files = _required_files(contract)
    for family in sorted(family_names):
        family_dir = repo_root / "boileroom/models" / family
        if not family_dir.exists():
            issues.append(
                _issue(
                    "missing-family-dir",
                    family,
                    f"Model family directory does not exist: {family_dir}",
                    f"Create boileroom/models/{family}/ or remove the stale registry/image reference.",
                    family_dir,
                )
            )
            continue
        for relative_path in required_files:
            expected = family_dir / relative_path
            if not expected.exists():
                issues.append(
                    _issue(
                        "missing-family-file",
                        family,
                        f"Missing required model-family file: {expected}",
                        f"Add {relative_path} for family {family}, or update harness/model_family_contract.yaml if the contract changed.",
                        expected,
                    )
                )
    return issues


def _is_type_checking_test(node: ast.AST) -> bool:
    """Return whether an if-test is a TYPE_CHECKING guard."""

    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    if isinstance(node, ast.Attribute):
        return node.attr == "TYPE_CHECKING"
    return False


def _module_name_for_path(repo_root: Path, path: Path) -> str:
    """Return the dotted module name for a Python source path under the repo root."""

    relative_path = path.relative_to(repo_root).with_suffix("")
    return ".".join(relative_path.parts)


def _module_path_for_name(repo_root: Path, module_path: str) -> tuple[Path, bool] | None:
    """Return the source file for a dotted module and whether it is a package."""

    module_file = repo_root / f"{module_path.replace('.', '/')}.py"
    if module_file.exists():
        return module_file, False

    package_file = repo_root / module_path.replace(".", "/") / "__init__.py"
    if package_file.exists():
        return package_file, True

    return None


def _resolve_import_from_module(repo_root: Path, path: Path, node: ast.ImportFrom) -> list[str]:
    """Resolve an ImportFrom node to fully qualified module names."""

    if node.level == 0:
        return [node.module] if node.module else []

    package_parts = _module_name_for_path(repo_root, path).split(".")[:-1]
    keep_count = len(package_parts) - node.level + 1
    if keep_count < 0:
        return []

    base_parts = package_parts[:keep_count]
    if node.module:
        return [".".join([*base_parts, *node.module.split(".")])]

    return [".".join([*base_parts, alias.name]) for alias in node.names if alias.name != "*"]


class _ImportVisitor(ast.NodeVisitor):
    """Collect imports outside TYPE_CHECKING guards."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.imports: list[ImportReference] = []
        self.path: Path | None = None
        self._type_checking_depth = 0

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_test(node.test):
            self._type_checking_depth += 1
            for child in node.body:
                self.visit(child)
            self._type_checking_depth -= 1
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        if self._type_checking_depth:
            return
        for alias in node.names:
            self.imports.append(ImportReference(alias.name, node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._type_checking_depth:
            return
        if self.path is None:
            raise RuntimeError("Import visitor path was not initialized.")
        for module in _resolve_import_from_module(self.repo_root, self.path, node):
            if module:
                self.imports.append(ImportReference(module, node.lineno))

    def collect(self, path: Path) -> list[ImportReference]:
        """Collect imports from a Python source file."""

        self.path = path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.visit(tree)
        return self.imports


def _imports(repo_root: Path, path: Path) -> list[ImportReference]:
    """Return imports used outside TYPE_CHECKING guards."""

    return _ImportVisitor(repo_root).collect(path)


def check_lightweight_types_imports(
    repo_root: Path,
    contract: dict[str, Any],
    family_names: Iterable[str],
) -> list[CheckIssue]:
    """Validate that model-family types modules stay lightweight."""

    allowed_roots, allowed_modules, forbidden = _type_import_rules(contract)
    issues: list[CheckIssue] = []
    for family in sorted(family_names):
        types_path = repo_root / "boileroom/models" / family / "types.py"
        if not types_path.exists():
            continue
        for import_ref in _imports(repo_root, types_path):
            if import_ref.root in forbidden:
                issues.append(
                    _issue(
                        "forbidden-types-import",
                        family,
                        f"{types_path}:{import_ref.line_no} imports heavyweight module {import_ref.module!r}.",
                        "Move heavyweight imports into core.py or behind a TYPE_CHECKING guard if only needed for annotations.",
                        types_path,
                    )
                )
                continue
            if import_ref.root in allowed_roots or import_ref.module in allowed_modules:
                continue
            if allowed_roots or allowed_modules:
                issues.append(
                    _issue(
                        "unexpected-types-import",
                        family,
                        f"{types_path}:{import_ref.line_no} imports module {import_ref.module!r}, which is not allowed by the lightweight types contract.",
                        f"Remove the import, move it behind TYPE_CHECKING, or add {import_ref.module!r} to allowed_import_modules after review.",
                        types_path,
                    )
                )
    return issues


def _resolve_reexported_class_path(
    repo_root: Path,
    current_module_path: str,
    current_path: Path,
    is_package: bool,
    node: ast.ImportFrom,
    class_name: str,
) -> str | None:
    """Return the original dotted class path for a matching re-export."""

    if node.level == 0:
        import_module_path = node.module
    else:
        package_parts = current_module_path.split(".") if is_package else current_module_path.split(".")[:-1]
        keep_count = len(package_parts) - node.level + 1
        if keep_count < 0:
            return None
        base_parts = package_parts[:keep_count]
        import_module_path = ".".join([*base_parts, *(node.module.split(".") if node.module else [])])

    for alias in node.names:
        if alias.name == "*":
            continue
        exported_name = alias.asname or alias.name
        if exported_name != class_name:
            continue
        if import_module_path:
            return f"{import_module_path}.{alias.name}"

        resolved_modules = _resolve_import_from_module(repo_root, current_path, node)
        matching_module = next((module for module in resolved_modules if module.rsplit(".", 1)[-1] == alias.name), None)
        if matching_module is not None:
            return matching_module

    return None


def _class_exists_in_module(repo_root: Path, dotted_path: str, seen: set[str] | None = None) -> bool:
    """Return whether a dotted class path exists without importing the module."""

    seen = set() if seen is None else seen
    if dotted_path in seen:
        return False
    seen.add(dotted_path)

    module_path, separator, class_name = dotted_path.rpartition(".")
    if not separator:
        return False
    module_source = _module_path_for_name(repo_root, module_path)
    if module_source is None:
        return False
    candidate, is_package = module_source
    tree = ast.parse(candidate.read_text(encoding="utf-8"), filename=str(candidate))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return True
        if isinstance(node, ast.ImportFrom):
            reexported_path = _resolve_reexported_class_path(
                repo_root,
                module_path,
                candidate,
                is_package,
                node,
                class_name,
            )
            if reexported_path is not None and _class_exists_in_module(repo_root, reexported_path, seen):
                return True
    return False


def _image_specs_by_key(image_specs: Sequence[RuntimeImageSpec]) -> dict[str, RuntimeImageSpec]:
    """Return image specs keyed by family key."""

    return {spec.key: spec for spec in image_specs}


def check_registry_image_consistency(
    repo_root: Path,
    model_specs: Sequence[ModelSpec],
    image_specs: Sequence[RuntimeImageSpec],
    contract: dict[str, Any] | None = None,
) -> list[CheckIssue]:
    """Validate registry/image metadata links without importing model cores."""

    issues: list[CheckIssue] = []
    image_by_key = _image_specs_by_key(image_specs)
    families_with_models = {spec.family for spec in model_specs}
    family_dirs = _family_dirs(repo_root)
    image_keys = set(image_by_key)
    required_backends = _required_backends(contract or {})

    for family in sorted(family_dirs - image_keys):
        issues.append(
            _issue(
                "unregistered-family-image",
                family,
                f"Model family {family!r} has a directory but no RuntimeImageSpec.",
                "Add a RuntimeImageSpec to boileroom/images/metadata.py or remove the unused family directory.",
                repo_root / "boileroom/models" / family,
            )
        )

    for family in sorted(families_with_models - image_keys):
        issues.append(
            _issue(
                "missing-image-spec",
                family,
                f"Model family {family!r} is used by a ModelSpec but has no RuntimeImageSpec.",
                "Add matching image metadata to MODEL_IMAGE_SPECS in boileroom/images/metadata.py.",
            )
        )

    for image_key in sorted(image_keys - families_with_models):
        issues.append(
            _issue(
                "image-without-model",
                image_key,
                f"RuntimeImageSpec {image_key!r} has no registered ModelSpec family.",
                "Register a ModelSpec for this family or remove the unused image metadata.",
            )
        )

    for image_spec in image_specs:
        for label, path in (
            ("context path", image_spec.context_path),
            ("Dockerfile", image_spec.dockerfile_path),
        ):
            if not path.exists():
                issues.append(
                    _issue(
                        "missing-image-path",
                        image_spec.key,
                        f"{label} does not exist for image spec {image_spec.key!r}: {path}",
                        "Fix the RuntimeImageSpec path or add the missing file.",
                        path,
                    )
                )
        if image_spec.config_relative_path is not None:
            config_path = repo_root / image_spec.config_relative_path
            if not config_path.exists():
                issues.append(
                    _issue(
                        "missing-image-config",
                        image_spec.key,
                        f"Image config does not exist for image spec {image_spec.key!r}: {config_path}",
                        "Fix config_relative_path or add the missing config.yaml.",
                        config_path,
                    )
                )

    seen_keys: set[str] = set()
    seen_public_names: set[str] = set()
    for spec in model_specs:
        if spec.key in seen_keys:
            issues.append(
                _issue("duplicate-model-key", spec.key, f"Duplicate ModelSpec key: {spec.key}", "Use unique model keys.")
            )
        seen_keys.add(spec.key)
        if spec.public_name in seen_public_names:
            issues.append(
                _issue(
                    "duplicate-public-name",
                    spec.public_name,
                    f"Duplicate ModelSpec public name: {spec.public_name}",
                    "Use unique public wrapper names.",
                )
            )
        seen_public_names.add(spec.public_name)

        missing_backends = required_backends - set(spec.supported_backends)
        if missing_backends:
            issues.append(
                _issue(
                    "missing-required-backend",
                    spec.key,
                    f"{spec.public_name} is missing required backend(s): {', '.join(sorted(missing_backends))}.",
                    "Add backend support metadata or update harness/model_family_contract.yaml if the policy changed.",
                )
            )

        linked_image_spec = image_by_key.get(spec.family)
        if linked_image_spec is not None and spec.apptainer_image_name != linked_image_spec.image_name:
            issues.append(
                _issue(
                    "apptainer-image-mismatch",
                    spec.key,
                    f"{spec.public_name} uses apptainer image {spec.apptainer_image_name!r}, expected {linked_image_spec.image_name!r}.",
                    "Set ModelSpec.apptainer_image_name from get_model_image_spec(<family>).image_name.",
                )
            )
        if "modal" in spec.supported_backends and spec.modal_class_path is None:
            issues.append(
                _issue(
                    "missing-modal-class",
                    spec.key,
                    f"{spec.public_name} declares Modal support but has no modal_class_path.",
                    "Add a Modal wrapper class path or remove modal from supported_backends.",
                )
            )
        if "apptainer" in spec.supported_backends:
            if spec.apptainer_core_class_path is None:
                issues.append(
                    _issue(
                        "missing-apptainer-core",
                        spec.key,
                        f"{spec.public_name} declares Apptainer support but has no apptainer_core_class_path.",
                        "Add an Apptainer core class path or remove apptainer from supported_backends.",
                    )
                )
            elif not _class_exists_in_module(repo_root, spec.apptainer_core_class_path):
                issues.append(
                    _issue(
                        "missing-apptainer-core-class",
                        spec.key,
                        f"Apptainer core class path does not resolve statically: {spec.apptainer_core_class_path}",
                        "Create the core class or fix ModelSpec.apptainer_core_class_path.",
                    )
                )
            if spec.apptainer_image_name is None:
                issues.append(
                    _issue(
                        "missing-apptainer-image",
                        spec.key,
                        f"{spec.public_name} declares Apptainer support but has no apptainer_image_name.",
                        "Set apptainer_image_name to the family RuntimeImageSpec image_name.",
                    )
                )
    return issues


def check_wrapper_exposure(model_specs: Sequence[ModelSpec]) -> list[CheckIssue]:
    """Validate public wrapper exposure and registry identity."""

    issues: list[CheckIssue] = []
    boileroom_module = importlib.import_module("boileroom")
    models_module = importlib.import_module("boileroom.models")
    top_level_all = set(getattr(boileroom_module, "__all__", []))
    models_all = set(getattr(models_module, "__all__", []))

    for spec in model_specs:
        try:
            wrapper_cls = resolve_object(spec.wrapper_class_path)
        except Exception as exc:  # noqa: BLE001
            # Wrapper modules can fail through third-party import-time errors; report them as harness issues.
            issues.append(
                _issue(
                    "unresolved-wrapper-class",
                    spec.key,
                    f"Could not import wrapper class {spec.wrapper_class_path}: {exc}",
                    "Fix ModelSpec.wrapper_class_path or the wrapper module import error.",
                )
            )
            continue

        if getattr(wrapper_cls, "MODEL_SPEC", None) is not spec:
            issues.append(
                _issue(
                    "wrapper-model-spec-mismatch",
                    spec.key,
                    f"{spec.wrapper_class_path}.MODEL_SPEC is not the registered ModelSpec object.",
                    "Set the wrapper class MODEL_SPEC attribute to the registry constant used in MODEL_SPECS.",
                )
            )

        if spec.modal_class_path is not None:
            try:
                resolve_object(spec.modal_class_path)
            except Exception as exc:  # noqa: BLE001
                # Modal wrapper imports can fail before the class is reached; report the import failure.
                issues.append(
                    _issue(
                        "unresolved-modal-class",
                        spec.key,
                        f"Could not import Modal class {spec.modal_class_path}: {exc}",
                        "Fix ModelSpec.modal_class_path or the Modal wrapper module import error.",
                    )
                )

        if spec.public_name not in top_level_all:
            issues.append(
                _issue(
                    "missing-top-level-export",
                    spec.key,
                    f"{spec.public_name} is missing from boileroom.__all__.",
                    "Add the public wrapper name to boileroom/__init__.py.",
                    "boileroom/__init__.py",
                )
            )
        elif getattr(boileroom_module, spec.public_name, None) is not wrapper_cls:
            issues.append(
                _issue(
                    "broken-top-level-export",
                    spec.key,
                    f"boileroom.{spec.public_name} does not resolve to {spec.wrapper_class_path}.",
                    "Fix boileroom/__init__.py lazy export wiring.",
                    "boileroom/__init__.py",
                )
            )
        if spec.public_name not in models_all:
            issues.append(
                _issue(
                    "missing-models-export",
                    spec.key,
                    f"{spec.public_name} is missing from boileroom.models.__all__.",
                    "Add the public wrapper name to boileroom/models/__init__.py.",
                    "boileroom/models/__init__.py",
                )
            )
        elif getattr(models_module, spec.public_name, None) is not wrapper_cls:
            issues.append(
                _issue(
                    "broken-models-export",
                    spec.key,
                    f"boileroom.models.{spec.public_name} does not resolve to {spec.wrapper_class_path}.",
                    "Fix boileroom/models/__init__.py lazy export wiring.",
                    "boileroom/models/__init__.py",
                )
            )

        family_module = importlib.import_module(f"boileroom.models.{spec.family}")
        family_all = set(getattr(family_module, "__all__", []))
        if spec.public_name not in family_all:
            issues.append(
                _issue(
                    "missing-family-export",
                    spec.key,
                    f"{spec.public_name} is missing from boileroom.models.{spec.family}.__all__.",
                    f"Add the public wrapper name to boileroom/models/{spec.family}/__init__.py.",
                    f"boileroom/models/{spec.family}/__init__.py",
                )
            )
        elif getattr(family_module, spec.public_name, None) is not wrapper_cls:
            issues.append(
                _issue(
                    "broken-family-export",
                    spec.key,
                    f"boileroom.models.{spec.family}.{spec.public_name} does not resolve to {spec.wrapper_class_path}.",
                    f"Fix boileroom/models/{spec.family}/__init__.py lazy export wiring.",
                    f"boileroom/models/{spec.family}/__init__.py",
                )
            )
    return issues


def check_smoke_targets(image_specs: Sequence[RuntimeImageSpec]) -> list[CheckIssue]:
    """Validate image smoke target enumeration."""

    issues: list[CheckIssue] = []
    expected_keys = {spec.key for spec in image_specs}
    actual_keys = {image_key for image_key, *_rest in iter_image_targets(None, [], image_specs=image_specs)}
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing:
        issues.append(
            _issue(
                "missing-smoke-target",
                "image-smoke",
                f"Image smoke target enumeration is missing: {', '.join(sorted(missing))}",
                "Update boileroom/images/import_checks.py so every RuntimeImageSpec is included.",
            )
        )
    if extra:
        issues.append(
            _issue(
                "extra-smoke-target",
                "image-smoke",
                f"Image smoke target enumeration has unexpected targets: {', '.join(sorted(extra))}",
                "Update boileroom/images/import_checks.py or MODEL_IMAGE_SPECS so they agree.",
            )
        )
    return issues


def run_checks(repo_root: Path = REPO_ROOT, contract: dict[str, Any] | None = None) -> list[CheckIssue]:
    """Run all objective harness checks."""

    loaded_contract = contract or load_contract()
    # Image-only keys are reported by check_registry_image_consistency, not treated as family directories.
    family_names = _family_dirs(repo_root) | {spec.family for spec in MODEL_SPECS}
    issues: list[CheckIssue] = []
    issues.extend(check_required_family_files(repo_root, loaded_contract, family_names))
    issues.extend(check_lightweight_types_imports(repo_root, loaded_contract, family_names))
    issues.extend(check_registry_image_consistency(repo_root, MODEL_SPECS, MODEL_IMAGE_SPECS, loaded_contract))
    issues.extend(check_wrapper_exposure(MODEL_SPECS))
    issues.extend(check_smoke_targets(MODEL_IMAGE_SPECS))
    return issues


def _write_json_report(path: Path, issues: Sequence[CheckIssue]) -> None:
    """Write a machine-readable harness report."""

    payload = {"ok": not issues, "issue_count": len(issues), "issues": [asdict(issue) for issue in issues]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_report(issues: Sequence[CheckIssue]) -> None:
    """Print a human-readable harness report."""

    if not issues:
        print("Boileroom harness checks passed.")
        return

    print(f"Boileroom harness found {len(issues)} issue(s):", file=sys.stderr)
    for index, issue in enumerate(issues, start=1):
        location = f" ({issue.path})" if issue.path else ""
        print(f"{index}. [{issue.code}] {issue.subject}{location}", file=sys.stderr)
        print(f"   Problem: {issue.message}", file=sys.stderr)
        print(f"   Fix: {issue.fix}", file=sys.stderr)


def run_harness(contract_path: Path = DEFAULT_CONTRACT_PATH, json_output: Path | None = None) -> int:
    """Run the harness CLI."""

    contract = load_contract(contract_path)
    issues = run_checks(REPO_ROOT, contract)
    if json_output is not None:
        _write_json_report(json_output, issues)
    _print_report(issues)
    return 1 if issues else 0


@click.command(context_settings=CONTEXT_SETTINGS, help="Run objective Boileroom harness checks.")
@click.option(
    "--contract",
    "contract_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_CONTRACT_PATH,
    help="Path to the YAML harness contract.",
)
@click.option(
    "--json-output",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional path for a JSON report.",
)
def cli(contract_path: Path, json_output: Path | None) -> None:
    """Run the harness Click command."""

    raise click.exceptions.Exit(run_harness(contract_path, json_output))


if __name__ == "__main__":
    cli()
