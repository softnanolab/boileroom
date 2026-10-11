"""Apply minimal, checked build-time fixes to the pinned upstream MiniFold source.

The upstream wheel omits subpackages, and eager Triton decorators initialize a
CUDA driver even when kernels=False. The non-kernel triangular update uses its
input device rather than querying MPS inside a full graph. Its FP32 autocast
subregion and all tensor operations are unchanged.
"""

import ast
import hashlib
import sys
from pathlib import Path


def _patch_triangular_device(text: str) -> str:
    """Change only the exact pinned non-kernel function's autocast device source."""
    try:
        definitions = [
            node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == "triangular"
        ]
    except SyntaxError as error:
        raise ValueError("Unexpected upstream triangular source; re-review the patch") from error
    if len(definitions) != 1:
        raise ValueError("Unexpected upstream triangular definition; re-review the patch")
    original = ast.get_source_segment(text, definitions[0])
    # Exact function in miniformer.py at upstream 63db8b91f5bb64535f20810f014e69df66b0faad.
    expected = "eff43f414c2af4c89b1167eb0bd525070151afa43bb89e9d7fbd88a429aa1427"
    if original is None or hashlib.sha256(original.encode()).hexdigest() != expected or text.count(original) != 1:
        raise ValueError("Unexpected upstream triangular body; re-review the patch")
    old = '    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")'
    return text.replace(original, original.replace(old, "    device = x.device", 1), 1)


def patch(root: Path) -> None:
    """Patch only the exact reviewed upstream source; fail on unexpected changes."""
    project = root / "pyproject.toml"
    old = 'packages = ["minifold"]'
    text = project.read_text()
    if text.count(old) != 1:
        raise ValueError("Unexpected upstream MiniFold packaging; re-review the patch")
    project.write_text(text.replace(old, 'packages = {find = {include = ["minifold*"]}}'))
    source = root / "minifold/model/miniformer.py"
    text = source.read_text()
    for name in ("gating", "mlp"):
        old = f"from minifold.model.kernels.{name} import {name}_kernel\n"
        if text.count(old) != 1:
            raise ValueError(f"Unexpected upstream {name} import; re-review the patch")
        text = text.replace(old, "")
    old = "    return mlp_kernel(x, w1, w2, b1, b2, wn, bn)"
    if text.count(old) != 1:
        raise ValueError("Unexpected upstream mlp call; re-review the patch")
    text = text.replace(old, "    from minifold.model.kernels.mlp import mlp_kernel\n\n" + old)
    old = "    x = gating_kernel(x, gi_w, pi_w, gi_b, pi_b, ni_w, ni_b)"
    if text.count(old) != 1:
        raise ValueError("Unexpected upstream gating call; re-review the patch")
    text = text.replace(old, "    from minifold.model.kernels.gating import gating_kernel\n\n" + old)
    text = _patch_triangular_device(text)
    source.write_text(text)

    source = root / "minifold/utils/esm.py"
    text = source.read_text()
    start, end = text.index("def load_hub_workaround(url):"), text.index("\ndef load_regression_hub(")
    previous = text[start:end]
    if previous.count("torch.hub.load_state_dict_from_url") != 1 or previous.count("torch.load(") != 1:
        raise ValueError("Unexpected upstream ESM loader; re-review the patch")
    replacement = (
        "def load_hub_workaround(url):\n"
        "    from boileroom.models.minifold.loading import load_esm_artifact\n\n"
        "    return load_esm_artifact(url)\n\n"
    )
    source.write_text(text[:start] + replacement + text[end:])


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
