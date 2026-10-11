"""Apply minimal, checked build-time fixes to the pinned upstream MiniFold source.

The upstream wheel omits subpackages, and eager Triton decorators initialize a
CUDA driver even when kernels=False. Lazy imports change no inference maths.
"""

import sys
from pathlib import Path


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
