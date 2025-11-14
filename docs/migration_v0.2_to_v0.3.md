# Migration Guide: v0.2 to v0.3

This guide helps you upgrade from boileroom v0.2 to v0.3.

## Breaking Changes

### Optional Dependency Key Renamed: `local` → `esm`

**What changed:**
The optional dependency key in `pyproject.toml` was renamed from `local` to `esm` in v0.3.0. This is a breaking change for users who install the package with the `[local]` extra.

**Before (v0.2):**
```bash
pip install boileroom[local]
```

**After (v0.3):**
```bash
pip install boileroom[esm]
```

**Why the change:**
The dependency key was renamed to better reflect its purpose: it installs dependencies required for ESM models (ESMFold and ESM-2). The name `esm` is more descriptive and aligns with the model-specific extras pattern used elsewhere in the project.

**Action required:**
If you were previously installing with `pip install boileroom[local]`, update your installation commands, CI/CD scripts, and documentation to use `pip install boileroom[esm]` instead.

**Example:**
```bash
# Old (v0.2)
pip install boileroom[local]

# New (v0.3)
pip install boileroom[esm]
```

### High-Level API Simplification

**What changed:**
The high-level API has been simplified in v0.3.0. You no longer need to import and use the `app` context manager, or call `.remote()` on model methods. Instead, models now use a backend parameter and handle execution automatically.

**Before (v0.2):**
```python
from boileroom import app, ESMFold

# Initialize the model
model = ESMFold()

# Predict structure for a protein sequence
sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"

with app.run():
    result = model.fold.remote([sequence])

# Access prediction results
coordinates = result.positions
confidence = result.plddt
```

**After (v0.3):**
```python
from boileroom import ESMFold

# Initialize the model
model = ESMFold(backend="modal")

# Predict structure for a protein sequence
sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"

result = model.fold([sequence])

# Access prediction results
# Extract coordinates from atom_array (which contains full metadata)
atom_array = result.atom_array[0]  # Get first structure
coordinates = atom_array.coord  # Get coordinates
confidence = result.plddt
```

**Why the change:**
The new API simplifies usage by:
- Removing the need for explicit context managers (`with app.run()`)
- Eliminating the `.remote()` method call requirement
- Providing a unified interface that works with both Modal and other backends (some already implemented)
- Making the backend selection explicit via the `backend` parameter

**Action required:**
1. Remove `app` from your imports
2. Add `backend="modal"` (or `backend="local"`) when initializing models
3. Remove `with app.run():` context managers
4. Remove `.remote()` calls from method invocations

**Key differences:**
- Model initialization: `ESMFold()` → `ESMFold(backend="modal")`
- Method calls: `model.fold.remote([sequence])` → `model.fold([sequence])`
- Context manager: No longer needed

## Other Changes

_Add other breaking changes, deprecations, or important updates here as needed._

