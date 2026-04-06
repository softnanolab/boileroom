# Migration Guide: v0.2 to v0.3

This guide covers the user-visible changes that matter when upgrading from `boileroom` `v0.2.x` to `v0.3.x`.

If you only used the high-level Python API, there are three things to check:

1. Stop using `app.run()` and `.remote()`.
2. Request optional output fields explicitly.
3. Rename the `local` extra to `esm` if you installed it.

## 1. High-Level API No Longer Uses `app.run()` or `.remote()`

`v0.3` uses the high-level model classes directly. You no longer need to import `app`, open a Modal context manager, or call `.remote()`.

### Before (`v0.2`)

```python
from boileroom import ESMFold, app

sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"
model = ESMFold()

with app.run():
    result = model.fold.remote([sequence])
```

### After (`v0.3`)

```python
from boileroom import ESMFold

sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"
model = ESMFold()
result = model.fold([sequence])
```

### What You Need To Change

- Remove `app` from your imports.
- Remove `with app.run():`.
- Replace `.remote(...)` with a normal method call such as `.fold(...)` or `.embed(...)`.

## 2. Optional Outputs Are Now Opt-In

`v0.3` returns a smaller default payload. Structure models always return `metadata` and `atom_array`, but optional fields such as `plddt`, `pae`, `pdb`, `cif`, and model-specific confidence outputs are no longer assumed to be present unless you request them.

This is the biggest behavioral change to watch for if your `v0.2` code accessed extra result fields directly.

### Before (`v0.2`)

```python
from boileroom import ESMFold, app

sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"
model = ESMFold()

with app.run():
    result = model.fold.remote([sequence])

coordinates = result.positions
confidence = result.plddt
```

### After (`v0.3`)

```python
from boileroom import ESMFold

sequence = "MLKNVHVLVLGAGDVGSVVVRLLEK"
model = ESMFold()

result = model.fold([sequence], options={"include_fields": ["pdb"]})

atom_array = result.atom_array[0]
coordinates = atom_array.coord
pdb_string = result.pdb[0]
```

### If You Need Optional Fields

Request them per call:

```python
result = model.fold(
    [sequence],
    options={"include_fields": ["plddt", "pae", "pdb", "cif"]},
)
```

Or set them in the model config if you want the same fields every time:

```python
model = ESMFold(config={"include_fields": ["plddt", "pae"]})
result = model.fold([sequence])
```

### What You Need To Change

- Audit code that reads fields such as `plddt`, `pae`, `pde`, `pdb`, `cif`, `hidden_states`, or other non-minimal outputs.
- Add `include_fields` wherever those fields are required.
- Prefer `atom_array` as the default structure representation.

## 3. Optional Dependency Extra Renamed: `local` -> `esm`

If you were installing the old `local` extra, rename it to `esm`.

### Before (`v0.2`)

```bash
pip install boileroom[local]
```

### After (`v0.3`)

```bash
pip install boileroom[esm]
```

### What You Need To Change

- Update installation commands.
- Update CI setup and environment files.
- Update any internal docs or notebooks that still reference `boileroom[local]`.

## Backend Selection

The default high-level backend remains Modal, so most users do not need to pass `backend="modal"` explicitly.

Only specify `backend` when you want something other than the default, for example:

```python
from boileroom import ESMFold

model = ESMFold(backend="apptainer")
result = model.fold(["MLKNVHVLVLGAGDVGSVVVRLLEK"])
```

Valid examples in `v0.3` are `backend="modal"` and `backend="apptainer"`. `backend="local"` is not a valid replacement.

## Quick Upgrade Checklist

- Remove `app.run()` usage.
- Remove `.remote()` calls.
- Rename `boileroom[local]` to `boileroom[esm]` if applicable.
- Add `include_fields` anywhere you rely on optional outputs.
- Switch code that previously read raw coordinates from legacy fields to `result.atom_array`.
