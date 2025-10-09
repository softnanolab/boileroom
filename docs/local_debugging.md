Local Debugging Steps
====================
To debug, or just simply run in the local environment, one can execute the `LocalBackend`.
This, however, requires that the execution python environment (of the main process) has all the required dependencies.

1. Install project dependencies manually: `uv pip install .`
2. Some models specific version of some packages (e.g. NumPy, thus install it with: `uv pip install "numpy<2.0"`). 
See the table for full description for each model.
3. Activate the virtual environment: `source .venv/bin/activate`
4. Confirm the interpreter comes from the environment: `which python`
5. Execute the script: `python script.py`

Backend Requirements
--------------------

| Backend | Extra | Additional Step |
|---------|-------|-----------------|
| Boltz-2 | boltz | Install `numpy<2.0` |
| Chai-1  | chai  | Install `numpy<2.0` |

Example Usage
-------------

```python
from boileroom import Chai1
model = Chai1(backend="local")
result = model.fold([sequence])
```

```
boltz = [
    "boltz==2.1.1",
]
chai = [
    "chai_lab==0.6.1",
    "hf_transfer==0.1.8",
]
```

Chai requires some rdkit, that is not supported on some older CentOS, hence need to do:
For this, we should make Docker/Apptainer images -- then one can debug in the Local environment.
```
micromamba install python=3.12 rdkit
pip install chai_lab==0.6.1 hf_transfer==0.1.8
```

This is a bit tricky, and I would have to implement Apptainer images to some extent - that is possible but not a priority.