#!/bin/sh
# Last step of the ESMFold2 kit image, after kit_wheels.sh: install the kit and boileroom's own runtime dependencies.
#
# `run.sh install` installs the kit's shared core (opt_core) and esmfold2_opt editable from /kit, then checks the install
# against the kit's pins (stock/check_pins.py). The kit's lock already carries numpy, biotite and pydantic; fastapi and
# uvicorn are what boileroom's Apptainer service imports on top. The pin check runs again afterwards so that nothing the
# kit pinned moved. Last, kit_smoke.py imports the compiled kernels and checks the fork's import-time switches, so that a
# build whose fast paths would not run fails here instead of at the first fold (it runs on a CPU builder: no kernel launch).
set -eu
cd /kit/esmfold2
bash run.sh install
python -m pip install --no-cache-dir fastapi==0.136.3 uvicorn==0.48.0
python -m pip check
python -I stock/check_pins.py
mkdir -p /opt/jit_cache
chmod -R a+rX,u+w /opt/jit_cache
python -I /opt/boileroom-kit/kit_smoke.py
