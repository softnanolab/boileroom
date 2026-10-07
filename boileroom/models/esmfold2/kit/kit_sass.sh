#!/bin/sh
# Check that each compiled CUDA extension wheel carries machine code (SASS) for every compute capability of the stack
# (step of the ESMFold2 kit image, run by kit_wheels.sh while the CUDA toolkit is still installed).
#
# Usage: kit_sass.sh <wheel directory> <archs>, where <archs> is the stack's list without dots, e.g. "80;90".
#
# kit_smoke.py only imports the kernels, on a builder without a GPU, so it cannot see an extension that compiled no code
# for a card: that image would build, then fail at the first kernel launch on that card ("no kernel image is
# available"), after the ~27 GB weight fetch. Here every shared object of the flash_attn, transformer_engine and xformers
# wheels is listed with `cuobjdump --list-elf`; across a wheel's objects, each `sm_<arch>` must appear at least once
# (sm_90a counts as sm_90). A wheel may split its kernels over several objects, some for one architecture only, so the
# check is per wheel, not per object. Exits 2, naming the wheel and the missing architectures, otherwise.
set -eu
W=$1
ARCHS=$2
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
status=0
for wheel in flash_attn transformer_engine xformers; do
  set -- "$W/$wheel"-*.whl
  if [ ! -f "$1" ]; then
    echo "esmfold2 kernel check: no $wheel wheel in $W" >&2
    status=2
    continue
  fi
  rm -rf "$TMP/unpacked"; mkdir -p "$TMP/unpacked"
  python -m zipfile -e "$1" "$TMP/unpacked"
  : > "$TMP/found"
  find "$TMP/unpacked" -type f -name '*.so*' | while IFS= read -r object; do
    # An object without device code makes cuobjdump fail; it adds nothing to the list.
    cuobjdump --list-elf "$object" 2>/dev/null | grep -o 'sm_[0-9]*' >> "$TMP/found" || true
  done
  found=$(sort -u "$TMP/found" | tr '\n' ' '); found=${found% }
  missing=""
  for arch in $(printf %s "$ARCHS" | tr ';' ' '); do
    grep -qx "sm_$arch" "$TMP/found" || missing="$missing sm_$arch"
  done
  if [ -n "$missing" ]; then
    echo "esmfold2 kernel check: $(basename "$1") has no machine code for$missing (found: ${found:-none})" >&2
    status=2
  else
    echo "esmfold2 kernel check: $(basename "$1"): $found"
  fi
done
exit $status
