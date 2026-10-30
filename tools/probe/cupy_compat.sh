#!/usr/bin/env bash
# Probe helper on the AMD GPU computer: CuPy (cupy-rocm-7-0) asks for versioned ROCm 7 libraries
# (libamdhip64.so.7, ...). The PyTorch ROCm 7.2 wheel has the same libraries under unversioned names. This script
# makes .local/rocm7-compat/ with versioned links to them, and prints the libraries that are still missing.
# It deletes nothing (it only adds links in the repository folder). Run: bash tools/probe/cupy_compat.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
P=.venv-probe/bin/python
SP=$($P -c "import site; print(site.getsitepackages()[0])")
TLIB=$SP/torch/lib
COMPAT=$PWD/.local/rocm7-compat
mkdir -p "$COMPAT"
for round in 1 2 3 4; do
  missing=$(find "$SP/cupy" "$SP/cupy_backends" -name "*.so" -print0 2>/dev/null \
    | LD_LIBRARY_PATH=$COMPAT:$TLIB xargs -0 ldd 2>/dev/null | awk '/not found/ {print $1}' | sort -u)
  [[ -z $missing ]] && break
  for so in $missing; do
    base=${so%%.so*}.so
    if [[ -e $TLIB/$base && ! -e $COMPAT/$so ]]; then
      ln -s "$TLIB/$base" "$COMPAT/$so" && echo "link $so -> torch/lib/$base"
    fi
  done
done
echo "still missing:"; find "$SP/cupy" "$SP/cupy_backends" -name "*.so" -print0 2>/dev/null \
  | LD_LIBRARY_PATH=$COMPAT:$TLIB xargs -0 ldd 2>/dev/null | awk '/not found/ {print "  " $1}' | sort -u
echo "LD_LIBRARY_PATH=$COMPAT:$TLIB"
