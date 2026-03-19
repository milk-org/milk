#!/usr/bin/env bash
# perfbench test wrapper for clitest
#
# Creates the dummy stream required by
# milk-fpsexec-clitest, then runs the
# benchmark.

set -euo pipefail

NBITER="${1:-100}"
OUTDIR="${2:-.}"

# Resolve milk-perfbench from same dir
SCRIPTDIR="$(cd "$(dirname "$0")/.." \
    && pwd)/scripts"
export PATH="${SCRIPTDIR}:${PATH}"

# Create dummy input stream (cam01)
milk-perfbench-mkstream cam01 10 10

# Run benchmark
milk-perfbench \
    milk-fpsexec-clitest "${NBITER}" \
    -o "${OUTDIR}"
