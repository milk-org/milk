#!/bin/bash

set -uo pipefail

MILK_SCRIPT="${MILK_SCRIPT:-milk-script}"
PASS=0
FAIL=0

cmd_exists() {
    echo "helpfull" | "$MILK_SCRIPT" - 2>&1 | grep -q "^$1"
}

check() {
    local desc="$1" key="$2"
    if cmd_exists "$key"; then
        printf "  PASS  %s (%s)\n" "$desc" "$key"
        PASS=$(( PASS + 1 ))
    else
        printf "  FAIL  %s — command '%s' not found\n" "$desc" "$key"
        FAIL=$(( FAIL + 1 ))
    fi
}

echo "=== milk-script coremod linkage smoke tests ==="

# Basic startup
if echo "echo hello" | "$MILK_SCRIPT" - 2>&1 | grep -q "hello"; then
    printf "  PASS  startup (echo)\n"
    PASS=$(( PASS + 1 ))
else
    printf "  FAIL  startup — echo produced unexpected output\n"
    FAIL=$(( FAIL + 1 ))
fi

# One representative command key per coremod
check "COREMOD_memory" "listim"
check "COREMOD_arith"  "imzero"
check "COREMOD_tools"  "fileutils"

# iofits is optional
if cmd_exists "loadfits"; then
    printf "  PASS  COREMOD_iofits (loadfits)\n"
    PASS=$(( PASS + 1 ))
else
    printf "  SKIP  COREMOD_iofits (USE_CFITSIO=OFF or not linked)\n"
fi

echo ""
printf "Results: %d passed, %d failed\n" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
