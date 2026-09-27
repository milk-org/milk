#!/usr/bin/env bash
# ============================================================
# milk-cli Hardening & Crash Prevention Verification Suite
# ============================================================
#
# Validates all vulnerability mitigations implemented during
# the CLI hardening effort:
#  1. Startup argument handling (-n "", -n ., unknown opts)
#  2. Interpreter recursion limits (function recursion & recursive source)
#  3. Arithmetic edge cases (LONG_MIN % -1, shift bounds, div zero)
#  4. Buffer expansion safety (brace range limits, here-strings)
#  5. FIFO input handling (non-blocking EOF, large buffer feeds)
#  6. Signal fault isolation & crash interception
#
# Usage:
#   bash tests/cli/test_cli_hardening.sh [path/to/milk-cli]
# ============================================================

set -uo pipefail

MILK_BIN="${1:-_build/milk-cli}"

if [[ ! -x "$MILK_BIN" ]]; then
    if which milk-cli >/dev/null 2>&1; then
        MILK_BIN="$(which milk-cli)"
    else
        echo "Error: milk-cli binary not found at $MILK_BIN"
        exit 1
    fi
fi

RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[0;33m'
CYN='\033[0;36m'
RST='\033[0m'

TOTAL=0
PASS=0
FAIL=0

run_check() {
    local test_name="$1"
    local cmd="$2"
    local expect_status="$3"  # "0", "nonzero", or "nocrash"
    local expect_pattern="${4:-}"

    TOTAL=$((TOTAL + 1))
    printf "  [%02d] %-55s " "$TOTAL" "$test_name"

    local out
    local status=0
    out=$(bash -c "$cmd" < /dev/null 2>&1) || status=$?

    # Check for crash exit codes (SIGSEGV=139, SIGABRT=134, SIGFPE=136, SIGBUS=135)
    if [[ $status -ge 128 && $status -le 160 ]]; then
        echo -e "${RED}CRASH (exit $status)${RST}"
        echo "       Command: $cmd"
        echo "       Output: $out"
        FAIL=$((FAIL + 1))
        return 1
    fi

    if [[ "$expect_status" == "0" && $status -ne 0 ]]; then
        echo -e "${RED}FAIL (expected 0, got $status)${RST}"
        echo "       Output: $out"
        FAIL=$((FAIL + 1))
        return 1
    elif [[ "$expect_status" == "nonzero" && $status -eq 0 ]]; then
        echo -e "${RED}FAIL (expected error, got 0)${RST}"
        FAIL=$((FAIL + 1))
        return 1
    fi

    if [[ -n "$expect_pattern" ]]; then
        if ! echo "$out" | grep -qiE "$expect_pattern"; then
            echo -e "${RED}FAIL (missing pattern: $expect_pattern)${RST}"
            echo "       Output: $out"
            FAIL=$((FAIL + 1))
            return 1
        fi
    fi

    echo -e "${GRN}PASS${RST}"
    PASS=$((PASS + 1))
    return 0
}

TMPDIR="$(mktemp -d /tmp/milk_harden_test.XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

echo -e "${CYN}═════════════════════════════════════════════════════════════════${RST}"
echo -e "${CYN} milk-cli Hardening & Crash Prevention Test Suite${RST}"
echo -e "${CYN} Target: ${MILK_BIN}${RST}"
echo -e "${CYN}═════════════════════════════════════════════════════════════════${RST}"

# 1. Startup option handling
echo -e "\n${YLW}--- Section 1: Startup & Options ---${RST}"
run_check "Option -n empty string (-n \"\")" \
    "$MILK_BIN -n \"\" -c \"echo ok\"" "0" "ok"

run_check "Option -n single dot (-n .)" \
    "$MILK_BIN -n . -c \"echo ok\"" "0" "ok"

run_check "Option -n multiple dots (-n a.b.c)" \
    "$MILK_BIN -n a.b.c -c \"echo ok\"" "0" "ok"

run_check "Unknown option returns error (no abort)" \
    "$MILK_BIN -x" "nonzero"

# 2. Call Stack & Recursion Protection
echo -e "\n${YLW}--- Section 2: Call Stack & Recursion ---${RST}"
cat << 'EOF' > "$TMPDIR/rec_func.milk"
function rec {
    rec
}
rec
EOF
run_check "Function infinite recursion bounded" \
    "$MILK_BIN -s $TMPDIR/rec_func.milk" "nocrash" "recursion"

# Recursive source test
cat << EOF > "$TMPDIR/rec_source.milk"
source $TMPDIR/rec_source.milk
EOF
run_check "Script source recursion bounded" \
    "$MILK_BIN -s $TMPDIR/rec_source.milk" "nocrash" "recursion"

# 3. Arithmetic Safety
echo -e "\n${YLW}--- Section 3: Arithmetic & Hardware Exceptions ---${RST}"
run_check "Hardware SIGFPE: LONG_MIN % -1" \
    "$MILK_BIN -c 'calc -9223372036854775808 % -1'" "nocrash"

run_check "Hardware SIGFPE: LONG_MIN / -1" \
    "$MILK_BIN -c 'calc -9223372036854775808 / -1'" "nocrash"

run_check "Arithmetic divide by zero" \
    "$MILK_BIN -c 'calc 100 / 0'" "nocrash"

run_check "Arithmetic modulo by zero" \
    "$MILK_BIN -c 'calc 100 % 0'" "nocrash"

run_check "Bit shift overflow (1 << 64)" \
    "$MILK_BIN -c 'echo \$(( 1 << 64 ))'" "0"

run_check "Bit shift overflow (1 >> 100)" \
    "$MILK_BIN -c 'echo \$(( 1 >> 100 ))'" "0"

# 4. Expansion Buffers & Syntax Safety
echo -e "\n${YLW}--- Section 4: Buffer Safety & Parsing ---${RST}"
run_check "Brace expansion large range bounded ({1..100000})" \
    "$MILK_BIN -c 'echo {1..100000}'" "0"

run_check "Compound subshell parens preserved" \
    "$MILK_BIN -c '(echo sub1) && (echo sub2)'" "0" "sub2"

run_check "Here-string syntax safety" \
    "$MILK_BIN -c 'cat <<< \"herestring test\"'" "nocrash"

# 5. FIFO Safety
echo -e "\n${YLW}--- Section 5: FIFO Input Robustness ---${RST}"
FIFO_NAME="$TMPDIR/test_fifo"
mkfifo "$FIFO_NAME"
# Test feeding 4096 bytes through FIFO without crashing or hanging
(
    sleep 0.2
    python3 -c "print('echo fifo_ok;' + 'a' * 4096)" > "$FIFO_NAME" 2>/dev/null || true
    sleep 0.2
    echo "exit" > "$FIFO_NAME" 2>/dev/null || true
) &
FIFO_FEED_PID=$!
run_check "Large input stream through FIFO (>1024 bytes)" \
    "timeout 5 $MILK_BIN -f -F \"$FIFO_NAME\"" "nocrash"
wait $FIFO_FEED_PID 2>/dev/null || true

# 6. Interactive Fault Isolation & Signal Hardening
echo -e "\n${YLW}--- Section 6: Interactive Fault Isolation ---${RST}"
cat << 'EOF' > "$TMPDIR/test_segv_recovery.py"
import pty, os, time, subprocess, signal, sys

milk_bin = sys.argv[1]
master, slave = pty.openpty()
proc = subprocess.Popen([milk_bin], stdin=slave, stdout=slave, stderr=slave, close_fds=True)
os.close(slave)

time.sleep(0.5)
os.write(master, b'sleep 3\n')
time.sleep(0.3)
os.kill(proc.pid, signal.SIGSEGV)
time.sleep(0.3)
os.write(master, b'echo SURVIVED_SEGV\nexit\n')
time.sleep(0.5)

output = b''
while True:
    try:
        data = os.read(master, 1024)
        if not data: break
        output += data
    except OSError:
        break
os.close(master)
proc.wait(timeout=2)
out_str = output.decode('utf-8', errors='replace')
if 'SURVIVED_SEGV' in out_str and 'CRASH INTERCEPTED' in out_str:
    sys.exit(0)
sys.exit(1)
EOF

run_check "Interactive fault isolation intercepts SIGSEGV" \
    "python3 $TMPDIR/test_segv_recovery.py $MILK_BIN" "0"

cat << 'EOF' > "$TMPDIR/test_sigint_recovery.py"
import pty, os, time, subprocess, sys

milk_bin = sys.argv[1]
master, slave = pty.openpty()
proc = subprocess.Popen([milk_bin], stdin=slave, stdout=slave, stderr=slave, close_fds=True)
os.close(slave)

time.sleep(0.5)
os.write(master, b'sleep 5\n')
time.sleep(0.3)
os.write(master, b'\x03')
time.sleep(0.3)
os.write(master, b'echo SURVIVED_SIGINT\nexit\n')
time.sleep(0.5)

output = b''
while True:
    try:
        data = os.read(master, 1024)
        if not data: break
        output += data
    except OSError:
        break
os.close(master)
proc.wait(timeout=2)
out_str = output.decode('utf-8', errors='replace')
if 'SURVIVED_SIGINT' in out_str:
    sys.exit(0)
sys.exit(1)
EOF

run_check "Interactive Ctrl+C cancels command safely" \
    "python3 $TMPDIR/test_sigint_recovery.py $MILK_BIN" "0"

echo -e "\n${CYN}═════════════════════════════════════════════════════════════════${RST}"
echo -e " Hardening Test Summary: ${GRN}$PASS passed${RST}, ${RED}$FAIL failed${RST} (total $TOTAL)"
echo -e "${CYN}═════════════════════════════════════════════════════════════════${RST}"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
exit 0
