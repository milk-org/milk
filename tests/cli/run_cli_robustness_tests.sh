#!/usr/bin/env bash
# ============================================================
# CLI Robustness Test Runner
# ============================================================
#
# Feeds test cases from cli_robustness_tests.milk
# to milk-cli one block at a time, capturing output
# and checking for crashes, hangs, and missing error
# messages.
#
# Usage:
#   bash run_cli_robustness_tests.sh [OPTIONS]
#
# Options:
#   --verbose         Show details for failures
#   --filter PATTERN  Run only test descriptions
#                     matching glob PATTERN
#   --stop-on-fail    Stop after first failure
#
# Exit codes:
#   0 — all tests passed
#   1 — one or more tests failed
#
# Output:
#   - Summary table on stdout
#   - Detailed report in cli_test_report.txt
#
# Requirements:
#   - milk-cli must be on PATH
#   - timeout(1) (coreutils)
# ============================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_FILE="${SCRIPT_DIR}/cli_robustness_tests.milk"
TIMEOUT_SEC=10
VERBOSE=0
FILTER=""
STOP_ON_FAIL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verbose)     VERBOSE=1 ;;
        --filter)      FILTER="${2:-}"; shift ;;
        --stop-on-fail) STOP_ON_FAIL=1 ;;
    esac
    shift
done

# ---- Color codes ----
RED='\033[0;31m'
GRN='\033[0;32m'
YLW='\033[0;33m'
CYN='\033[0;36m'
DIM='\033[2m'
RST='\033[0m'

# ---- Temp directory ----
TMPDIR="$(mktemp -d /tmp/clitest.XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

REPORT="${TMPDIR}/report.txt"
> "$REPORT"

# ---- Counters ----
TOTAL=0
PASS=0
FAIL=0
ERRFAIL=0    # Expected error but no message
CRASH=0
HANG=0
TOTAL_TIME_MS=0
SLOWEST_MS=0
SLOWEST_DESC=""

echo -e "${CYN}═══════════════════════════════════════════${RST}"
echo -e "${CYN} milk-cli Robustness Test Suite${RST}"
echo -e "${CYN}═══════════════════════════════════════════${RST}"
echo ""

# ============================================================
# Parse test file into blocks
# ============================================================
#
# A "test block" is:
#   #DESC: description
#   #EXPECT:OK or #EXPECT:ERR
#   one or more command lines (until next #DESC or EOF)
#
# Multi-line constructs (if/while/for/function/case)
# are accumulated into a single block.

declare -a TEST_DESC=()
declare -a TEST_EXPECT=()
declare -a TEST_CMD=()

cur_desc=""
cur_expect=""
cur_cmd=""

flush_test() {
    if [[ -n "$cur_desc" && -n "$cur_cmd" ]]; then
        TEST_DESC+=("$cur_desc")
        TEST_EXPECT+=("$cur_expect")
        TEST_CMD+=("$cur_cmd")
    fi
    cur_desc=""
    cur_expect=""
    cur_cmd=""
}

while IFS= read -r line || [[ -n "$line" ]]; do
    # Check for DESC annotation
    if [[ "$line" =~ ^#DESC:\ *(.*) ]]; then
        flush_test
        cur_desc="${BASH_REMATCH[1]}"
        continue
    fi

    # Check for EXPECT annotation
    if [[ "$line" =~ ^#EXPECT:(OK|ERR) ]]; then
        cur_expect="${BASH_REMATCH[1]}"
        continue
    fi

    # Skip pure comment lines (not annotations)
    stripped="${line#"${line%%[![:space:]]*}"}"
    if [[ -z "$stripped" || "$stripped" == \#* ]]; then
        # If we have an active test block with
        # no commands yet, this is just spacing
        if [[ -z "$cur_cmd" ]]; then
            continue
        fi
        # Otherwise include blank/comment in
        # multi-line blocks
        continue
    fi

    # Accumulate command lines
    if [[ -n "$cur_desc" ]]; then
        if [[ -n "$cur_cmd" ]]; then
            cur_cmd="${cur_cmd}
${line}"
        else
            cur_cmd="$line"
        fi
    fi
done < "$TEST_FILE"
flush_test

NUM_TESTS=${#TEST_DESC[@]}
echo -e "Parsed ${CYN}${NUM_TESTS}${RST} test blocks"
echo ""

# ============================================================
# Run tests
# ============================================================

run_one_test() {
    local idx="$1"
    local desc="${TEST_DESC[$idx]}"
    local expect="${TEST_EXPECT[$idx]}"
    local cmd="${TEST_CMD[$idx]}"

    # Skip if filter is set and doesn't match
    if [[ -n "$FILTER" ]]; then
        case "$desc" in
            *${FILTER}*) ;;
            *) return 0 ;;
        esac
    fi

    local cmd_file="${TMPDIR}/cmd_${idx}.milk"
    local out_file="${TMPDIR}/out_${idx}.txt"
    local err_file="${TMPDIR}/err_${idx}.txt"

    # Write commands to a temp script
    echo "$cmd" > "$cmd_file"
    echo "exit" >> "$cmd_file"

    TOTAL=$((TOTAL + 1))

    # Run milk-cli with the test script, capturing
    # stdout and stderr, with timeout
    local exit_code=0
    local t_start t_end elapsed_ms
    t_start=$(date +%s%N)
    timeout "${TIMEOUT_SEC}" milk-cli \
        --no-autocomplete \
        --no-history-suggest \
        --no-arg-hints \
        -s "$cmd_file" \
        -f \
        > "$out_file" 2>"$err_file" < /dev/null \
        || exit_code=$?
    t_end=$(date +%s%N)
    elapsed_ms=$(( (t_end - t_start) / 1000000 ))
    TOTAL_TIME_MS=$((TOTAL_TIME_MS + elapsed_ms))
    if [[ $elapsed_ms -gt $SLOWEST_MS ]]; then
        SLOWEST_MS=$elapsed_ms
        SLOWEST_DESC="$desc"
    fi

    local result="PASS"
    local detail=""

    # Check for timeout (exit 124)
    if [[ $exit_code -eq 124 ]]; then
        result="HANG"
        detail="Command timed out after ${TIMEOUT_SEC}s"
        HANG=$((HANG + 1))
    # Check for crash (signal exits: 128+signal)
    elif [[ $exit_code -ge 128 ]]; then
        local sig=$((exit_code - 128))
        result="CRASH"
        detail="Killed by signal $sig"
        CRASH=$((CRASH + 1))
    # Check expected outcome
    elif [[ "$expect" == "OK" ]]; then
        if [[ $exit_code -ne 0 ]]; then
            # Check if it's a minor issue —
            # some commands return non-zero
            # but are still "working"
            result="FAIL"
            detail="Expected success, got exit=$exit_code"
            FAIL=$((FAIL + 1))
        else
            PASS=$((PASS + 1))
        fi
    elif [[ "$expect" == "ERR" ]]; then
        # We expect an error — check that an
        # error message was printed
        local combined
        combined="$(cat "$out_file" "$err_file")"
        if echo "$combined" | \
            grep -qiE \
            '(error|usage|missing|cannot|not found|does not exist|wrong|invalid|did you mean|unknown|no such)'; then
            PASS=$((PASS + 1))
        else
            if [[ $exit_code -eq 0 ]]; then
                result="MISSING_ERROR"
                detail="Expected error message but got none (exit=0)"
                ERRFAIL=$((ERRFAIL + 1))
            else
                # Non-zero exit but no message
                result="MISSING_ERROR"
                detail="Exit=$exit_code but no error message found"
                ERRFAIL=$((ERRFAIL + 1))
            fi
        fi
    fi

    # Print result
    local color="$GRN"
    case "$result" in
        FAIL)          color="$RED" ;;
        CRASH)         color="$RED" ;;
        HANG)          color="$RED" ;;
        MISSING_ERROR) color="$YLW" ;;
    esac

    printf "  [%3d/%3d] ${color}%-14s${RST} %s" \
        "$TOTAL" "$NUM_TESTS" "$result" "$desc"
    if [[ -n "$detail" ]]; then
        printf "  ${DIM}(%s)${RST}" "$detail"
    fi
    printf "  ${DIM}[%dms]${RST}" "$elapsed_ms"
    echo ""

    if [[ $VERBOSE -eq 1 && "$result" != "PASS" ]]; then
        echo -e "    ${DIM}Command: $(head -1 "$cmd_file")${RST}"
        echo -e "    ${DIM}Stdout:  $(head -3 "$out_file" | \
            tr '\n' '|')${RST}"
        echo -e "    ${DIM}Stderr:  $(head -3 "$err_file" | \
            tr '\n' '|')${RST}"
    fi

    # Write to report
    {
        echo "--- Test $TOTAL: $desc ---"
        echo "Expect: $expect"
        echo "Result: $result"
        echo "Exit:   $exit_code"
        if [[ -n "$detail" ]]; then
            echo "Detail: $detail"
        fi
        echo "Command:"
        cat "$cmd_file"
        echo "Stdout:"
        cat "$out_file"
        echo "Stderr:"
        cat "$err_file"
        echo ""
    } >> "$REPORT"
}

for i in $(seq 0 $((NUM_TESTS - 1))); do
    run_one_test "$i"
    if [[ $STOP_ON_FAIL -eq 1 ]]; then
        PROBLEMS=$((FAIL + ERRFAIL + CRASH + HANG))
        if [[ $PROBLEMS -gt 0 ]]; then
            echo ""
            echo -e "${RED}Stopping on first failure (--stop-on-fail)${RST}"
            break
        fi
    fi
done

# ============================================================
# Test -c Flag
# ============================================================
TOTAL=$((TOTAL + 1))
exit_code=0
c_output=$(timeout "${TIMEOUT_SEC}" milk-cli -c "echo c_flag_test" 2>/dev/null) || exit_code=$?

if [[ $exit_code -eq 0 && "$c_output" == *"c_flag_test"* ]]; then
    PASS=$((PASS + 1))
    printf "  [%3d/%3d] ${GRN}%-14s${RST} %s\n" "$TOTAL" "$((NUM_TESTS + 1))" "PASS" "Test -c flag execution"
else
    FAIL=$((FAIL + 1))
    printf "  [%3d/%3d] ${RED}%-14s${RST} %s\n" "$TOTAL" "$((NUM_TESTS + 1))" "FAIL" "Test -c flag execution (got exit=$exit_code, output=$c_output)"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo -e "${CYN}═══════════════════════════════════════════${RST}"
echo -e "${CYN} Summary${RST}"
echo -e "${CYN}═══════════════════════════════════════════${RST}"
echo ""
printf "  Total:         %d\n" "$TOTAL"
printf "  ${GRN}Pass:          %d${RST}\n" "$PASS"
printf "  ${RED}Fail:          %d${RST}\n" "$FAIL"
printf "  ${YLW}Missing Error: %d${RST}\n" "$ERRFAIL"
printf "  ${RED}Crash:         %d${RST}\n" "$CRASH"
printf "  ${RED}Hang:          %d${RST}\n" "$HANG"
echo ""
printf "  Total time:    %d.%03ds\n" \
    "$((TOTAL_TIME_MS / 1000))" \
    "$((TOTAL_TIME_MS % 1000))"
if [[ -n "$SLOWEST_DESC" ]]; then
    printf "  Slowest test:  %dms (%s)\n" \
        "$SLOWEST_MS" "$SLOWEST_DESC"
fi
echo ""

# Copy report to working directory
FINAL_REPORT="${SCRIPT_DIR}/cli_test_report.txt"
cp "$REPORT" "$FINAL_REPORT"
echo -e "Detailed report: ${CYN}${FINAL_REPORT}${RST}"
echo ""

PROBLEMS=$((FAIL + ERRFAIL + CRASH + HANG))
if [[ $PROBLEMS -eq 0 ]]; then
    echo -e "${GRN}All tests passed!${RST}"
    exit 0
else
    echo -e "${RED}${PROBLEMS} problem(s) found.${RST}"
    exit 1
fi
