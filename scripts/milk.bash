#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# milk.bash: First-class GNU Bash integration for milk
#
# Usage (source into any bash shell or script):
#   source milk.bash
#

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1:-}" in
        -h1|--help-oneline)
            echo "sourceable bash library for milk (@commands, @calc, @get, @set)"
            exit 0
            ;;
        -h|--help)
            echo "Usage: source milk.bash"
            echo ""
            echo "Loads milk commands (@<cmd>), entity accessors (@get, @set, @exists, @running),"
            echo "zero-fork math/slice evaluation (@calc, @test), stream synchronization (@waitfor, @grab),"
            echo "and LIFO cleanup handlers (@defer) into the current GNU Bash session."
            exit 0
            ;;
        *)
            echo -e "\033[1;31mERROR: milk.bash must be sourced into bash, not executed.\033[0m" >&2
            echo "Usage: source ${BASH_SOURCE[0]}" >&2
            exit 1
            ;;
    esac
fi

# ---------------------------------------------------------------------------
# 1. Resolve MILK_INSTALLDIR and update PATH / LD_LIBRARY_PATH
# ---------------------------------------------------------------------------
__milk_resolve_installdir() {
    if [[ -n "${MILK_INSTALLDIR:-}" && -x "${MILK_INSTALLDIR}/bin/milk-cli" ]]; then
        echo "${MILK_INSTALLDIR}"
        return 0
    fi
    local src_dir
    src_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
    if [[ -x "${src_dir}/milk-cli" ]]; then
        echo "$(cd "${src_dir}/.." && pwd)"
        return 0
    fi
    if [[ -x "${src_dir}/../../bin/milk-cli" ]]; then
        echo "$(cd "${src_dir}/../.." && pwd)"
        return 0
    fi
    if [[ -n "${MILK_INSTALLDIR:-}" ]]; then
        echo "${MILK_INSTALLDIR}"
        return 0
    fi
    echo "/usr/local/milk"
}

export MILK_INSTALLDIR="$(__milk_resolve_installdir)"
unset -f __milk_resolve_installdir

if [[ ":${PATH}:" != *":${MILK_INSTALLDIR}/bin:"* ]]; then
    export PATH="${MILK_INSTALLDIR}/bin:${PATH}"
fi
if [[ ":${LD_LIBRARY_PATH:-}:" != *":${MILK_INSTALLDIR}/lib:"* ]]; then
    export LD_LIBRARY_PATH="${MILK_INSTALLDIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# ---------------------------------------------------------------------------
# 2. Validate MILKCLI_SIGIL (default '@')
# ---------------------------------------------------------------------------
__milk_validate_sigil() {
    local s="${MILKCLI_SIGIL:-@}"
    if [[ ${#s} -lt 1 || ${#s} -gt 8 ]]; then
        echo "milk.bash: warning: invalid MILKCLI_SIGIL '${s}' (length must be 1..8); using '@'" >&2
        s="@"
    elif [[ "${s:0:1}" != "@" && "${s:0:1}" != "," ]]; then
        echo "milk.bash: warning: MILKCLI_SIGIL '${s}' must start with '@' or ','; using '@'" >&2
        s="@"
    elif [[ "${s}" =~ [^A-Za-z0-9_@,:./~%^+-] ]]; then
        echo "milk.bash: warning: MILKCLI_SIGIL '${s}' contains shell metacharacters; using '@'" >&2
        s="@"
    elif [[ ${#s} -gt 1 && "${s: -1}" =~ [A-Za-z0-9_] ]]; then
        echo "milk.bash: warning: multi-char MILKCLI_SIGIL '${s}' must end with a non-alphanumeric separator; using '@'" >&2
        s="@"
    fi
    export MILKCLI_SIGIL="${s}"
}
__milk_validate_sigil
unset -f __milk_validate_sigil

# ---------------------------------------------------------------------------
# 3. Session socket address (server starts lazily on first command)
# ---------------------------------------------------------------------------
if [[ -z "${MILKCLI_SOCK:-}" ]]; then
    export MILKCLI_SOCK="@milk-cli-${UID:-$(id -u)}-$$"
fi

# ---------------------------------------------------------------------------
# 4. Load C builtins (libmilkbash.so) or install milk-call fallback shims
# ---------------------------------------------------------------------------
_MILK_BUILTIN_LOADED=0
if [[ -f "${MILK_INSTALLDIR}/lib/libmilkbash.so" ]]; then
    if MILK_QUIET=1 enable -f "${MILK_INSTALLDIR}/lib/libmilkbash.so" \
        milkcall milkget milkset milkcalc milktest milkexists \
        milkwaitfor milkgrab milkcomplete milkshutdown 2>/dev/null; then
        _MILK_BUILTIN_LOADED=1
    fi
fi

if [[ ${_MILK_BUILTIN_LOADED} -eq 0 ]]; then
    milkcall() {
        local eval_file="${TMPDIR:-/tmp}/.milk_eval_$$_${BASHPID}"
        local rc=0
        exec {__milk_efd}>"${eval_file}"
        MILK_CALL_EVAL_FD="${__milk_efd}" "${MILK_INSTALLDIR}/bin/milk-call" call "$@" || rc=$?
        exec {__milk_efd}>&-
        if [[ -s "${eval_file}" ]]; then
            # shellcheck disable=SC1090
            source "${eval_file}"
        fi
        rm -f "${eval_file}"
        return ${rc}
    }
    milkget() {
        if [[ "${1:-}" == "-v" && -n "${2:-}" ]]; then
            local _vname="$2"
            shift 2
            local _val
            _val="$("${MILK_INSTALLDIR}/bin/milk-call" get "$@")" || return $?
            printf -v "${_vname}" '%s' "${_val}"
            return 0
        fi
        "${MILK_INSTALLDIR}/bin/milk-call" get "$@"
    }
    milkset()      { "${MILK_INSTALLDIR}/bin/milk-call" set "$@"; }
    milkcalc() {
        local eval_file="${TMPDIR:-/tmp}/.milk_eval_$$_${BASHPID}"
        local rc=0
        exec {__milk_efd}>"${eval_file}"
        MILK_CALL_EVAL_FD="${__milk_efd}" "${MILK_INSTALLDIR}/bin/milk-call" calc "$@" || rc=$?
        exec {__milk_efd}>&-
        if [[ -s "${eval_file}" ]]; then
            # shellcheck disable=SC1090
            source "${eval_file}"
        fi
        rm -f "${eval_file}"
        return ${rc}
    }
    milktest()     { "${MILK_INSTALLDIR}/bin/milk-call" test "$@"; }
    milkexists()   { "${MILK_INSTALLDIR}/bin/milk-call" exists "$@"; }
    milkwaitfor()  { "${MILK_INSTALLDIR}/bin/milk-call" waitfor "$@"; }
    milkgrab()     { "${MILK_INSTALLDIR}/bin/milk-call" grab "$@"; }
    milkcomplete() { "${MILK_INSTALLDIR}/bin/milk-call" complete "$@"; }
    milkshutdown() { "${MILK_INSTALLDIR}/bin/milk-call" shutdown "$@"; }
fi

# ---------------------------------------------------------------------------
# 5. Lifecycle helpers: @defer (LIFO cleanup) and @on (background watchers)
# ---------------------------------------------------------------------------
declare -ga _MILK_DEFER_STACK=()
declare -gi _MILK_DEFER_RAN=0
declare -ga _MILK_ON_PIDS=()

_milk_run_defer_stack() {
    local exit_rc=$?
    if [[ ${_MILK_DEFER_RAN} -ne 0 ]]; then
        return ${exit_rc}
    fi
    _MILK_DEFER_RAN=1
    local i
    for (( i=${#_MILK_DEFER_STACK[@]}-1; i>=0; i-- )); do
        eval "${_MILK_DEFER_STACK[$i]}" || true
    done
    _MILK_DEFER_STACK=()
    return ${exit_rc}
}

_milk_defer_add() {
    if [[ $# -eq 0 ]]; then
        return 0
    fi
    if [[ ${#_MILK_DEFER_STACK[@]} -ge 32 ]]; then
        echo "Error: defer stack full (max 32)" >&2
        return 1
    fi
    _MILK_DEFER_RAN=0
    _MILK_DEFER_STACK+=("$*")
    trap '_milk_run_defer_stack' EXIT
}

_milk_source_impl() {
    if [[ $# -eq 0 ]]; then
        echo "Usage: ${MILKCLI_SIGIL}source <filename> [args...]" >&2
        return 1
    fi
    if [[ ! -f "$1" ]]; then
        echo "Error: ${MILKCLI_SIGIL}source: file not found: $1" >&2
        return 1
    fi
    local __milk_src_file="$1"
    shift
    # shellcheck disable=SC1090
    builtin source <("${MILK_INSTALLDIR}/bin/milk-call" translate --legacy "${__milk_src_file}") "$@"
}

export PROCINFO_NCPU="${PROCINFO_NCPU:-$(nproc 2>/dev/null || echo 1)}"
export PROCINFO_NPROC="${PROCINFO_NPROC:-0}"

_milk_waitany_impl() {
    local timeout_sec="0"
    local -a streams=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -t|--timeout)
                timeout_sec="${2:-0}"
                shift 2
                ;;
            --timeout=*)
                timeout_sec="${1#--timeout=}"
                shift
                ;;
            *)
                streams+=("$1")
                shift
                ;;
        esac
    done
    if [[ ${#streams[@]} -eq 0 ]]; then
        echo "Usage: ${MILKCLI_SIGIL}waitany <s1> <s2> [...] [--timeout <sec>]" >&2
        return 1
    fi
    local s
    local -a cnts=()
    for s in "${streams[@]}"; do
        local c
        c="$(milkget "s.${s#@s.}.cnt0" 2>/dev/null)" || c="-1"
        cnts+=("${c}")
    done
    local start_ns
    start_ns="$(date +%s%N)"
    local timeout_ns=0
    if [[ "${timeout_sec}" != "0" ]]; then
        timeout_ns="$(awk -v t="${timeout_sec}" 'BEGIN { printf "%.0f", t * 1e9 }')"
    fi
    while true; do
        local idx
        for (( idx=0; idx<${#streams[@]}; idx++ )); do
            s="${streams[$idx]}"
            local c_now
            c_now="$(milkget "s.${s#@s.}.cnt0" 2>/dev/null)" || c_now="-1"
            if [[ "${c_now}" != "-1" && "${c_now}" != "${cnts[$idx]}" ]]; then
                if declare -p REPLY &>/dev/null && [[ -z "${REPLY+x}" ]]; then
                    REPLY="${s}"
                else
                    echo "${s}"
                fi
                return 0
            fi
        done
        if [[ ${timeout_ns} -gt 0 ]]; then
            local now_ns
            now_ns="$(date +%s%N)"
            if (( now_ns - start_ns >= timeout_ns )); then
                return 124
            fi
        fi
        sleep 0.002
    done
}

_milk_on_impl() {
    if [[ "${1:-}" == "--list" || $# -eq 0 ]]; then
        local pid
        for pid in "${_MILK_ON_PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                echo "${pid}"
            fi
        done
        return 0
    fi
    if [[ "${1:-}" == "--clear" ]]; then
        local pid
        for pid in "${_MILK_ON_PIDS[@]}"; do
            kill "${pid}" 2>/dev/null || true
        done
        _MILK_ON_PIDS=()
        return 0
    fi
    local target="$1"
    shift
    local action="$*"
    (
        while milkwaitfor "${target}" >/dev/null 2>&1; do
            eval "${action}" || true
        done
    ) &
    _MILK_ON_PIDS+=("$!")
    return 0
}

_milk_unload_impl() {
    _milk_on_impl --clear
    milkshutdown 2>/dev/null || true
    local fn
    while IFS= read -r fn; do
        if [[ "${fn}" == "${MILKCLI_SIGIL}"* ]]; then
            unset -f "${fn}" 2>/dev/null || true
        fi
    done < <(declare -F | awk '{print $3}')
    if [[ ${_MILK_BUILTIN_LOADED} -eq 1 ]]; then
        enable -d milkcall milkget milkset milkcalc milktest milkexists \
            milkwaitfor milkgrab milkcomplete milkshutdown 2>/dev/null || true
        _MILK_BUILTIN_LOADED=0
    fi
    if [[ "$(declare -p PROMPT_COMMAND 2>/dev/null)" == "declare -a"* ]]; then
        local -a __new_pc=()
        local __pc_item
        for __pc_item in "${PROMPT_COMMAND[@]}"; do
            [[ "${__pc_item}" != "_milk_apply_ps1" ]] && __new_pc+=("${__pc_item}")
        done
        PROMPT_COMMAND=("${__new_pc[@]}")
    elif [[ -n "${PROMPT_COMMAND:-}" ]]; then
        PROMPT_COMMAND="${PROMPT_COMMAND//; _milk_apply_ps1/}"
        PROMPT_COMMAND="${PROMPT_COMMAND//_milk_apply_ps1; /}"
        PROMPT_COMMAND="${PROMPT_COMMAND//_milk_apply_ps1/}"
    fi
    if [[ -n "${_MILK_ORIG_PS1+x}" ]]; then
        PS1="${_MILK_ORIG_PS1}"
        unset _MILK_ORIG_PS1
    fi
}

# ---------------------------------------------------------------------------
# 6. Define Tier-1 Core Primitives under ${MILKCLI_SIGIL}
# ---------------------------------------------------------------------------
eval "
function ${MILKCLI_SIGIL}            { milkcalc \"\$@\"; }
function ${MILKCLI_SIGIL}calc        { milkcalc \"\$@\"; }
function ${MILKCLI_SIGIL}test        { milktest \"\$@\"; }
function ${MILKCLI_SIGIL}get         { milkget \"\$@\"; }
function ${MILKCLI_SIGIL}set         { milkset \"\$@\"; }
function ${MILKCLI_SIGIL}exists      { milkexists \"\$@\"; }
function ${MILKCLI_SIGIL}running     { milkexists -r \"\$@\"; }
function ${MILKCLI_SIGIL}waitfor     { milkwaitfor \"\$@\"; }
function ${MILKCLI_SIGIL}waitany     { _milk_waitany_impl \"\$@\"; }
function ${MILKCLI_SIGIL}grab        { milkgrab \"\$@\"; }
function ${MILKCLI_SIGIL}defer       { _milk_defer_add \"\$@\"; }
function ${MILKCLI_SIGIL}on          { _milk_on_impl \"\$@\"; }
function ${MILKCLI_SIGIL}source      { _milk_source_impl \"\$@\"; }
function ${MILKCLI_SIGIL}h           { milkcall h \"\$@\"; }
function ${MILKCLI_SIGIL}hc          { milkcall hc \"\$@\"; }
function ${MILKCLI_SIGIL}hm          { milkcall hm \"\$@\"; }
function ${MILKCLI_SIGIL}hf          { milkcall hf \"\$@\"; }
function ${MILKCLI_SIGIL}streamcat   { milkcall streamcat \"\$@\"; }
function ${MILKCLI_SIGIL}streamwrite { milkcall streamwrite \"\$@\"; }
function ${MILKCLI_SIGIL}unload      { _milk_unload_impl \"\$@\"; }
"

# ---------------------------------------------------------------------------
# 7. Pre-register all @<cmd> shims & external symlink farm (libexec/milk-bin)
# ---------------------------------------------------------------------------
__milk_setup_commands() {
    local cmd_list="${MILK_INSTALLDIR}/share/milk/milk-commands.list"
    if [[ ! -f "${cmd_list}" || "${MILK_INSTALLDIR}/bin/milk-cli" -nt "${cmd_list}" ]]; then
        mkdir -p "$(dirname "${cmd_list}")" 2>/dev/null || true
        if ! "${MILK_INSTALLDIR}/bin/milk-cli" --dump-commands > "${cmd_list}.tmp" 2>/dev/null; then
            cmd_list="${TMPDIR:-/tmp}/milk-commands-${UID:-$(id -u)}.list"
            "${MILK_INSTALLDIR}/bin/milk-cli" --dump-commands > "${cmd_list}.tmp" 2>/dev/null || true
        fi
        mv -f "${cmd_list}.tmp" "${cmd_list}" 2>/dev/null || true
    fi

    if [[ -f "${cmd_list}" ]]; then
        local shim_code=""
        local cmd
        while IFS= read -r cmd; do
            [[ -z "${cmd}" ]] && continue
            # Skip names that conflict with core primitives defined above
            case "${cmd}" in
                calc|test|get|set|exists|running|waitfor|waitany|grab|defer|on|source|h|hc|hm|hf|streamcat|streamwrite|unload)
                    continue
                    ;;
            esac
            shim_code+="function ${MILKCLI_SIGIL}${cmd} { milkcall \"${cmd}\" \"\$@\"; };"
        done < "${cmd_list}"
        eval "${shim_code}"
    fi

    # Populate symlink directory for external tools (xargs, find -exec, timeout)
    local bin_farm="${MILK_INSTALLDIR}/libexec/milk-bin"
    if ! mkdir -p "${bin_farm}" 2>/dev/null; then
        bin_farm="${TMPDIR:-/tmp}/milk-bin-${UID:-$(id -u)}"
        mkdir -p "${bin_farm}" 2>/dev/null || true
    fi
    if [[ -d "${bin_farm}" && -w "${bin_farm}" ]]; then
        local marker="${bin_farm}/.sigil_${MILKCLI_SIGIL}"
        if [[ ! -f "${marker}" || "${cmd_list}" -nt "${marker}" ]]; then
            local target_bin="${MILK_INSTALLDIR}/bin/milk-call"
            local core_c
            for core_c in "" calc test get set exists running waitfor grab h hc hm hf streamcat streamwrite unload; do
                ln -sf "${target_bin}" "${bin_farm}/${MILKCLI_SIGIL}${core_c}" 2>/dev/null || true
            done
            if [[ -f "${cmd_list}" ]]; then
                while IFS= read -r cmd; do
                    [[ -z "${cmd}" ]] && continue
                    ln -sf "${target_bin}" "${bin_farm}/${MILKCLI_SIGIL}${cmd}" 2>/dev/null || true
                done < "${cmd_list}"
            fi
            touch "${marker}" 2>/dev/null || true
        fi
        if [[ ":${PATH}:" != *":${bin_farm}:"* ]]; then
            export PATH="${bin_farm}:${PATH}"
        fi
    fi
}
__milk_setup_commands
unset -f __milk_setup_commands

# Dynamic fallback for newly loaded modules or commands via command_not_found_handle
if declare -F command_not_found_handle >/dev/null && ! declare -F __milk_prev_cnf_handle >/dev/null; then
    eval "$(declare -f command_not_found_handle | sed '1s/command_not_found_handle/__milk_prev_cnf_handle/')"
fi

command_not_found_handle() {
    local cmd="$1"
    if [[ "${cmd}" == "${MILKCLI_SIGIL}"* ]]; then
        shift
        local bare="${cmd#"${MILKCLI_SIGIL}"}"
        milkcall "${bare}" "$@"
        return $?
    fi
    if declare -F __milk_prev_cnf_handle >/dev/null; then
        __milk_prev_cnf_handle "$@"
        return $?
    fi
    echo "bash: ${cmd}: command not found" >&2
    return 127
}

# ---------------------------------------------------------------------------
# 8. Programmable Bash Tab-Completion
# ---------------------------------------------------------------------------
_milk_complete() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    if [[ ${COMP_CWORD} -eq 0 ]]; then
        if [[ "${cur}" == "${MILKCLI_SIGIL}"* ]]; then
            mapfile -t COMPREPLY < <(milkcomplete "${cur}" cmd 2>/dev/null)
        fi
        return 0
    fi
    local cmd="${COMP_WORDS[0]}"
    if [[ "${cmd}" == "${MILKCLI_SIGIL}"* ]]; then
        mapfile -t COMPREPLY < <(milkcomplete "${cur}" arg 2>/dev/null)
    fi
}

complete -o default -F _milk_complete "${MILKCLI_SIGIL}" "${MILKCLI_SIGIL}calc" \
    "${MILKCLI_SIGIL}test" "${MILKCLI_SIGIL}get" "${MILKCLI_SIGIL}set" \
    "${MILKCLI_SIGIL}exists" "${MILKCLI_SIGIL}running" "${MILKCLI_SIGIL}waitfor" \
    "${MILKCLI_SIGIL}grab" "${MILKCLI_SIGIL}h" "${MILKCLI_SIGIL}hc" "${MILKCLI_SIGIL}hm" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 9. Visible Prompt Label ([milk]) for milk-capable terminals
# ---------------------------------------------------------------------------
_MILK_PS1_PREFIX='\[\033[1;36m\][milk]\[\033[0m\] '

_milk_apply_ps1() {
    local __milk_last_rc=$?
    if [[ -z "${MILK_NO_PROMPT:-}" ]]; then
        if [[ -z "${_MILK_ORIG_PS1+x}" ]]; then
            _MILK_ORIG_PS1="${PS1:-}"
        fi
        if [[ "${PS1:-}" != *"[milk]"* ]]; then
            PS1="${_MILK_PS1_PREFIX}${PS1:-\\s-\\v\\\$ }"
        fi
    fi
    return ${__milk_last_rc}
}

if [[ $- == *i* || -n "${PS1:-}" ]]; then
    _milk_apply_ps1
    if [[ "$(declare -p PROMPT_COMMAND 2>/dev/null)" == "declare -a"* ]]; then
        if [[ " ${PROMPT_COMMAND[*]} " != *" _milk_apply_ps1 "* ]]; then
            PROMPT_COMMAND+=("_milk_apply_ps1")
        fi
    else
        if [[ "${PROMPT_COMMAND:-}" != *"_milk_apply_ps1"* ]]; then
            PROMPT_COMMAND="${PROMPT_COMMAND:+${PROMPT_COMMAND%;}; }_milk_apply_ps1"
        fi
    fi
fi
