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
            echo "Loads milk commands (@<cmd>, @batch, @pipe), entity discovery & accessors"
            echo "(@get [-v|-a|-A|--json], @set, @exists, @running), zero-fork math/slice"
            echo "evaluation (@calc [-v], @test), stream capture & playback (@waitfor, @waitany,"
            echo "@grab [--fits|--timestamps], @replay, @on), and LIFO cleanup (@defer) into"
            echo "the current GNU Bash session."
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
export _MILK_LOCAL_SOCK="@milk-cli-${UID:-$(id -u)}-$$"
_milk_ov_file="${TMPDIR:-/tmp}/.milk_active_sock_${UID:-$(id -u)}_$$"
if [[ "${1:-}" == "--server" || "${1:-}" == "-s" ]] && [[ -n "${2:-}" ]]; then
    export _MILK_SERVER_NAME="$2"
    export MILKCLI_SOCK="@milk-cli-${UID:-$(id -u)}-$2"
    printf '%s' "${MILKCLI_SOCK}" > "${_milk_ov_file}" 2>/dev/null || true
    shift 2
elif [[ "${1:-}" == --server=* ]]; then
    export _MILK_SERVER_NAME="${1#--server=}"
    export MILKCLI_SOCK="@milk-cli-${UID:-$(id -u)}-${_MILK_SERVER_NAME}"
    printf '%s' "${MILKCLI_SOCK}" > "${_milk_ov_file}" 2>/dev/null || true
    shift
elif [[ -f "${_milk_ov_file}" ]]; then
    export MILKCLI_SOCK="$(<"${_milk_ov_file}")"
    export _MILK_SERVER_NAME="${MILKCLI_SOCK#@milk-cli-*-}"
elif [[ -z "${MILKCLI_SOCK:-}" ]]; then
    export MILKCLI_SOCK="${_MILK_LOCAL_SOCK}"
fi
unset _milk_ov_file

# ---------------------------------------------------------------------------
# 4. Load C builtins (libmilkbash.so) or install milk-call fallback shims
# ---------------------------------------------------------------------------
_MILK_BUILTIN_LOADED=0
enable -d milkcall milkget milkset milkcalc milktest milkexists \
    milkwaitfor milkwaitany milkgrab milkcomplete milkshutdown milkserver milklog 2>/dev/null || true
if [[ -f "${MILK_INSTALLDIR}/lib/libmilkbash.so" ]] && ! grep -q "libmilk.*(deleted)" "/proc/$$/maps" 2>/dev/null; then
    if MILK_QUIET=1 enable -f "${MILK_INSTALLDIR}/lib/libmilkbash.so" \
        milkcall milkget milkset milkcalc milktest milkexists \
        milkwaitfor milkwaitany milkgrab milkcomplete milkshutdown milkserver milklog 2>/dev/null; then
        _MILK_BUILTIN_LOADED=1
    fi
fi

if [[ ${_MILK_BUILTIN_LOADED} -eq 0 ]]; then
    milkcall() {
        if [[ "${1:-}" == "streamcat" ]]; then
            local _aname=""
            local -a _fargs=()
            while [[ $# -gt 0 ]]; do
                if [[ "$1" == "-a" && $# -ge 2 ]]; then
                    _aname="$2"
                    shift 2
                else
                    _fargs+=("$1")
                    shift
                fi
            done
            if [[ -n "${_aname}" ]]; then
                local _out
                _out="$("${MILK_INSTALLDIR}/bin/milk-call" call "${_fargs[@]}")" || return $?
                read -r -a "${_aname}" <<< "${_out//$'\n'/ }"
                return 0
            fi
            set -- "${_fargs[@]}"
        fi
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
        if [[ "${1:-}" == "-a" && -n "${2:-}" ]]; then
            local _aname="$2"
            shift 2
            local _val
            _val="$("${MILK_INSTALLDIR}/bin/milk-call" get "$@")" || return $?
            read -r -a "${_aname}" <<< "${_val//$'\n'/ }"
            return 0
        fi
        if [[ "${1:-}" == "-A" && -n "${2:-}" && -n "${3:-}" ]]; then
            local _Aname="$2"
            local _spec="${3#@}"
            case "${_spec}" in
                *.kw.\*) _spec="${_spec%\*}.kv" ;;
                *.kw)    _spec="${_spec}.kv" ;;
                *.\*)    _spec="${_spec%\*}.kv" ;;
                *.list)  _spec="${_spec%.list}.kv" ;;
                *.*.*)   ;;
                *[\*\?]*) ;;
                *.*)     _spec="${_spec}.kv" ;;
            esac
            local _val
            _val="$("${MILK_INSTALLDIR}/bin/milk-call" get "${_spec}")" || return $?
            declare -gA "${_Aname}=()"
            local _line _k _v
            while IFS= read -r _line; do
                if [[ "${_line}" == *$'\t'* ]]; then
                    _k="${_line%%$'\t'*}"
                    _v="${_line#*$'\t'}"
                elif [[ "${_line}" == *=* ]]; then
                    _k="${_line%%=*}"
                    _v="${_line#*=}"
                else
                    continue
                fi
                [[ -n "${_k}" ]] && printf -v "${_Aname}[${_k}]" '%s' "${_v}"
            done <<< "${_val}"
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
    milkwaitany()  { "${MILK_INSTALLDIR}/bin/milk-call" waitany "$@"; }
    milkgrab()     { "${MILK_INSTALLDIR}/bin/milk-call" grab "$@"; }
    milkcomplete() { "${MILK_INSTALLDIR}/bin/milk-call" complete "$@"; }
    milkshutdown() { "${MILK_INSTALLDIR}/bin/milk-call" shutdown "$@"; }
    milkserver() {
        local eval_file="${TMPDIR:-/tmp}/.milk_eval_$$_${BASHPID}"
        local rc=0
        exec {__milk_efd}>"${eval_file}"
        MILK_CALL_EVAL_FD="${__milk_efd}" "${MILK_INSTALLDIR}/bin/milk-call" server "$@" || rc=$?
        exec {__milk_efd}>&-
        if [[ -s "${eval_file}" ]]; then
            # shellcheck disable=SC1090
            source "${eval_file}"
        fi
        rm -f "${eval_file}"
        return ${rc}
    }
    milklog()      { "${MILK_INSTALLDIR}/bin/milk-call" log "$@"; }
fi

# ---------------------------------------------------------------------------
# 5. Lifecycle helpers: @defer (LIFO cleanup) and @on (background watchers)
# ---------------------------------------------------------------------------
declare -ga _MILK_DEFER_STACK=()
declare -gi _MILK_DEFER_RAN=0
declare -ga _MILK_ON_PIDS=()
declare -ga _MILK_ON_TARGETS=()
declare -ga _MILK_ON_CMDS=()

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

_milk_on_impl() {
    if [[ $# -eq 0 ]]; then
        echo "Usage: ${MILKCLI_SIGIL}on [--once|-1] [--max-hz <hz>] [--cooldown <sec>] [-t|--timeout <sec>] <target[,target2...]> \"<cmd>\" | --list | --stop <id|pid|target> | --clear" >&2
        return 1
    fi
    if [[ "${1:-}" == "--list" ]]; then
        local idx
        for idx in "${!_MILK_ON_PIDS[@]}"; do
            local pid="${_MILK_ON_PIDS[$idx]}"
            if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
                echo "$((idx + 1)) ${pid} ${_MILK_ON_TARGETS[$idx]} ${_MILK_ON_CMDS[$idx]}"
            fi
        done
        return 0
    fi
    if [[ "${1:-}" == "--clear" ]]; then
        local pid
        for pid in "${_MILK_ON_PIDS[@]}"; do
            if [[ -n "${pid}" ]]; then
                pkill -P "${pid}" 2>/dev/null || true
                kill "${pid}" 2>/dev/null || true
            fi
        done
        _MILK_ON_PIDS=()
        _MILK_ON_TARGETS=()
        _MILK_ON_CMDS=()
        return 0
    fi
    if [[ "${1:-}" == "--stop" || "${1:-}" == "-s" ]]; then
        local sel="${2:-}"
        if [[ -z "${sel}" ]]; then
            echo "Usage: ${MILKCLI_SIGIL}on --stop <id|pid|target>" >&2
            return 1
        fi
        local idx stopped=0
        for idx in "${!_MILK_ON_PIDS[@]}"; do
            local pid="${_MILK_ON_PIDS[$idx]}"
            local tgt="${_MILK_ON_TARGETS[$idx]}"
            if [[ -n "${pid}" ]] && ( [[ "$((idx + 1))" == "${sel}" || "${pid}" == "${sel}" || "${tgt}" == "${sel}" || "s.${tgt}" == "${sel}" || "${tgt#s.}" == "${sel}" ]] ); then
                pkill -P "${pid}" 2>/dev/null || true
                kill "${pid}" 2>/dev/null || true
                unset '_MILK_ON_PIDS[idx]' '_MILK_ON_TARGETS[idx]' '_MILK_ON_CMDS[idx]'
                stopped=1
            fi
        done
        return $(( stopped ? 0 : 1 ))
    fi

    local once=0
    local cooldown=""
    local timeout_sec=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --once|-1)
                once=1
                shift
                ;;
            --cooldown)
                cooldown="${2:-}"
                shift 2
                ;;
            --cooldown=*)
                cooldown="${1#*=}"
                shift
                ;;
            --max-hz)
                if [[ -n "${2:-}" ]]; then
                    cooldown="$(awk -v hz="${2}" 'BEGIN { if (hz > 0) printf "%.6f", 1.0 / hz; else print "0" }')"
                fi
                shift 2
                ;;
            --max-hz=*)
                local hz="${1#*=}"
                cooldown="$(awk -v hz="${hz}" 'BEGIN { if (hz > 0) printf "%.6f", 1.0 / hz; else print "0" }')"
                shift
                ;;
            -t|--timeout)
                timeout_sec="${2:-}"
                shift 2
                ;;
            --timeout=*)
                timeout_sec="${1#*=}"
                shift
                ;;
            *)
                break
                ;;
        esac
    done

    if [[ $# -lt 2 ]]; then
        echo "Usage: ${MILKCLI_SIGIL}on [--once|-1] [--max-hz <hz>] [--cooldown <sec>] [-t|--timeout <sec>] <target[,target2...]> \"<cmd>\" | --list | --stop <id|pid|target> | --clear" >&2
        return 1
    fi

    local -a targets=()
    if [[ "$1" == *,* ]]; then
        local old_ifs="${IFS}"
        IFS=',' read -r -a targets <<< "$1"
        IFS="${old_ifs}"
        shift
    else
        targets+=("$1")
        shift
        while [[ $# -ge 2 && ( "$1" == s.* || "$1" == stream.* ) ]]; do
            targets+=("$1")
            shift
        done
    fi

    local action="$*"
    local tgt_label
    tgt_label="$(IFS=','; echo "${targets[*]}")"

    (
        local -a wait_opts=()
        if [[ -n "${timeout_sec}" ]]; then
            wait_opts+=(--timeout "${timeout_sec}")
        fi
        if [[ ${#targets[@]} -gt 1 ]]; then
            while milkwaitany "${wait_opts[@]}" "${targets[@]}" >/dev/null 2>&1; do
                export MILK_TRIGGER
                eval "${action}" || true
                if [[ ${once} -eq 1 ]]; then
                    break
                fi
                if [[ -n "${cooldown}" && "${cooldown}" != "0" ]]; then
                    sleep "${cooldown}"
                fi
            done
        else
            local single_t="${targets[0]}"
            while milkwaitfor "${wait_opts[@]}" "${single_t}" >/dev/null 2>&1; do
                export MILK_TRIGGER="${single_t#s.}"
                MILK_TRIGGER="${MILK_TRIGGER#stream.}"
                eval "${action}" || true
                if [[ ${once} -eq 1 ]]; then
                    break
                fi
                if [[ -n "${cooldown}" && "${cooldown}" != "0" ]]; then
                    sleep "${cooldown}"
                fi
            done
        fi
    ) &
    _MILK_ON_PIDS+=("$!")
    _MILK_ON_TARGETS+=("${tgt_label}")
    _MILK_ON_CMDS+=("${action}")
    return 0
}

_milk_unload_impl() {
    _milk_on_impl --clear
    rm -f "${TMPDIR:-/tmp}/.milk_active_sock_${UID:-$(id -u)}_$$" 2>/dev/null || true
    if [[ -n "${_MILK_LOCAL_SOCK:-}" ]]; then
        MILKCLI_SOCK="${_MILK_LOCAL_SOCK}" milkshutdown 2>/dev/null || true
    else
        milkshutdown 2>/dev/null || true
    fi
    unset _MILK_SERVER_NAME _MILK_LOCAL_SOCK

    # 1. Remove completion bindings and unset all @<cmd> functions registered under ${MILKCLI_SIGIL}
    local fn
    while IFS= read -r fn; do
        if [[ -n "${fn}" ]]; then
            complete -r "${fn}" 2>/dev/null || true
            unset -f "${fn}" 2>/dev/null || true
        fi
    done < <(compgen -A function "${MILKCLI_SIGIL}")

    # 2. Remove milk-bin symlink farm from PATH and flush bash command hash table
    local new_path=""
    local p_entry
    local old_ifs="${IFS}"
    IFS=':'
    for p_entry in ${PATH}; do
        if [[ -n "${_MILK_BIN_FARM:-}" && "${p_entry}" == "${_MILK_BIN_FARM}" ]]; then
            continue
        fi
        if [[ "${p_entry}" == */libexec/milk-bin || "${p_entry}" == */milk-bin-* ]]; then
            continue
        fi
        new_path="${new_path:+${new_path}:}${p_entry}"
    done
    IFS="${old_ifs}"
    export PATH="${new_path}"
    unset _MILK_BIN_FARM
    hash -r 2>/dev/null || true

    # 3. Restore command_not_found_handle
    if declare -F __milk_prev_cnf_handle >/dev/null; then
        eval "$(declare -f __milk_prev_cnf_handle | sed '1s/__milk_prev_cnf_handle/command_not_found_handle/')"
        unset -f __milk_prev_cnf_handle
    else
        unset -f command_not_found_handle 2>/dev/null || true
    fi

    # 4. Disable C builtins and unset fallback shims
    if [[ ${_MILK_BUILTIN_LOADED:-0} -eq 1 ]]; then
        enable -d milkcall milkget milkset milkcalc milktest milkexists \
            milkwaitfor milkwaitany milkgrab milkcomplete milkshutdown milkserver milklog 2>/dev/null || true
        _MILK_BUILTIN_LOADED=0
    fi
    unset -f milkcall milkget milkset milkcalc milktest milkexists \
        milkwaitfor milkwaitany milkgrab milkcomplete milkshutdown milkserver milklog 2>/dev/null || true

    # 5. Restore PROMPT_COMMAND and PS1
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

    if [[ ( $- == *i* && -z "${MILK_QUIET:-}" ) || -n "${MILK_BANNER:-}" ]]; then
        echo -e "\033[1;36m[milk]\033[0m Unloaded milk bash integration."
    fi
}

# ---------------------------------------------------------------------------
# 6. Define Core milk.bash Primitives under ${MILKCLI_SIGIL}
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
function ${MILKCLI_SIGIL}waitany     { milkwaitany \"\$@\"; }
function ${MILKCLI_SIGIL}grab        { milkgrab \"\$@\"; }
function ${MILKCLI_SIGIL}replay      { milkcall replay \"\$@\"; }
function ${MILKCLI_SIGIL}defer       { _milk_defer_add \"\$@\"; }
function ${MILKCLI_SIGIL}on          { _milk_on_impl \"\$@\"; }
function ${MILKCLI_SIGIL}source      { _milk_source_impl \"\$@\"; }
function ${MILKCLI_SIGIL}server      { milkserver \"\$@\"; local __rc=\$?; export MILKCLI_SOCK _MILK_SERVER_NAME; _milk_apply_ps1; return \${__rc}; }
function ${MILKCLI_SIGIL}log         { milklog \"\$@\"; }
function ${MILKCLI_SIGIL}h           { milkcall h \"\$@\"; }
function ${MILKCLI_SIGIL}hb          { milkcall hb \"\$@\"; }
function ${MILKCLI_SIGIL}hc          { if [[ \"\${1:-}\" == \"--sig\" || \"\${1:-}\" == \"-s\" ]]; then milkcomplete \"\${3:-}\" \"sig:\${2:-}\"; else milkcall hc \"\$@\"; fi; }
function ${MILKCLI_SIGIL}hm          { milkcall hm \"\$@\"; }
function ${MILKCLI_SIGIL}hf          { milkcall hf \"\$@\"; }
function ${MILKCLI_SIGIL}streamcat   { milkcall streamcat \"\$@\"; }
function ${MILKCLI_SIGIL}streamwrite { milkcall streamwrite \"\$@\"; }
function ${MILKCLI_SIGIL}batch       { milkcall batch \"\$@\"; }
function ${MILKCLI_SIGIL}pipe        { milkcall pipe \"\$@\"; }
function ${MILKCLI_SIGIL}unload      { _milk_unload_impl \"\$@\"; }
"

# ---------------------------------------------------------------------------
# 7. Pre-register all @<cmd> shims & external symlink farm (libexec/milk-bin)
# ---------------------------------------------------------------------------
_MILK_CMD_COUNT=26
__milk_setup_commands() {
    local cmd_list="${MILK_INSTALLDIR}/share/milk/milk-commands.list"
    if [[ ! -f "${cmd_list}" \
          || "${MILK_INSTALLDIR}/bin/milk-cli" -nt "${cmd_list}" \
          || "${MILK_INSTALLDIR}/lib/libmilkscript.so" -nt "${cmd_list}" ]]; then
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
            # Skip core primitives, shell builtins, REPL-only commands, and internal help topics
            case "${cmd}" in
                calc|test|get|set|exists|running|waitfor|waitany|grab|replay|defer|on|source|server|log|h|hb|hc|hm|hf|streamcat|streamwrite|batch|pipe|unload|\
                cd|pwd|echo|printf|read|shift|export|unset|vars|alias|unalias|aliases|time|watch|exit|quit|exitCLI|\
                history|searchhist|fhist|ghistory|lhistory|savehistory|savescript|setprompt|bookmark|sessionlog|synhl|syndiag|cliindent|showmatch|helprl|fhelp|fparam|\
                help|help-*|list-streams|list-fps|fpsset)
                    continue
                    ;;
            esac
            shim_code+="function ${MILKCLI_SIGIL}${cmd} { milkcall \"${cmd}\" \"\$@\"; };"
            (( _MILK_CMD_COUNT++ ))
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
        _MILK_BIN_FARM="${bin_farm}"
        local marker="${bin_farm}/.sigil_${MILKCLI_SIGIL}"
        if [[ ! -f "${marker}" || "${cmd_list}" -nt "${marker}" || "${BASH_SOURCE[0]}" -nt "${marker}" ]]; then
            local target_bin="${MILK_INSTALLDIR}/bin/milk-call"
            if [[ -x "${target_bin}" ]]; then
                "${target_bin}" setup-bin-farm "${bin_farm}" "${MILKCLI_SIGIL}" "${target_bin}" "${cmd_list}" 2>/dev/null || true
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

# Dynamic fallback for newly loaded modules, commands, or entity shorthands via command_not_found_handle
if declare -F command_not_found_handle >/dev/null && ! declare -F __milk_prev_cnf_handle >/dev/null; then
    eval "$(declare -f command_not_found_handle | sed '1s/command_not_found_handle/__milk_prev_cnf_handle/')"
fi

command_not_found_handle() {
    local cmd="$1"
    if [[ "${cmd}" == "${MILKCLI_SIGIL}"* ]]; then
        shift
        local bare="${cmd#"${MILKCLI_SIGIL}"}"
        case "${bare}" in
            s.*.*|stream.*.*|f.*.*|fps.*.*|p.*.*|proc.*.*|q.*.*|seq.*.*)
                if [[ "${bare}" == *=* || "${1:-}" == "=" ]]; then
                    milkset "${bare}" "$@"
                    return $?
                fi
                case "${bare}" in
                    f.*.confstart|f.*.confstop|f.*.confstep|f.*.runstart|f.*.runstop|f.*.reset|\
                    fps.*.confstart|fps.*.confstop|fps.*.confstep|fps.*.runstart|fps.*.runstop|fps.*.reset)
                        milkset "${bare}" 1
                        return $?
                        ;;
                esac
                if [[ $# -eq 0 ]]; then
                    milkget "${bare}"
                    return $?
                fi
                milkset "${bare}" "$@"
                return $?
                ;;
        esac
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
    if [[ "${cmd}" == "${MILKCLI_SIGIL}server" && ${COMP_CWORD} -eq 1 ]]; then
        mapfile -t COMPREPLY < <(compgen -W "status list ls start use local detach stop attach log" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "${MILKCLI_SIGIL}log" ]]; then
        mapfile -t COMPREPLY < <(compgen -W "-n --last -a --all -g --grep -e --errors --slow --since -f --follow -s --server --json --clear -h --help" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "${MILKCLI_SIGIL}get" && "${cur}" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "--json -j -v -a -A" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "${MILKCLI_SIGIL}on" && "${cur}" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "--once -1 --max-hz --cooldown --timeout -t --list --stop -s --clear" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "${MILKCLI_SIGIL}grab" && "${cur}" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "--fits -f --timestamps --ts --timeout -t" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "${MILKCLI_SIGIL}replay" && "${cur}" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "--fps -r --count -n --loop -l" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "${MILKCLI_SIGIL}"* ]]; then
        mapfile -t COMPREPLY < <(milkcomplete "${cur}" "arg:${cmd}:${COMP_CWORD}" 2>/dev/null)
    fi
}

mapfile -t __milk_all_fns < <(compgen -A function "${MILKCLI_SIGIL}")
if [[ ${#__milk_all_fns[@]} -gt 0 ]]; then
    complete -o default -F _milk_complete "${__milk_all_fns[@]}" 2>/dev/null || true
fi
unset __milk_all_fns

# ---------------------------------------------------------------------------
# 9. Visible Prompt Label ([milk] / [milk:<name>]) & Interactive Startup Status
# ---------------------------------------------------------------------------
_MILK_PS1_PREFIX='\[\033[1;36m\][milk]\[\033[0m\] '

_milk_apply_ps1() {
    local __milk_last_rc=$?
    local __ov="${TMPDIR:-/tmp}/.milk_active_sock_${UID:-$(id -u)}_$$"
    if [[ -f "${__ov}" ]]; then
        export MILKCLI_SOCK="$(<"${__ov}")"
        export _MILK_SERVER_NAME="${MILKCLI_SOCK#@milk-cli-*-}"
    fi
    if [[ -z "${MILK_NO_PROMPT:-}" ]]; then
        if [[ -z "${_MILK_ORIG_PS1+x}" ]]; then
            _MILK_ORIG_PS1="${PS1:-}"
        fi
        local __tag="[milk]"
        if [[ -n "${_MILK_SERVER_NAME:-}" ]]; then
            __tag="[milk:${_MILK_SERVER_NAME}]"
        fi
        _MILK_PS1_PREFIX="\[\033[1;36m\]${__tag}\[\033[0m\] "
        if [[ "${PS1:-}" != *"${__tag}"* ]]; then
            PS1="${_MILK_PS1_PREFIX}${_MILK_ORIG_PS1:-\\s-\\v\\\$ }"
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

if [[ ( $- == *i* && -z "${MILK_QUIET:-}" ) || -n "${MILK_BANNER:-}" ]]; then
    __milk_ver="1.03.00"
    if [[ -f "${MILK_INSTALLDIR}/include/milk_config.h" ]]; then
        __milk_maj="$(awk '/#define MILK_VERSION_MAJOR/ {print $3}' "${MILK_INSTALLDIR}/include/milk_config.h")"
        __milk_min="$(awk '/#define MILK_VERSION_MINOR/ {print $3}' "${MILK_INSTALLDIR}/include/milk_config.h")"
        __milk_pat="$(awk '/#define MILK_VERSION_PATCH/ {print $3}' "${MILK_INSTALLDIR}/include/milk_config.h")"
        if [[ -n "${__milk_maj}" ]]; then
            __milk_ver="${__milk_maj}.${__milk_min}.${__milk_pat}"
        fi
        unset __milk_maj __milk_min __milk_pat
    fi
    echo -e "\033[1;36m[milk]\033[0m \033[1mmilk v${__milk_ver}\033[0m bash integration loaded (${_MILK_CMD_COUNT} ${MILKCLI_SIGIL}-commands)."
    echo -e "\033[1;36m[milk]\033[0m Type \033[1;33m${MILKCLI_SIGIL}h\033[0m for help, \033[1;33m${MILKCLI_SIGIL}hb\033[0m for bash syntax, \033[1;33m${MILKCLI_SIGIL}hm\033[0m for modules, \033[1;33m${MILKCLI_SIGIL}hc <cmd>\033[0m for command help, or \033[1;33m${MILKCLI_SIGIL}unload\033[0m to exit."
    unset __milk_ver
fi
