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
            echo "Loads the milk module commands (module.cmd) and keywords into the current GNU Bash"
            echo "session: entities (@get, @set, @exists, @running, @ls), calculator (@calc, @test,"
            echo "@assert), events (@waitfor, @waitany, @on, @sleep), data (@grab, @replay, @snapshot,"
            echo "@diff, @restore, @pipe), programs (@source, @defer, @module, @config, @server,"
            echo "@unload) and introspection (@help, @log). @help lists them all."
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
# (no subshell: sourcing milk.bash forks nothing)
__milk_resolve_installdir() {
    if [[ -n "${MILK_INSTALLDIR:-}" && -x "${MILK_INSTALLDIR}/bin/milk-cli" ]]; then
        return 0
    fi
    local d="${BASH_SOURCE[0]%/*}" # <prefix>/bin or <prefix>/share/milk
    [[ "${d}" == "${BASH_SOURCE[0]}" ]] && d="."
    [[ "${d}" == /* ]] || d="${PWD}/${d}"
    d="${d%/.}"
    if [[ -x "${d}/milk-cli" ]]; then
        MILK_INSTALLDIR="${d%/*}"
    elif [[ -x "${d}/../../bin/milk-cli" ]]; then
        MILK_INSTALLDIR="${d%/*/*}"
    elif [[ -z "${MILK_INSTALLDIR:-}" ]]; then
        MILK_INSTALLDIR="/usr/local/milk"
    fi
}
__milk_resolve_installdir
export MILK_INSTALLDIR
unset -f __milk_resolve_installdir

if [[ ":${PATH}:" != *":${MILK_INSTALLDIR}/bin:"* ]]; then
    export PATH="${MILK_INSTALLDIR}/bin:${PATH}"
fi
if [[ ":${LD_LIBRARY_PATH:-}:" != *":${MILK_INSTALLDIR}/lib:"* ]]; then
    export LD_LIBRARY_PATH="${MILK_INSTALLDIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

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
# the builtins of libmilkbash.so (also what @unload disables)
declare -ga __MILK_BUILTINS=(milkcall milkget milkset milkcalc milktest milkexists milkls milkwaitfor
    milkwaitany milkgrab milkcomplete milkshutdown milkserver milklog milktranslate milkcloexec milksleep)
# the keywords that are builtins themselves, loaded as kw<name> and named @<name> (no shell function
# per call); "${__MILK_KWBUILTINS[@]/#kw/@}" are their names
declare -ga __MILK_KWBUILTINS=(kwcalc kwtest kwget kwset kwexists kwls kwwaitfor kwwaitany kwsleep kwgrab kwlog)
enable -d "${__MILK_BUILTINS[@]}" "${__MILK_KWBUILTINS[@]/#kw/@}" 2>/dev/null || true
# MILK_NO_BUILTINS=1 forces the milk-call shims (used by the tests to compare both paths)
__milk_stale=0 # a libmilk library replaced on disk while mapped in this shell
while IFS= read -r __milk_l; do
    [[ "${__milk_l}" == *libmilk*"(deleted)" ]] && { __milk_stale=1; break; }
done < "/proc/$$/maps" 2>/dev/null
if [[ "${MILK_NO_BUILTINS:-0}" != 1 && -f "${MILK_INSTALLDIR}/lib/libmilkbash.so" && ${__milk_stale} -eq 0 ]]; then
    if MILK_QUIET=1 enable -f "${MILK_INSTALLDIR}/lib/libmilkbash.so" "${__MILK_BUILTINS[@]}" \
        "${__MILK_KWBUILTINS[@]}" 2>/dev/null && help -s @get >/dev/null 2>&1; then
        _MILK_BUILTIN_LOADED=1
    fi
fi
unset __milk_stale __milk_l

if [[ ${_MILK_BUILTIN_LOADED} -eq 0 ]]; then
    # run milk-call with an eval fd and source the variable assignments it writes back; milk-call binds
    # -v/-a/-A results exactly like the builtins (same C code), so the shims only forward their arguments
    _milk_call_eval() {
        local __milk_ef __milk_rc=0
        __milk_ef="$(mktemp "${TMPDIR:-/tmp}/.milk_eval.XXXXXX")" || return 1
        exec {__milk_efd}>"${__milk_ef}"
        MILK_CALL_EVAL_FD="${__milk_efd}" "${MILK_INSTALLDIR}/bin/milk-call" "$@" || __milk_rc=$?
        exec {__milk_efd}>&-
        if [[ -s "${__milk_ef}" ]]; then
            # shellcheck disable=SC1090
            source "${__milk_ef}"
        fi
        rm -f "${__milk_ef}"
        return ${__milk_rc}
    }
    milkcall()     { _milk_call_eval call "$@"; }
    milkget()      { _milk_call_eval get "$@"; }
    milkset()      { "${MILK_INSTALLDIR}/bin/milk-call" set "$@"; }
    milkcalc()     { _milk_call_eval calc "$@"; }
    milktest()     { "${MILK_INSTALLDIR}/bin/milk-call" test "$@"; }
    milkexists()   { "${MILK_INSTALLDIR}/bin/milk-call" exists "$@"; }
    milkls()       { _milk_call_eval ls "$@"; }
    milkwaitfor()  { "${MILK_INSTALLDIR}/bin/milk-call" waitfor "$@"; }
    milkwaitany()  { "${MILK_INSTALLDIR}/bin/milk-call" waitany "$@"; }
    milkgrab()     { _milk_call_eval grab "$@"; }
    milkcomplete() { "${MILK_INSTALLDIR}/bin/milk-call" complete "$@"; }
    milkshutdown() { "${MILK_INSTALLDIR}/bin/milk-call" shutdown "$@"; }
    milkserver() { _milk_call_eval server "$@"; }
    milklog()      { "${MILK_INSTALLDIR}/bin/milk-call" log "$@"; }
fi

# ---------------------------------------------------------------------------
# 5. Lifecycle helpers: @defer (LIFO cleanup) and @on (background watchers)
# ---------------------------------------------------------------------------
declare -ga _MILK_DEFER_STACK=()
declare -gi _MILK_DEFER_RAN=0
declare -g _MILK_USER_EXIT_TRAP=""
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
    # the user's own EXIT trap was registered before any @defer: run it last
    if [[ -n "${_MILK_USER_EXIT_TRAP}" ]]; then
        eval "${_MILK_USER_EXIT_TRAP}" || true
    fi
    return ${exit_rc}
}

_milk_defer_impl() {
    if [[ $# -eq 0 ]]; then
        return 0
    fi
    if [[ ${#_MILK_DEFER_STACK[@]} -ge 32 ]]; then
        echo "Error: defer stack full (max 32)" >&2
        return 1
    fi
    _MILK_DEFER_RAN=0
    # one argument is a command string (like trap); several are kept as quoted words
    if [[ $# -eq 1 ]]; then
        _MILK_DEFER_STACK+=("$1")
    else
        _MILK_DEFER_STACK+=("$(printf '%q ' "$@")")
    fi
    local cur
    cur="$(trap -p EXIT)"
    if [[ -n "${cur}" && "${cur}" != *_milk_run_defer_stack* ]]; then
        eval "set -- ${cur}"   # trap -- '<cmd>' EXIT
        _MILK_USER_EXIT_TRAP="$3"
    fi
    trap '_milk_run_defer_stack' EXIT
}

# Report a failed inline expansion (status $1, source text $2 if any), empty the variable it was
# bound to ($3, for NAME=@{...}), and pass the status on.
# MILK_STRICT=1 makes it fatal like `set -u`, except at an interactive prompt.
__milk_inline_err() {
    if [[ -n "${2:-}" ]]; then
        printf 'milk: %s: inline expansion failed\n' "$2" >&2
    fi
    if [[ -n "${3:-}" ]]; then
        printf -v "$3" '%s' ''
    fi
    if [[ "${MILK_STRICT:-0}" == 1 && $- != *i* && "${__MILK_INTERACTIVE:-0}" != 1 ]]; then
        exit "$1"
    fi
    return "$1"
}

# Entity namespaces (s, stream, l, ...) and their kind, from the C table (milk_entity_ns.h)
declare -gA __MILK_NS_KIND=()
milkls --namespaces --define 2>/dev/null

declare -gi __MILK_SOURCE_DEPTH=0
declare -gA __MILK_INC_ONCE=()

# Translate a .milk file and source it: in-process with the milktranslate builtin (no fork),
# else through milk-call. Positional arguments are passed on, $LINENO matches the file.
_milk_source_impl() {
    if [[ "${1:-}" == --once ]]; then # skip a file this shell already sourced (by real path)
        shift
        local __rp="${1:-}"
        [[ -f "${__rp}" ]] && __rp="$(realpath "${__rp}" 2>/dev/null || echo "${__rp}")"
        [[ -n "${__MILK_INC_ONCE[$__rp]:-}" ]] && return 0
        __MILK_INC_ONCE[$__rp]=1
    fi
    if [[ $# -eq 0 ]]; then
        echo "Usage: @source [--once] <filename> [args...]" >&2
        return 1
    fi
    if [[ ! -f "$1" ]]; then
        echo "Error: source: file not found: $1" >&2
        return 1
    fi
    if (( __MILK_SOURCE_DEPTH >= 32 )); then
        echo "Error: maximum source recursion depth (32) exceeded" >&2
        return 1
    fi
    local __milk_src_file="$1" __milk_p __milk_rc
    shift
    : $(( __MILK_SOURCE_DEPTH += 1 ))
    if [[ ${_MILK_BUILTIN_LOADED} -eq 1 ]]; then
        milktranslate -v __milk_p "${__milk_src_file}" || { : $(( __MILK_SOURCE_DEPTH -= 1 )); return 1; }
        # shellcheck disable=SC1090
        builtin source "${__milk_p}" "$@"
        __milk_rc=$?
        milktranslate --done "${__milk_p}"
    else
        # shellcheck disable=SC1090
        builtin source <("${MILK_INSTALLDIR}/bin/milk-call" translate --legacy "${__milk_src_file}") "$@"
        __milk_rc=$?
    fi
    : $(( __MILK_SOURCE_DEPTH -= 1 ))
    return ${__milk_rc}
}

if [[ -z "${PROCINFO_NCPU:-}" ]]; then # online CPUs ("0-11,16"), without running nproc
    PROCINFO_NCPU=0
    if IFS= read -r __milk_on < /sys/devices/system/cpu/online 2>/dev/null; then
        for __milk_r in ${__milk_on//,/ }; do
            (( PROCINFO_NCPU += (${__milk_r#*-} - ${__milk_r%-*}) + 1 ))
        done
    fi
    (( PROCINFO_NCPU > 0 )) || PROCINFO_NCPU=1
    unset __milk_on __milk_r
fi
export PROCINFO_NCPU
export PROCINFO_NPROC="${PROCINFO_NPROC:-0}"

_milk_on_impl() {
    if [[ $# -eq 0 ]]; then
        echo "Usage: @on [--once|-1] [--max-hz <hz>] [--cooldown <sec>] [-t|--timeout <sec>] <target[,target2...]> \"<cmd>\" | --list | --stop <id|pid|target> | --clear" >&2
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
            echo "Usage: @on --stop <id|pid|target>" >&2
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
        echo "Usage: @on [--once|-1] [--max-hz <hz>] [--cooldown <sec>] [-t|--timeout <sec>] <target[,target2...]> \"<cmd>\" | --list | --stop <id|pid|target> | --clear" >&2
        return 1
    fi

    local -a targets=()
    local __milk_a
    for __milk_a in "$@"; do
        [[ "${__milk_a}" == "--" ]] && break
    done
    if [[ "${__milk_a:-}" == "--" ]]; then # <t1> [t2...] -- <cmd words...>
        while [[ "$1" != "--" ]]; do
            targets+=("$1")
            shift
        done
        shift
        local action
        action="$(printf '%q ' "$@")"
    elif [[ "$1" == *,* ]]; then
        local old_ifs="${IFS}"
        IFS=',' read -r -a targets <<< "$1"
        IFS="${old_ifs}"
        shift
    else
        targets+=("$1")
        shift
        while [[ $# -ge 2 && "${__MILK_NS_KIND[${1%%.*}]:-}" == stream && "$1" == *.* ]]; do
            targets+=("$1")
            shift
        done
    fi

    if [[ "${__milk_a:-}" != "--" ]]; then
        local action="$*"
    fi
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

_milk_config_impl() {
    if [[ "${1:-}" == prompt && "${_MILK_ENGINE:-}" != cli ]]; then
        shift
        _milk_prompt_impl "$@"
        return $?
    fi
    if [[ "${1:-}" == inline ]]; then
        shift
        _milk_inline_impl "$@"
        return $?
    fi
    milkcall config "$@"
}

# @config inline on: Enter first translates the @{...} of the line typed at this interactive prompt
# (milk-cli and @source always translate). Lines still incomplete (open quote, `do` ...) run as typed.
_milk_inline_rl() {
    [[ "${READLINE_LINE}" == *@\{* ]] || return 0
    local __milk_t
    if [[ ${_MILK_BUILTIN_LOADED} -eq 1 ]]; then
        milktranslate -s __milk_t "${READLINE_LINE}" || return 0
    else
        __milk_t="$(printf '%s\n' "${READLINE_LINE}" | "${MILK_INSTALLDIR}/bin/milk-call" translate -)" || return 0
    fi
    READLINE_LINE="${__milk_t}"
    READLINE_POINT=${#READLINE_LINE}
}

_milk_inline_impl() {
    case "${1:-}" in
    "")
        echo "inline ${_MILK_INLINE:-off}"
        ;;
    on | off)
        if [[ "${_MILK_ENGINE:-}" == cli ]]; then
            [[ "$1" == on ]] && return 0
            echo "milk: @config inline: milk-cli always expands @{...}" >&2
            return 2
        fi
        if [[ $- != *i* ]]; then
            echo "milk: @config inline: for an interactive shell (scripts: @source)" >&2
            return 2
        fi
        if [[ "$1" == on ]]; then # Enter runs the hook (bound to a key sequence no terminal sends), then accept-line
            bind -x '"\e[9999~": _milk_inline_rl'
            bind '"\C-m": "\e[9999~\C-j"'
        else
            bind '"\C-m": accept-line'
            bind -r '\e[9999~'
        fi
        _MILK_INLINE="$1"
        ;;
    *)
        echo "Usage: @config inline [on|off]" >&2
        return 2
        ;;
    esac
}

_milk_unload_impl() {
    _milk_on_impl --clear
    [[ "${_MILK_INLINE:-off}" == on ]] && _milk_inline_impl off
    unset _MILK_INLINE
    rm -f "${TMPDIR:-/tmp}/.milk_active_sock_${UID:-$(id -u)}_$$" 2>/dev/null || true
    if [[ -n "${_MILK_LOCAL_SOCK:-}" ]]; then
        MILKCLI_SOCK="${_MILK_LOCAL_SOCK}" milkshutdown 2>/dev/null || true
    else
        milkshutdown 2>/dev/null || true
    fi
    unset _MILK_SERVER_NAME _MILK_LOCAL_SOCK

    # 1. Remove the completion bindings, the module-command functions and every @ function
    local fn
    for fn in "${__MILK_CATALOG[@]}" $(compgen -A function @) "${__MILK_KWBUILTINS[@]/#kw/@}"; do
        complete -r "${fn}" 2>/dev/null || true
        unset -f "${fn}" 2>/dev/null || true
    done
    unset __MILK_KWSET __MILK_CATALOG __MILK_SHORT

    # 2. Remove the libexec/milk/bin links from PATH and flush bash command hash table
    local new_path=""
    local p_entry
    local old_ifs="${IFS}"
    IFS=':'
    for p_entry in ${PATH}; do
        if [[ "${p_entry}" == */libexec/milk/bin || "${p_entry}" == */libexec/milk-bin ||
              "${p_entry}" == */milk-bin-* ]]; then
            continue
        fi
        new_path="${new_path:+${new_path}:}${p_entry}"
    done
    IFS="${old_ifs}"
    export PATH="${new_path}"
    hash -r 2>/dev/null || true

    # 3. Restore command_not_found_handle
    unset __MILK_CNF_INSTALLED
    if declare -F __milk_prev_cnf_handle >/dev/null; then
        eval "$(declare -f __milk_prev_cnf_handle | sed '1s/__milk_prev_cnf_handle/command_not_found_handle/')"
        unset -f __milk_prev_cnf_handle
    else
        unset -f command_not_found_handle 2>/dev/null || true
    fi

    # 4. Disable C builtins and unset fallback shims
    if [[ ${_MILK_BUILTIN_LOADED:-0} -eq 1 ]]; then
        enable -d "${__MILK_BUILTINS[@]}" "${__MILK_KWBUILTINS[@]/#kw/@}" 2>/dev/null || true
        _MILK_BUILTIN_LOADED=0
    fi
    unset -f "${__MILK_BUILTINS[@]}" 2>/dev/null || true

    # 5. Restore PROMPT_COMMAND and PS1
    unset _MILK_PROMPT_WATCH _MILK_PROMPT_PRESET _MILK_PROMPT_PLAIN 2>/dev/null || true
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
# 6. Language keywords (@calc, @get, ...): one function each, generated from the keyword table
#    (milk_lexicon.c) for this front-end; shell keywords are implemented by _milk_<name>_impl.
#    With libmilkbash loaded, the builtin keywords are builtins named @<name> instead (section 4).
# ---------------------------------------------------------------------------
_milk_running_impl() { milkexists -r "$@"; }
_milk_server_impl() {
    milkserver "$@"
    local __rc=$?
    export MILKCLI_SOCK _MILK_SERVER_NAME
    _milk_apply_ps1
    return ${__rc}
}
declare -gA __MILK_KWSET=()
milkls --keywords --define # defines the @<keyword> functions and fills __MILK_KWSET
if [[ ${_MILK_BUILTIN_LOADED} -eq 1 ]]; then # these keywords are builtins: a function would shadow them
    unset -f "${__MILK_KWBUILTINS[@]/#kw/@}"
fi

# ---------------------------------------------------------------------------
# 7. Module commands: one function each from the install-time catalog (milk-cli --catalog), and the
#    libexec/milk/bin links that let external tools (xargs, find -exec, timeout) run them
# ---------------------------------------------------------------------------
declare -ga __MILK_CATALOG=()
declare -gA __MILK_SHORT=()
if [[ -f "${MILK_INSTALLDIR}/share/milk/catalog.bash" ]]; then
    # shellcheck disable=SC1091
    builtin source "${MILK_INSTALLDIR}/share/milk/catalog.bash"
fi
_MILK_CMD_COUNT=$(( ${#__MILK_KWSET[@]} + ${#__MILK_CATALOG[@]} ))
if [[ -d "${MILK_INSTALLDIR}/libexec/milk/bin" && ":${PATH}:" != *":${MILK_INSTALLDIR}/libexec/milk/bin:"* ]]; then
    export PATH="${PATH}:${MILK_INSTALLDIR}/libexec/milk/bin"
fi

# command_not_found_handle: hints for removed spellings and the entity shorthands; it never asks the
# server (the user's own handler is kept once; sourcing milk.bash again must not save milk's handler)
if [[ -z "${__MILK_CNF_INSTALLED:-}" ]] && declare -F command_not_found_handle >/dev/null &&
    ! declare -F __milk_prev_cnf_handle >/dev/null; then
    eval "$(declare -f command_not_found_handle | sed '1s/command_not_found_handle/__milk_prev_cnf_handle/')"
fi
__MILK_CNF_INSTALLED=1

command_not_found_handle() {
    local cmd="$1" __milk_new
    if [[ -n "${__MILK_IN_CNF:-}" ]] ||
        { [[ "${_MILK_BUILTIN_LOADED:-0}" != 1 ]] && ! declare -F milkls >/dev/null; }; then
        echo "bash: ${cmd}: command not found" >&2 # milk is unloaded, or a milk word is itself missing
        return 127
    fi
    local __MILK_IN_CNF=1
    if __milk_new="$(milkls --rename "${cmd}" 2>/dev/null)"; then
        echo "milk: ${cmd} was removed; use: ${__milk_new}" >&2
        return 127
    fi
    if [[ "${cmd}" == @* ]]; then
        local bare="${cmd#@}"
        if [[ -n "${__MILK_NS_KIND[${bare%%.*}]:-}" && "${bare#*.}" == ?*.?* ]]; then # @s.x.y statements
            if [[ "${bare}" == *=* || "${2:-}" == "=" ]]; then
                echo "milk: ${cmd} was removed; use: @set ${bare%%=*}=<value>" >&2
            else
                echo "milk: ${cmd} was removed; use: @get ${bare} (or @{${bare}} in a word)" >&2
            fi
            return 127
        fi
        if [[ -n "${__MILK_SHORT[${bare}]:-}" ]]; then # @short: module commands are module.command
            echo "milk: ${cmd} was removed; use: ${__MILK_SHORT[${bare}]// / or }" >&2
            return 127
        fi
        if [[ " ${__MILK_CATALOG[*]} " == *" ${bare} "* ]]; then
            echo "milk: ${cmd} was removed; use: ${bare}" >&2
            return 127
        fi
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
        mapfile -t COMPREPLY < <(milkcomplete "${cur}" cmd 2>/dev/null)
        return 0
    fi
    local cmd="${COMP_WORDS[0]}"
    if [[ "${cmd}" == "@server" && ${COMP_CWORD} -eq 1 ]]; then
        mapfile -t COMPREPLY < <(compgen -W "status list ls start use local detach stop attach log" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "@config" && ${COMP_CWORD} -eq 1 ]]; then
        mapfile -t COMPREPLY < <(compgen -W "digits precision procinfo prompt inline" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "@config" && "${COMP_WORDS[1]}" == "prompt" && ${COMP_CWORD} -eq 2 ]]; then
        mapfile -t COMPREPLY < <(compgen -W "status watch unwatch clear preset on off render" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "@config" && "${COMP_WORDS[1]}" == "prompt" && ${COMP_CWORD} -eq 3 &&
          "${COMP_WORDS[2]}" == "preset" ]]; then
        mapfile -t COMPREPLY < <(compgen -W "minimal std ao" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "@config" && ${COMP_CWORD} -eq 2 ]]; then
        mapfile -t COMPREPLY < <(compgen -W "single double on off" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "@module" && ${COMP_CWORD} -eq 1 ]]; then
        mapfile -t COMPREPLY < <(compgen -W "list load" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "@log" ]]; then
        mapfile -t COMPREPLY < <(compgen -W "-n --last -a --all -g --grep -e --errors --slow --since -f --follow -s --server --json --clear -h --help" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "@ls" && "${cur}" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "-l --long -q --quiet --json --local -h --help" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "@get" && "${cur}" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "--json -j -v -a -A" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "@on" && "${cur}" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "--once -1 --max-hz --cooldown --timeout -t --list --stop -s --clear" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "@grab" && "${cur}" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "--fits -f --timestamps --ts --timeout -t --stdout --raw -a" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "@replay" && "${cur}" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "--fps -r --count -n --loop -l --stdin --raw --dtype" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "@snapshot" && "${cur}" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "-o --output" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "@restore" && "${cur}" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "--dry-run -n" -- "${cur}")
        return 0
    fi
    if [[ "${cmd}" == "@assert" && "${cur}" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "--exists -e --running -r --fps-min --max-age --cnt-min --dim -m --msg" -- "${cur}")
        return 0
    fi
    mapfile -t COMPREPLY < <(milkcomplete "${cur}" "arg:${cmd}:${COMP_CWORD}" 2>/dev/null)
}

__milk_all_fns=("${__MILK_CATALOG[@]}")
for __milk_k in "${!__MILK_KWSET[@]}"; do
    __milk_all_fns+=("@${__milk_k}")
done
complete -o default -F _milk_complete "${__milk_all_fns[@]}" 2>/dev/null || true
unset __milk_all_fns __milk_k

# ---------------------------------------------------------------------------
# 9. Dynamic Telemetry Prompt (@config prompt) & Interactive Startup Status
# ---------------------------------------------------------------------------
declare -ga _MILK_PROMPT_WATCH=()
declare -g  _MILK_PROMPT_PRESET="${_MILK_PROMPT_PRESET:-std}"
declare -g  _MILK_PROMPT_PLAIN="[milk]"
_MILK_PS1_PREFIX='\[\033[1;36m\][milk]\[\033[0m\] '

_milk_build_prompt_strings() {
    local head="milk"
    if [[ -n "${_MILK_SERVER_NAME:-}" ]]; then
        head="milk:${_MILK_SERVER_NAME}"
    fi

    local plain="[${head}"
    local ps1_items=""

    if [[ "${_MILK_PROMPT_PRESET:-std}" == "ao" && -n "${MILK_SHM_DIR:-}" && "${MILK_SHM_DIR}" != "/dev/shm" ]]; then
        local shmb="${MILK_SHM_DIR%/}"
        shmb="${shmb##*/}"
        plain+=" | shm:${shmb}"
        ps1_items+=" \[\033[0;90m\]|\[\033[0m\] \[\033[0;35m\]shm:${shmb}\[\033[0m\]"
    fi

    if [[ "${_MILK_PROMPT_PRESET:-std}" != "minimal" && ${#_MILK_PROMPT_WATCH[@]} -gt 0 ]]; then
        local entry lbl spec has_lbl val tmp clr
        for entry in "${_MILK_PROMPT_WATCH[@]}"; do
            [[ -z "${entry}" ]] && continue
            lbl=""
            spec="${entry}"
            has_lbl=0
            if [[ "${entry}" == *=* ]]; then
                lbl="${entry%%=*}"
                spec="${entry#*=}"
                has_lbl=1
            fi
            spec="${spec#@}"
            val=""
            if ! milkget -v val "${spec}" 2>/dev/null; then
                val="?"
            fi

            if [[ "${__MILK_NS_KIND[${spec%%.*}]:-}" == fps && ( "${spec##*.}" == run || "${spec##*.}" == status ) ]]; then
                if [[ ${has_lbl} -eq 0 ]]; then
                    tmp="${spec#*.}"
                    lbl="${tmp%%.*}"
                fi
                case "${val}" in
                    ON|RUN|RUNNING) clr="1;32" ;;
                    CONF)           clr="1;33" ;;
                    *)              clr="0;33" ;;
                esac
                plain+=" | ${lbl}:${val}"
                ps1_items+=" \[\033[0;90m\]|\[\033[0m\] \[\033[${clr}m\]${lbl}:${val}\[\033[0m\]"
            elif [[ "${spec}" == *.fps || "${spec}" == *.fps_hz ]]; then
                if [[ ${has_lbl} -eq 0 ]]; then
                    tmp="${spec#*.}"
                    lbl="${tmp%%.*}"
                fi
                if [[ "${val}" =~ ^[0-9]+\.[0-9]+$ ]]; then
                    printf -v val '%.0f' "${val}" 2>/dev/null || true
                fi
                plain+=" | ${lbl}:${val}Hz"
                ps1_items+=" \[\033[0;90m\]|\[\033[0m\] \[\033[0;36m\]${lbl}:${val}Hz\[\033[0m\]"
            else
                if [[ ${has_lbl} -eq 0 ]]; then
                    lbl="${spec##*.}"
                fi
                plain+=" | ${lbl}=${val}"
                ps1_items+=" \[\033[0;90m\]|\[\033[0m\] \[\033[0;37m\]${lbl}=${val}\[\033[0m\]"
            fi
        done
    fi

    plain+="]"
    _MILK_PROMPT_PLAIN="${plain}"
    if [[ -z "${ps1_items}" ]]; then
        _MILK_PS1_PREFIX="\[\033[1;36m\][${head}]\[\033[0m\] "
    else
        _MILK_PS1_PREFIX="\[\033[1;36m\][${head}\[\033[0m\]${ps1_items}\[\033[1;36m\]]\[\033[0m\] "
    fi
}

_milk_apply_ps1() {
    local __milk_last_rc=$?
    local __ov="${TMPDIR:-/tmp}/.milk_active_sock_${UID:-$(id -u)}_$$"
    if [[ -f "${__ov}" ]]; then
        export MILKCLI_SOCK="$(<"${__ov}")"
        export _MILK_SERVER_NAME="${MILKCLI_SOCK#@milk-cli-*-}"
    fi
    if [[ -n "${MILK_NO_PROMPT:-}" ]]; then
        if [[ -n "${_MILK_ORIG_PS1+x}" ]]; then
            PS1="${_MILK_ORIG_PS1}"
        fi
        return ${__milk_last_rc}
    fi
    if [[ -z "${_MILK_ORIG_PS1+x}" ]] || [[ -n "${PS1:-}" && "${PS1}" != *"[milk"* ]]; then
        _MILK_ORIG_PS1="${PS1:-}"
    fi
    _milk_build_prompt_strings
    PS1="${_MILK_PS1_PREFIX}${_MILK_ORIG_PS1:-\\s-\\v\\\$ }"
    return ${__milk_last_rc}
}

_milk_prompt_impl() {
    local sub="${1:-status}"
    case "${sub}" in
        status|--status|-s)
            _milk_build_prompt_strings
            local st="ENABLED"
            [[ -n "${MILK_NO_PROMPT:-}" ]] && st="DISABLED"
            echo "Prompt:   ${st}"
            echo "Preset:   ${_MILK_PROMPT_PRESET:-std}"
            echo "Server:   ${_MILK_SERVER_NAME:-local (private)}"
            if [[ ${#_MILK_PROMPT_WATCH[@]} -eq 0 ]]; then
                echo "Watched:  (none)"
            else
                echo "Watched:  ${_MILK_PROMPT_WATCH[*]}"
            fi
            echo "Preview:  ${_MILK_PROMPT_PLAIN}"
            return 0
            ;;
        render|--render)
            _milk_build_prompt_strings
            echo "${_MILK_PROMPT_PLAIN}"
            return 0
            ;;
        watch|add|--watch|-w)
            shift
            if [[ $# -eq 0 ]]; then
                echo "Usage: @config prompt watch [label=]<entity.attr> [...]" >&2
                return 1
            fi
            local item existing found
            for item in "$@"; do
                found=0
                for existing in "${_MILK_PROMPT_WATCH[@]}"; do
                    if [[ "${existing}" == "${item}" ]]; then
                        found=1
                        break
                    fi
                done
                [[ ${found} -eq 0 ]] && _MILK_PROMPT_WATCH+=("${item}")
            done
            _milk_apply_ps1
            return 0
            ;;
        unwatch|rm|remove|--unwatch|-u)
            shift
            if [[ $# -eq 0 ]]; then
                echo "Usage: @config prompt unwatch <label|entity.attr> [...]" >&2
                return 1
            fi
            local target existing elbl espec
            local -a kept=()
            for existing in "${_MILK_PROMPT_WATCH[@]}"; do
                elbl=""
                espec="${existing}"
                if [[ "${existing}" == *=* ]]; then
                    elbl="${existing%%=*}"
                    espec="${existing#*=}"
                else
                    espec="${existing#@}"
                    elbl="${espec##*.}"
                fi
                local drop=0
                for target in "$@"; do
                    if [[ "${existing}" == "${target}" || "${elbl}" == "${target}" || "${espec}" == "${target#@}" ]]; then
                        drop=1
                        break
                    fi
                done
                [[ ${drop} -eq 0 ]] && kept+=("${existing}")
            done
            _MILK_PROMPT_WATCH=("${kept[@]}")
            _milk_apply_ps1
            return 0
            ;;
        clear|--clear|-c)
            _MILK_PROMPT_WATCH=()
            _milk_apply_ps1
            return 0
            ;;
        preset|--preset)
            local p="${2:-}"
            if [[ "${p}" != "minimal" && "${p}" != "std" && "${p}" != "ao" ]]; then
                echo "Usage: @config prompt preset <minimal|std|ao>" >&2
                return 1
            fi
            _MILK_PROMPT_PRESET="${p}"
            _milk_apply_ps1
            return 0
            ;;
        on|enable|--on)
            unset MILK_NO_PROMPT
            _milk_apply_ps1
            return 0
            ;;
        off|disable|--off)
            export MILK_NO_PROMPT=1
            _milk_apply_ps1
            return 0
            ;;
        *)
            echo "Usage: @config prompt [status | watch [lbl=]<entity.attr>... | unwatch <target>... | clear | preset <minimal|std|ao> | on | off | render]" >&2
            return 1
            ;;
    esac
}

if [[ $- == *i* || -n "${PS1:-}" ]]; then
    _milk_apply_ps1
    if [[ "${PROMPT_COMMAND@a}" == *a* ]]; then
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
        while read -r _ __milk_k __milk_v; do
            case "${__milk_k}" in
                MILK_VERSION_MAJOR) __milk_maj="${__milk_v}" ;;
                MILK_VERSION_MINOR) __milk_min="${__milk_v}" ;;
                MILK_VERSION_PATCH) __milk_pat="${__milk_v}" ;;
            esac
        done < "${MILK_INSTALLDIR}/include/milk_config.h"
        unset __milk_k __milk_v
        if [[ -n "${__milk_maj}" ]]; then
            __milk_ver="${__milk_maj}.${__milk_min}.${__milk_pat}"
        fi
        unset __milk_maj __milk_min __milk_pat
    fi
    echo -e "\033[1;36m[milk]\033[0m \033[1mmilk v${__milk_ver}\033[0m bash integration loaded (${_MILK_CMD_COUNT} commands)."
    echo -e "\033[1;36m[milk]\033[0m Type \033[1;33m@help\033[0m for help, \033[1;33m<cmd> -h\033[0m for a command, or \033[1;33m@unload\033[0m to exit."
    unset __milk_ver
fi
