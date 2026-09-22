#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# milk-env.sh: Switch shell environment between milk+cacao 'dev' and 'framework-dev'
#
# Usage (must be sourced):
#   source milk-env.sh fdev        # Switch to framework-dev environment
#   source milk-env.sh dev         # Switch to dev environment
#   source milk-env.sh status      # Show current environment settings
#

# Ensure the script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo -e "\033[1;31mERROR: This script must be sourced into your shell, not executed.\033[0m" >&2
    echo -e "Usage:" >&2
    echo -e "  source ${BASH_SOURCE[0]} fdev     # Switch to framework-dev" >&2
    echo -e "  source ${BASH_SOURCE[0]} dev      # Switch to dev" >&2
    echo -e "  source ${BASH_SOURCE[0]} status   # Display current status" >&2
    exit 1
fi

# Helper function to remove a path from colon-delimited list
__milk_path_remove() {
    local var_name="$1"
    local target="$2"
    [ -z "$target" ] && return 0
    local val=":${!var_name}:"
    while [[ "$val" == *":$target:"* ]]; do
        val="${val//:$target:/:}"
    done
    val="${val#:}"
    val="${val%:}"
    eval "$var_name=\"$val\""
}

# Helper function to prepend a path to colon-delimited list
__milk_path_prepend() {
    local var_name="$1"
    local target="$2"
    [ -z "$target" ] && return 0
    __milk_path_remove "$var_name" "$target"
    local current_val="${!var_name}"
    if [ -z "$current_val" ]; then
        eval "$var_name=\"$target\""
    else
        eval "$var_name=\"$target:\$current_val\""
    fi
}

# Locate repository roots
__milk_find_fdev_root() {
    if [ -n "$MILK_FDEV_ROOT" ]; then
        echo "$MILK_FDEV_ROOT"
    elif [ -d "$HOME/src-frameworkdev/milk" ]; then
        echo "$HOME/src-frameworkdev/milk"
    elif [ -d "$HOME/src-milk-frameworkdev/milk" ]; then
        echo "$HOME/src-milk-frameworkdev/milk"
    else
        echo "$HOME/src/milk"
    fi
}

__milk_find_dev_root() {
    if [ -n "$MILK_DEV_ROOT" ]; then
        echo "$MILK_DEV_ROOT"
    elif [ -d "$HOME/src/milk" ]; then
        echo "$HOME/src/milk"
    elif [ -d "$HOME/src/milk-00" ]; then
        echo "$HOME/src/milk-00"
    else
        echo "$HOME/src/milk"
    fi
}

# Locate install directories
__milk_find_fdev_installdir() {
    local fdev_root="$1"
    if [ -n "$MILK_FDEV_INSTALLDIR" ]; then
        echo "$MILK_FDEV_INSTALLDIR"
    elif [ -d "$fdev_root/_install" ]; then
        local ver_dir
        ver_dir="$(ls -d "$fdev_root/_install/milk-"* 2>/dev/null | head -n 1)"
        if [ -n "$ver_dir" ] && [ -d "$ver_dir/bin" ]; then
            echo "$ver_dir"
        else
            echo "$fdev_root/_install"
        fi
    elif [ -d "$fdev_root/_build/_install" ]; then
        echo "$fdev_root/_build/_install"
    else
        echo "$fdev_root/_install"
    fi
}

__milk_find_dev_installdir() {
    if [ -n "$MILK_DEV_INSTALLDIR" ]; then
        echo "$MILK_DEV_INSTALLDIR"
    elif [ -d "/usr/local/milk" ]; then
        echo "/usr/local/milk"
    elif [ -d "/usr/local/milk-1.03.00" ]; then
        echo "/usr/local/milk-1.03.00"
    else
        echo "/usr/local/milk"
    fi
}

# Resolve candidate locations
_FDEV_ROOT="$(__milk_find_fdev_root)"
_FDEV_INSTALLDIR="$(__milk_find_fdev_installdir "$_FDEV_ROOT")"
_FDEV_SHMDIR="${MILK_FDEV_SHMDIR:-/milk/shm-fdev}"

_DEV_ROOT="$(__milk_find_dev_root)"
_DEV_INSTALLDIR="$(__milk_find_dev_installdir)"
_DEV_SHMDIR="${MILK_DEV_SHMDIR:-/milk/shm}"

_FDEV_PROMPT_TAG='\[\033[1;31m\][MILK:fdev]\[\033[0m\] '

__milk_env_status() {
    echo -e "\033[1;34m=== milk + cacao Environment Status ===\033[0m"
    if [ "$MILK_BRANCH" == "framework-dev" ]; then
        echo -e "  Active Branch:   \033[1;31m${MILK_BRANCH}\033[0m"
    elif [ "$MILK_BRANCH" == "dev" ]; then
        echo -e "  Active Branch:   \033[1;32m${MILK_BRANCH}\033[0m"
    else
        echo -e "  Active Branch:   \033[1;33m${MILK_BRANCH:-unset (default dev)}\033[0m"
    fi
    echo -e "  MILK_ROOT:       ${MILK_ROOT:-unset}"
    echo -e "  MILK_INSTALLDIR: ${MILK_INSTALLDIR:-unset}"
    echo -e "  MILK_SHM_DIR:    ${MILK_SHM_DIR:-/milk/shm (default)}"
    echo -e "  MILK_PROC_DIR:   ${MILK_PROC_DIR:-unset}"
    echo -e "  TMUX_TMPDIR:     ${TMUX_TMPDIR:-unset (system default)}"

    if [ "$MILK_BRANCH" == "framework-dev" ]; then
        local milk_cli_path
        milk_cli_path="$(type -p milk-cli 2>/dev/null)"
        if [ -n "$milk_cli_path" ]; then
            echo -e "  Active milk-cli: \033[0;32m${milk_cli_path}\033[0m"
        else
            echo -e "  Active milk-cli: \033[0;31mnot found in PATH\033[0m"
        fi
    else
        local milk_path
        milk_path="$(type -p milk 2>/dev/null)"
        if [ -n "$milk_path" ]; then
            echo -e "  Active milk:     \033[0;32m${milk_path}\033[0m"
        else
            echo -e "  Active milk:     \033[0;31mnot found in PATH\033[0m"
        fi
        if [ -f "$_DEV_INSTALLDIR/bin/milk-cli" ]; then
            echo -e "  \033[1;33m[WARNING] Stale 'milk-cli' found in $_DEV_INSTALLDIR/bin (run: sudo rm -f $_DEV_INSTALLDIR/bin/milk-cli*)\033[0m"
        fi
    fi

    # Check SHM directory status
    if [ -d "${MILK_SHM_DIR:-/milk/shm}" ]; then
        if [ -w "${MILK_SHM_DIR:-/milk/shm}" ]; then
            echo -e "  SHM Status:      \033[0;32mexists, writable\033[0m"
        else
            echo -e "  SHM Status:      \033[0;31mexists, NOT writable\033[0m"
        fi
    else
        echo -e "  SHM Status:      \033[1;33mdoes not exist\033[0m"
    fi
}

__milk_env_switch_fdev() {
    echo -e "\033[1;33mSwitching environment to \033[1;31mframework-dev\033[1;33m...\033[0m"

    # 1. Clean out dev paths
    __milk_path_remove PATH "$_DEV_INSTALLDIR/bin"
    __milk_path_remove LD_LIBRARY_PATH "$_DEV_INSTALLDIR/lib"
    __milk_path_remove PKG_CONFIG_PATH "$_DEV_INSTALLDIR/lib/pkgconfig"
    __milk_path_remove PYTHONPATH "$_DEV_INSTALLDIR/python"

    __milk_path_remove PATH "/usr/local/milk/bin"
    __milk_path_remove LD_LIBRARY_PATH "/usr/local/milk/lib"
    __milk_path_remove PKG_CONFIG_PATH "/usr/local/milk/lib/pkgconfig"
    __milk_path_remove PYTHONPATH "/usr/local/milk/python"

    for vdir in /usr/local/milk-*; do
        [ -d "$vdir/bin" ] && __milk_path_remove PATH "$vdir/bin"
        [ -d "$vdir/lib" ] && __milk_path_remove LD_LIBRARY_PATH "$vdir/lib"
        [ -d "$vdir/lib/pkgconfig" ] && __milk_path_remove PKG_CONFIG_PATH "$vdir/lib/pkgconfig"
        [ -d "$vdir/python" ] && __milk_path_remove PYTHONPATH "$vdir/python"
    done

    # 2. Add framework-dev paths
    __milk_path_prepend PATH "$_FDEV_INSTALLDIR/bin"
    __milk_path_prepend LD_LIBRARY_PATH "$_FDEV_INSTALLDIR/lib"
    __milk_path_prepend PKG_CONFIG_PATH "$_FDEV_INSTALLDIR/lib/pkgconfig"
    __milk_path_prepend PYTHONPATH "$_FDEV_INSTALLDIR/python"

    # Export standard environment variables
    export MILK_BRANCH="framework-dev"
    export MILK_ROOT="$_FDEV_ROOT"
    export MILK_INSTALLDIR="$_FDEV_INSTALLDIR"
    export MILK_SHM_DIR="$_FDEV_SHMDIR"
    export MILK_PROC_DIR="$_FDEV_SHMDIR"
    export TMUX_TMPDIR="$_FDEV_SHMDIR"
    unset TMUX
    export PATH
    export LD_LIBRARY_PATH
    export PKG_CONFIG_PATH
    export PYTHONPATH

    # Ensure shared memory directory exists and check permissions
    if [ ! -d "$MILK_SHM_DIR" ]; then
        mkdir -p "$MILK_SHM_DIR" 2>/dev/null || true
        if [ ! -d "$MILK_SHM_DIR" ]; then
            echo -e "\033[1;31m[WARNING] Shared memory directory '${MILK_SHM_DIR}' does not exist and cannot be created without root.\033[0m"
            echo -e "\033[1;33mPlease create it once on the machine: sudo mkdir -m 1777 ${MILK_SHM_DIR}\033[0m"
            if [ -d "/milk/shm" ]; then
                echo -e "\033[0;33mFalling back temporarily to /milk/shm/fdev...\033[0m"
                mkdir -p /milk/shm/fdev 2>/dev/null || true
                export MILK_SHM_DIR="/milk/shm/fdev"
                export MILK_PROC_DIR="/milk/shm/fdev"
                export TMUX_TMPDIR="/milk/shm/fdev"
            fi
        fi
    fi

    # Check if framework-dev install directory exists
    if [ ! -d "$_FDEV_INSTALLDIR/bin" ]; then
        echo -e "\033[0;33m[NOTICE] Framework-dev install dir '$_FDEV_INSTALLDIR' does not contain binaries yet.\033[0m"
        echo -e "\033[0;33mTo build and install locally to this directory, run:\033[0m"
        echo -e "  cd \"$MILK_ROOT\" && mkdir -p _build && cd _build"
        echo -e "  cmake .. -DCMAKE_INSTALL_PREFIX=\"$MILK_INSTALLDIR\" -DCMAKE_BUILD_TYPE=Release"
        echo -e "  make -j\$(nproc) && make install"
    fi

    # 3. Command visibility: milk-cli is active; block legacy 'milk'
    unset -f milk-cli 2>/dev/null || true
    milk() {
        echo -e "\033[1;31mError: In framework-dev mode, the CLI command is 'milk-cli'.\033[0m" >&2
        echo -e "Please run '\033[1;32mmilk-cli\033[0m', or switch to dev mode: \033[1;33mmilk-env dev\033[0m" >&2
        return 1
    }

    # 4. Add bold red prompt tag if not already present
    if [[ "$PS1" != *"$_FDEV_PROMPT_TAG"* ]]; then
        PS1="${_FDEV_PROMPT_TAG}${PS1}"
    fi

    echo -e "\033[1;32mSwitched to \033[1;31mframework-dev\033[1;32m environment.\033[0m"
    echo -e "  MILK_SHM_DIR:    $MILK_SHM_DIR"
    echo -e "  MILK_INSTALLDIR: $MILK_INSTALLDIR"
    echo -e "  MILK_ROOT:       $MILK_ROOT"
    echo -e "  TMUX_TMPDIR:     $TMUX_TMPDIR (fenced)"
}

__milk_env_switch_dev() {
    echo -e "\033[1;33mSwitching environment to \033[1;32mdev\033[1;33m (default)...\033[0m"

    # 1. Clean out framework-dev paths
    __milk_path_remove PATH "$_FDEV_INSTALLDIR/bin"
    __milk_path_remove LD_LIBRARY_PATH "$_FDEV_INSTALLDIR/lib"
    __milk_path_remove PKG_CONFIG_PATH "$_FDEV_INSTALLDIR/lib/pkgconfig"
    __milk_path_remove PYTHONPATH "$_FDEV_INSTALLDIR/python"

    # Also clean out potential fallback paths under _install
    if [ -d "$_FDEV_ROOT/_install/bin" ]; then
        __milk_path_remove PATH "$_FDEV_ROOT/_install/bin"
        __milk_path_remove LD_LIBRARY_PATH "$_FDEV_ROOT/_install/lib"
        __milk_path_remove PKG_CONFIG_PATH "$_FDEV_ROOT/_install/lib/pkgconfig"
        __milk_path_remove PYTHONPATH "$_FDEV_ROOT/_install/python"
    fi
    for vdir in "$_FDEV_ROOT"/_install/milk-*; do
        [ -d "$vdir/bin" ] && __milk_path_remove PATH "$vdir/bin"
        [ -d "$vdir/lib" ] && __milk_path_remove LD_LIBRARY_PATH "$vdir/lib"
        [ -d "$vdir/lib/pkgconfig" ] && __milk_path_remove PKG_CONFIG_PATH "$vdir/lib/pkgconfig"
        [ -d "$vdir/python" ] && __milk_path_remove PYTHONPATH "$vdir/python"
    done

    # 2. Add dev paths
    __milk_path_prepend PATH "$_DEV_INSTALLDIR/bin"
    __milk_path_prepend LD_LIBRARY_PATH "$_DEV_INSTALLDIR/lib"
    __milk_path_prepend PKG_CONFIG_PATH "$_DEV_INSTALLDIR/lib/pkgconfig"
    __milk_path_prepend PYTHONPATH "$_DEV_INSTALLDIR/python"

    # Export standard environment variables
    export MILK_BRANCH="dev"
    export MILK_ROOT="$_DEV_ROOT"
    export MILK_INSTALLDIR="$_DEV_INSTALLDIR"
    export MILK_SHM_DIR="$_DEV_SHMDIR"
    export MILK_PROC_DIR="$_DEV_SHMDIR"
    # Keep legacy/dev tmux sessions on system default socket (visible to standard 'tmux ls')
    unset TMUX_TMPDIR
    unset TMUX
    export PATH
    export LD_LIBRARY_PATH
    export PKG_CONFIG_PATH
    export PYTHONPATH

    # 3. Command visibility: milk is active; block 'milk-cli' in dev mode
    unset -f milk 2>/dev/null || true
    milk-cli() {
        echo -e "\033[1;31mError: 'milk-cli' is not a valid command in dev mode.\033[0m" >&2
        echo -e "In dev mode, the CLI command is '\033[1;32mmilk\033[0m'." >&2
        echo -e "To use 'milk-cli', switch to framework-dev: \033[1;33mmilk-env fdev\033[0m" >&2
        return 1
    }

    # 4. Warn if stale milk-cli binary exists in dev install directory
    if [ -f "$_DEV_INSTALLDIR/bin/milk-cli" ]; then
        echo -e "\033[1;33m[WARNING] Stale 'milk-cli' found in $_DEV_INSTALLDIR/bin from an earlier install.\033[0m"
        echo -e "\033[1;33m          In dev mode, the CLI command is 'milk'.\033[0m"
        echo -e "\033[1;33m          To avoid PATH confusion, remove it: sudo rm -f $_DEV_INSTALLDIR/bin/milk-cli*\033[0m"
    fi

    # 5. Strip bold red prompt tag if present
    while [[ "$PS1" == *"$_FDEV_PROMPT_TAG"* ]]; do
        PS1="${PS1/"$_FDEV_PROMPT_TAG"/}"
    done

    echo -e "\033[1;32mSwitched to \033[1;32mdev\033[1;32m environment.\033[0m"
    echo -e "  MILK_SHM_DIR:    $MILK_SHM_DIR"
    echo -e "  MILK_INSTALLDIR: $MILK_INSTALLDIR"
    echo -e "  MILK_ROOT:       $MILK_ROOT"
    echo -e "  TMUX_TMPDIR:     unset (system default, visible to tmux ls)"
}

# Main command dispatch
case "$1" in
    fdev|framework-dev|frameworkdev)
        __milk_env_switch_fdev
        ;;
    dev)
        __milk_env_switch_dev
        ;;
    status)
        __milk_env_status
        ;;
    -h|--help|help)
        echo "Usage: source milk-env.sh [fdev|dev|status]"
        echo "  fdev    : Switch to framework-dev branch (SHM: $_FDEV_SHMDIR)"
        echo "  dev     : Switch to dev branch (SHM: $_DEV_SHMDIR)"
        echo "  status  : Print current active environment configuration"
        ;;
    "")
        __milk_env_status
        echo ""
        echo "Usage: source milk-env.sh [fdev|dev|status]"
        ;;
    *)
        echo -e "\033[1;31mUnknown argument: $1\033[0m" >&2
        echo "Usage: source milk-env.sh [fdev|dev|status]" >&2
        ;;
esac

# Clean up helper function definitions from the interactive environment
unset -f __milk_find_fdev_root __milk_find_dev_root
unset -f __milk_find_fdev_installdir __milk_find_dev_installdir
unset -f __milk_env_status __milk_env_switch_fdev __milk_env_switch_dev
unset _FDEV_ROOT _FDEV_INSTALLDIR _FDEV_SHMDIR
unset _DEV_ROOT _DEV_INSTALLDIR _DEV_SHMDIR _FDEV_PROMPT_TAG
