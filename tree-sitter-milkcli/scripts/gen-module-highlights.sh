#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Max Brunsfeld
#
# SPDX-License-Identifier: MIT

# gen-module-highlights.sh — Generate highlight queries
#                             for dynamic milk-cli commands
#
# Usage:
#   ./scripts/gen-module-highlights.sh
#   ./scripts/gen-module-highlights.sh --install
#   ./scripts/gen-module-highlights.sh --stdout
#
# Reads the install-time command catalog to list all module
# commands (mem.listim, iofits.saveFITS, etc.) and
# generates tree-sitter highlight predicates so
# they appear as @function.builtin in Neovim.
#
# Without --install, writes to queries/modules.scm.
# With --install, appends to the Neovim query dir.
# With --stdout, prints to stdout (for piping).
#
# Prerequisites: milk must be installed and in PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAMMAR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NVIM_CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/nvim"
NVIM_QUERIES="$NVIM_CONFIG/after/queries/milkcli"
OUTPUT="$GRAMMAR_DIR/queries/modules.scm"

# ---- Color helpers ----
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*" >&2; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*" >&2; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---- The command catalog written at install (milk-cli --catalog) ----
if [[ -z "${MILK_INSTALLDIR:-}" ]] && command -v milk-cli &>/dev/null; then
    MILK_INSTALLDIR="$(cd "$(dirname "$(command -v milk-cli)")/.." && pwd)"
fi
CATALOG="${MILK_INSTALLDIR:-}/share/milk/catalog.d"
if [[ ! -d "${CATALOG}" ]]; then
    err "No command catalog (${CATALOG}): install milk, or set MILK_INSTALLDIR."
    exit 1
fi

info "Reading the module commands of ${CATALOG}..."
CMD_LIST=$(cut -f1 "${CATALOG}"/*.tsv | sort -u)

COUNT=$(echo "$CMD_LIST" | wc -l)
info "Found $COUNT module commands"

# ---- Generate highlight queries ----
generate_queries() {
    echo "; modules.scm — Auto-generated highlight rules"
    echo "; for dynamically loaded milk-cli module commands"
    echo ";"
    echo "; Generated: $(date -Iseconds)"
    echo "; Commands:  $COUNT"
    echo ";"
    echo "; Regenerate with:"
    echo ";   ./scripts/gen-module-highlights.sh"
    echo ""
    echo "; Match registered module commands as builtins"
    echo "((module_command) @function.builtin"
    echo " (#any-of? @function.builtin"

    echo "$CMD_LIST" | while read -r cmd; do
        echo "  \"$cmd\""
    done

    echo " ))"
}

# ---- Output mode ----
case "${1:-}" in
    --stdout)
        generate_queries
        ;;
    --install)
        # Write to both local and Neovim query dir
        generate_queries > "$OUTPUT"
        ok "Written to $OUTPUT"

        if [[ -d "$NVIM_QUERIES" ]]; then
            cp "$OUTPUT" "$NVIM_QUERIES/modules.scm"
            ok "Installed to $NVIM_QUERIES/modules.scm"
            info "Restart Neovim to pick up changes"
        else
            info "Neovim queries dir not found:"
            info "  $NVIM_QUERIES"
            info "Run ./scripts/nvim-install.sh first"
        fi
        ;;
    *)
        generate_queries > "$OUTPUT"
        ok "Written to $OUTPUT ($COUNT commands)"
        echo ""
        info "To install into Neovim, run:"
        info "  ./scripts/gen-module-highlights.sh --install"
        ;;
esac
