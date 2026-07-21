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
# Runs milk-cli to discover all registered module
# commands (listim, saveFITS, imcrop, etc.) and
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

# ---- Check milk is available ----
if ! command -v milk &>/dev/null; then
    err "milk not found in PATH."
    err "Make sure milk is installed and sourced."
    err "  source /usr/local/milk/bin/milk-setup.sh"
    exit 1
fi

# ---- Extract command list from milk-cli ----
info "Extracting registered commands from milk-cli..."

# Run milk with cmd? to list all commands, extract
# the first column (command name). Filter out:
#  - empty lines
#  - header lines (starting with -)
#  - help/internal commands (already in grammar)
CMDS=$(milk << 'MILKEOF' 2>/dev/null
cmd?
exitCLI
MILKEOF
)

# Parse out just command names (first word of each
# non-header line).
# The cmd? output format is:
#   cmdname   arginfo   description
CMD_LIST=$(echo "$CMDS" \
    | grep -v '^\s*$' \
    | grep -v '^\s*-' \
    | grep -v '^milk' \
    | grep -v '^$' \
    | awk '{print $1}' \
    | grep -v '^cmd' \
    | grep -E '^[a-zA-Z_][a-zA-Z0-9_.]*$' \
    | sort -u)

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
    echo "((identifier) @function.builtin"
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
