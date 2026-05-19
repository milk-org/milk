#!/usr/bin/env bash
# nvim-install.sh — Install milkcli parser + queries
#                    into Neovim
#
# Usage:
#   ./scripts/nvim-install.sh           # full install
#   ./scripts/nvim-install.sh --check   # dry-run
#   ./scripts/nvim-install.sh --uninstall
#
# This script:
#  1. Installs nvim-treesitter if not present
#  2. Copies highlight queries to Neovim runtime
#  3. Generates a Lua config for parser registration
#  4. Compiles the parser .so
#
# Supports out-of-the-box: lazy.nvim, packer,
# vim-plug, or no plugin manager.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAMMAR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NVIM_CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/nvim"
NVIM_DATA="${XDG_DATA_HOME:-$HOME/.local/share}/nvim"
NVIM_QUERIES="$NVIM_CONFIG/after/queries/milkcli"
NVIM_PLUGIN_DIR="$NVIM_CONFIG/plugin"
LUA_SNIPPET="$NVIM_PLUGIN_DIR/milkcli-treesitter.lua"

# Site pack path for manual plugin installs
SITE_PACK="$NVIM_DATA/site/pack/milkcli/start"

# ---- Color helpers ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC}  $*"; }

# ---- Uninstall mode ----
if [[ "${1:-}" == "--uninstall" ]]; then
    info "Uninstalling milkcli from Neovim..."

    [[ -d "$NVIM_QUERIES" ]] \
        && rm -rf "$NVIM_QUERIES" \
        && ok "Removed $NVIM_QUERIES"
    [[ -f "$LUA_SNIPPET" ]] \
        && rm -f "$LUA_SNIPPET" \
        && ok "Removed $LUA_SNIPPET"

    # Remove compiled parser if present
    PARSER_SO="$NVIM_DATA/site/parser/milkcli.so"
    [[ -f "$PARSER_SO" ]] \
        && rm -f "$PARSER_SO" \
        && ok "Removed $PARSER_SO"

    info "Uninstall complete."
    exit 0
fi

# ---- Preflight checks ----
CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=1
    info "Dry-run mode — no files will be modified"
    echo ""
fi

# Check parser was built
if [[ ! -f "$GRAMMAR_DIR/src/parser.c" ]]; then
    err "Parser not built. Run: ./scripts/build.sh"
    exit 1
fi
ok "Parser source: $GRAMMAR_DIR/src/parser.c"

# Check Neovim
if ! command -v nvim &>/dev/null; then
    err "Neovim not found in PATH"
    exit 1
fi
NVIM_VER=$(nvim --version | head -1)
ok "Neovim: $NVIM_VER"

# Check if C compiler is available (needed to
# compile parser .so)
if command -v cc &>/dev/null; then
    ok "C compiler found: $(cc --version | head -1)"
elif command -v gcc &>/dev/null; then
    ok "C compiler found: $(gcc --version | head -1)"
else
    warn "No C compiler found (cc/gcc)."
    warn "Parser .so compilation will fail."
fi

# ---- Detect nvim-treesitter ----
find_nvim_treesitter() {
    # Search common plugin manager locations
    local search_dirs=(
        "$NVIM_DATA/lazy/nvim-treesitter"
        "$NVIM_DATA/site/pack/packer/start/nvim-treesitter"
        "$NVIM_DATA/plugged/nvim-treesitter"
        "$SITE_PACK/nvim-treesitter"
    )
    for d in "${search_dirs[@]}"; do
        if [[ -d "$d" ]]; then
            echo "$d"
            return 0
        fi
    done

    # Broad search as fallback
    local found
    found=$(find "$NVIM_DATA" -maxdepth 5 \
        -type d -name "nvim-treesitter" \
        2>/dev/null | head -1)
    if [[ -n "$found" ]]; then
        echo "$found"
        return 0
    fi

    return 1
}

TS_DIR=""
if TS_DIR=$(find_nvim_treesitter); then
    ok "nvim-treesitter: $TS_DIR"
else
    warn "nvim-treesitter plugin not found."
    echo ""
    info "nvim-treesitter is required for :TSInstall"
    info "and automatic parser management."
    echo ""
    info "Options:"
    info "  1) Install now (git clone to Neovim"
    info "     site/pack — no plugin manager needed)"
    info "  2) Skip — you'll need to compile the"
    info "     parser .so manually"
    echo ""

    if [[ $CHECK_ONLY -eq 1 ]]; then
        warn "Skipping install (dry-run mode)"
    else
        read -rp "Install nvim-treesitter now? [Y/n] " \
            ans
        ans="${ans:-y}"
        if [[ "$ans" =~ ^[Yy] ]]; then
            info "Cloning nvim-treesitter..."
            mkdir -p "$SITE_PACK"
            git clone --depth 1 \
                https://github.com/nvim-treesitter/nvim-treesitter.git \
                "$SITE_PACK/nvim-treesitter"
            TS_DIR="$SITE_PACK/nvim-treesitter"
            ok "Installed to $TS_DIR"
        else
            warn "Skipping nvim-treesitter install"
        fi
    fi
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
    echo ""
    info "Dry-run complete. Run without --check to install."
    exit 0
fi

# ---- Step 1: Copy highlight queries ----
info "Installing highlight queries..."
mkdir -p "$NVIM_QUERIES"
cp "$GRAMMAR_DIR/queries/highlights.scm" \
   "$NVIM_QUERIES/highlights.scm"
ok "highlights.scm -> $NVIM_QUERIES/"

# Copy optional query files if they exist
for qf in locals.scm folds.scm textobjects.scm \
           modules.scm; do
    if [[ -f "$GRAMMAR_DIR/queries/$qf" ]]; then
        cp "$GRAMMAR_DIR/queries/$qf" \
           "$NVIM_QUERIES/$qf"
        ok "$qf -> $NVIM_QUERIES/"
    fi
done

# ---- Step 2: Generate Lua config snippet ----
info "Generating Neovim config..."
mkdir -p "$NVIM_PLUGIN_DIR"

cat > "$LUA_SNIPPET" << 'LUAEOF'
-- milkcli-treesitter.lua
-- Auto-generated by tree-sitter-milkcli
-- Re-run scripts/nvim-install.sh to regenerate.

-- Register .milk file extension
vim.filetype.add({
    extension = {
        milk = "milkcli",
    },
    pattern = {
        [".*"] = {
            priority = -math.huge,
            function(_path, bufnr)
                local line = vim.api.nvim_buf_get_lines(
                    bufnr, 0, 1, false
                )[1] or ""
                if line:match("^#!.*milk") then
                    return "milkcli"
                end
            end,
        },
    },
})

-- Parser registration
vim.treesitter.language.register("milkcli", "milkcli")

-----------------------------------------------
-- milk-cli color palette (dark backgrounds)
--
-- Designed for maximum readability regardless
-- of the active Neovim colorscheme.
--
-- Uses @group.milkcli scoped highlight groups
-- so these colors only apply to .milk files.
-----------------------------------------------
local hl = {
    -- Comments: muted gray, italic
    ["@comment.milkcli"] = {
        fg = "#676e95", italic = true,
    },

    -- Strings: soft green
    ["@string.milkcli"]  = { fg = "#c3e88d" },
    ["@string.special.path.milkcli"] = {
        fg = "#c3e88d", underline = true,
    },
    ["@string.special.milkcli"] = {
        fg = "#c3e88d",
    },

    -- Numbers: warm orange, bold
    ["@number.milkcli"] = {
        fg = "#f78c6c", bold = true,
    },

    -- Flow keywords (if/for/while/fi/done): purple
    ["@keyword.milkcli"] = {
        fg = "#c792ea", bold = true,
    },
    ["@keyword.return.milkcli"] = {
        fg = "#c792ea", italic = true,
    },
    ["@keyword.operator.milkcli"] = {
        fg = "#89ddff",
    },

    -- Booleans: orange like numbers
    ["@boolean.milkcli"] = {
        fg = "#f78c6c", bold = true,
    },

    -- Shell builtins (echo, export): cyan
    ["@function.builtin.milkcli"] = {
        fg = "#89ddff",
    },

    -- *** milk commands: bright gold, bold ***
    -- assigncheck, procctl, waitfor_stream,
    -- on_update, dpdigits, etc.
    ["@function.macro.milkcli"] = {
        fg = "#ffcb6b", bold = true,
    },

    -- Generic commands: blue
    ["@function.call.milkcli"] = {
        fg = "#82aaff",
    },

    -- Function definitions: blue, bold
    ["@function.milkcli"] = {
        fg = "#82aaff", bold = true,
    },

    -- FPS/sequencer vars (@fps.*, @seq.*): green
    ["@property.milkcli"] = {
        fg = "#c3e88d", bold = true,
    },

    -- Stream metadata (${s.wfs.cnt0}): teal
    ["@type.milkcli"] = {
        fg = "#64ffda", bold = true,
    },

    -- Variable names in assignments
    ["@variable.parameter.milkcli"] = {
        fg = "#eeffff",
    },

    -- Variable expansions ($VAR, ${VAR}): amber
    ["@variable.builtin.milkcli"] = {
        fg = "#e2b93d",
    },

    -- Operators (=, |, |>, &&, ||)
    ["@operator.milkcli"] = { fg = "#89ddff" },

    -- Punctuation: subtle cyan
    ["@punctuation.bracket.milkcli"] = {
        fg = "#89ddff",
    },

    -- Command substitution
    ["@embedded.milkcli"] = { fg = "#82aaff" },
}

-- Apply highlights + start treesitter
vim.api.nvim_create_autocmd("FileType", {
    pattern = "milkcli",
    callback = function()
        for group, attrs in pairs(hl) do
            vim.api.nvim_set_hl(0, group, attrs)
        end
        vim.treesitter.start()
    end,
})
LUAEOF

ok "Created $LUA_SNIPPET"

# ---- Step 3: Compile parser .so ----
info "Compiling parser shared library..."

PARSER_OUT="$NVIM_DATA/site/parser/milkcli.so"
mkdir -p "$(dirname "$PARSER_OUT")"

CC="${CC:-cc}"
if $CC -o "$PARSER_OUT" -shared \
    "$GRAMMAR_DIR/src/parser.c" \
    -I "$GRAMMAR_DIR/src" \
    -Os -fPIC 2>&1; then
    ok "Compiled parser: $PARSER_OUT"
else
    warn "Compilation failed. Try manually:"
    warn "  :TSInstall milkcli  (in Neovim)"
fi

# ---- Done ----
echo ""
info "=========================================="
info " Installation complete!"
info "=========================================="
echo ""
info "Files installed:"
info "  $NVIM_QUERIES/highlights.scm"
info "  $LUA_SNIPPET"
[[ -f "$PARSER_OUT" ]] && \
    info "  $PARSER_OUT"
echo ""
info "To verify, open a .milk file in Neovim:"
info "  nvim $GRAMMAR_DIR/examples/demo.milk"
echo ""
info "Inspect the parse tree with  :InspectTree"
echo ""
info "To uninstall:"
info "  ./scripts/nvim-install.sh --uninstall"
