#!/usr/bin/env bash
# build.sh — Build the tree-sitter-milkcli parser
#
# Usage:
#   ./scripts/build.sh          # build + test
#   ./scripts/build.sh --test   # build + parse example
#   ./scripts/build.sh --clean  # remove generated files
#
# Prerequisites: npm (Node.js)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GRAMMAR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$GRAMMAR_DIR"

# ---- Clean mode ----
if [[ "${1:-}" == "--clean" ]]; then
    echo "Cleaning generated files..."
    rm -rf src/parser.c src/tree_sitter/ src/grammar.json
    rm -rf bindings/
    echo "Done."
    exit 0
fi

# ---- Install npm deps if needed ----
if [[ ! -d node_modules ]]; then
    echo "Installing npm dependencies..."
    npm install --silent
fi

# ---- Generate parser ----
echo "Generating parser from grammar.js..."
npx tree-sitter generate

if [[ ! -f src/parser.c ]]; then
    echo "ERROR: parser generation failed"
    exit 1
fi

LINES=$(wc -l < src/parser.c)
echo "Parser generated: src/parser.c ($LINES lines)"

# ---- Optional test ----
if [[ "${1:-}" == "--test" ]]; then
    echo ""
    echo "=== Parsing examples/demo.milk ==="
    npx tree-sitter parse examples/demo.milk 2>/dev/null

    ERRORS=$(npx tree-sitter parse examples/demo.milk \
        2>/dev/null | grep -c "ERROR" || true)
    echo ""
    echo "Parse errors: $ERRORS"

    echo ""
    echo "=== Highlighting examples/demo.milk ==="
    npx tree-sitter highlight examples/demo.milk \
        2>/dev/null || true
fi

echo ""
echo "Build complete. Next steps:"
echo "  ./scripts/nvim-install.sh   # install into Neovim"
echo "  ./scripts/build.sh --test   # test with example"
