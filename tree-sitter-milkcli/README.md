# tree-sitter-milkcli

Tree-sitter grammar for the **milk-cli** scripting
language, providing syntax highlighting in Neovim
(and other tree-sitter-capable editors).

## Quick Start

```bash
# 1. Build the parser
./scripts/build.sh

# 2. Install into Neovim
./scripts/nvim-install.sh

# 3. In Neovim, compile the parser
:TSInstall milkcli

# 4. Open a .milk file — highlighting is active
nvim examples/demo.milk
```

## What Gets Highlighted

| Syntax | Color Group | Example |
|--------|------------|---------|
| Flow control | `@keyword` | `if`, `for`, `while`, `fi` |
| Shell builtins | `@function.builtin` | `echo`, `export`, `source` |
| milk commands | `@function.macro` | `assigncheck`, `procctl` |
| FPS variables | `@property` | `@fps.loop.gain` |
| Stream metadata | `@type` | `${s.wfs.cnt0}` |
| Variables | `@variable.builtin` | `$VAR`, `${VAR}` |
| Strings | `@string` | `"hello"`, `'literal'` |
| Numbers | `@number` | `42`, `3.14` |
| Comments | `@comment` | `# comment` |

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/build.sh` | Build parser from `grammar.js` |
| `scripts/nvim-install.sh` | Install queries + config into Neovim |
| `scripts/gen-module-highlights.sh` | Generate highlights for runtime module commands |

### `build.sh`

```bash
./scripts/build.sh           # generate parser
./scripts/build.sh --test    # generate + parse example
./scripts/build.sh --clean   # remove generated files
```

### `nvim-install.sh`

```bash
./scripts/nvim-install.sh             # full install
./scripts/nvim-install.sh --check     # dry-run
./scripts/nvim-install.sh --uninstall # remove from nvim
```

Installs:
- `queries/highlights.scm` →
  `~/.config/nvim/after/queries/milkcli/`
- Lua config snippet →
  `~/.config/nvim/plugin/milkcli-treesitter.lua`

### `gen-module-highlights.sh`

Registered module commands (e.g. `listim`, `saveFITS`,
`imcrop`) are loaded dynamically at runtime and not
hardcoded in the grammar. This script extracts all
registered commands from a running `milk` and generates
tree-sitter highlight predicates for them.

```bash
# Generate queries/modules.scm
./scripts/gen-module-highlights.sh

# Generate and install into Neovim
./scripts/gen-module-highlights.sh --install

# Print to stdout (for piping)
./scripts/gen-module-highlights.sh --stdout
```

Run this after installing new milk modules to update
the editor highlighting.

## Directory Structure

```
tree-sitter-milkcli/
├── grammar.js              ← Language grammar
├── queries/
│   └── highlights.scm      ← Neovim highlight queries
├── scripts/
│   ├── build.sh            ← Build the parser
│   ├── nvim-install.sh     ← Install into Neovim
│   └── gen-module-highlights.sh  ← Dynamic commands
├── examples/
│   └── demo.milk           ← Example script
├── package.json            ← npm config
├── tree-sitter.json        ← tree-sitter config
└── src/                    ← Generated (after build)
```

## Requirements

- **Node.js** (for `tree-sitter-cli`)
- **Neovim** ≥ 0.9 (for tree-sitter support)
- **nvim-treesitter** plugin
- **milk** (for `gen-module-highlights.sh`)
