# Module Files

The code is organized in modules under two main directories:

| Location | Contains |
|----------|----------|
| `src/` | Core modules (engine, CLI, coremods) |
| `plugins/` | Optional plugin modules |

Within each module directory:

| Path | Content |
|------|---------|
| `<modulename>.c` | Main C source and module registration |
| `<modulename>.h` | Module header and function prototypes |
| `CMakeLists.txt` | CMake build configuration |
| `README.md` | Module overview, source file list, dependencies |
| `scripts/` | Shell scripts and utilities (optional) |
| `*.c` / `*.h` | Additional source files for individual functions |

Modules are compiled into shared object libraries (`.so`) and loaded
by `milk-cli` at runtime. Standalone executables (`milk-fpsexec-*`)
are built separately and installed to `bin/`.

## Examples

- Core module: `src/coremods/COREMOD_memory/`
- Plugin module: `plugins/milk-extra-src/fft/`
- Template module: `src/milk_module_example/`
