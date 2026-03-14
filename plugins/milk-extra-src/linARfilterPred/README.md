# Module: linARfilterPred

Linear AutoRegressive prediction filter

## Source Files

| File | Description |
|------|-------------|
| `applyPF.c` | Apply predictive filter to stream |
| `build_linPF.c` | Build linear predictive filter from training data |

## Standalone Executables

| Executable | Source File | Description |
|------------|-------------|-------------|
| `milk-fpsexec-linpred-applyPF` | `applyPF.c` | Apply predictive filter to stream |
| `milk-fpsexec-linpred-buildlinPF` | `build_linPF.c` | Build linear predictive filter |

## Dependencies

- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
