# Module: ZernikePolyn

Create and fit Zernike polynomials for wavefront analysis.

## Source Files

| File              | Description                             |
| ----------------- | --------------------------------------- |
| `mkzercube.c`     | Generate Zernike modes as 3D image cube |
| `zernike_value.c` | Evaluate Zernike polynomial at a point  |

## Standalone Executables

| Executable                       | Source File   | Description                 |
| -------------------------------- | ------------- | --------------------------- |
| `milk-fpsexec-zernike-mkzercube` | `mkzercube.c` | Generate Zernike modes cube |

## Dependencies

- `CLIcore` (includes transitive: `milkfps`, `ImageStreamIO`, `milkdata`)
