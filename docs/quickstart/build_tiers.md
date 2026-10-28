# Build Tiers

<!-- prettier-ignore -->
!!! note
    The milk build system supports multiple build tiers, controlled by CMake options. Lower tiers
    have fewer external dependencies and produce a smaller footprint.

---

## Overview

The default build is the **Full build**, but without extra dependencies.

| Build layer     | Code pulled in                                                                 | Dependencies                                                                                       |
| --------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| **Full build**  | CLI (`milk-cli`, `milk-script`, `milk-call`, `milk.bash`, `milk-CTRL`)         | readline, tree-sitter (bundled), ncurses                                                           |
| **Core + FITS** | `COREMOD_iofits`, plugins, `fpsexec` compute units                             | cfitsio, per-plugin deps (see [CMake system](../developer/dependency_system.md))                   |
| **Core**        | `COREMOD_memory`, `COREMOD_arith`, `COREMOD_tools`, plugins, `fpsexec`s        |                                                                                                    |
| **Engine**      | `ImageStreamIO`, `libmilkcommon`, `libprocessinfo`, `libfps`, `libcomputeunit`, `libmilkdata`, engine TUIs (`milk-fpsCTRL`, `milk-procCTRL`) | POSIX-only (+ ncurses for TUIs)                                                                    |

## Optional dependencies

Some optional dependencies are detected by MILK and enable additional features:

| Option           | Default | Description                                        |
| ---------------- | ------- | -------------------------------------------------- |
| `USE_READLINE`   | ON      | Enable readline for interactive CLI input          |
| `USE_NCURSES`    | ON      | Enable ncurses for TUI dashboards                  |
| `USE_TREESITTER` | ON      | Enable tree-sitter syntax highlighting in CLI      |
| `USE_CFITSIO`    | ON      | Enable FITS file I/O (`COREMOD_iofits`)            |
| `USE_OPENBLAS`   | ON      | BLAS: multicore linear algebra on CPU (if found)   |
| `USE_LAPACKE`    | ON      | LAPACKE linear algebra routines (if found)         |
| `USE_MKL`        | OFF     | Intel MKL compute library                          |
| `USE_CUDA`       | OFF     | CUDA GPU computations                              |
| `USE_MAGMA`      | OFF     | MAGMA linear algebra on GPU routines               |
| `USE_LTO`        | ON      | Enable Link-Time Optimization (`-flto=auto`)       |
| `USE_STATIC_LTO` | OFF     | Link standalone `fpsexec`s against static archives |

If you have the dependency `<X>` installed, you may request MILK to fetch it with `-DUSE_<X>=ON`.

CMake will issue warnings when some features are disabled due to a missing dependency -- either because `-DUSE_<X>=OFF` was set, or because the library wasn't found by CMake.

<!-- prettier-ignore -->
!!! note
    **There is crosstalk between those dependencies:**
    - MKL always provides BLAS and Lapacke
    - OpenBLAS always provides BLAS, sometimes LAPACKE
    - Lapacke can be installed standalone, and MKL and OpenBLAS deactivated (in which case BLAS
      features are missing, but Lapacke features are enabled).

See [Dependency System](../developer/dependency_system.md) on how to integrate these optional
dependencies as a developer.

## How to build each tier of MILK

| Tier               | cmake command                               | What you get                                                       |
| ------------------ | ------------------------------------------- | ------------------------------------------------------------------ |
| **Engine**         | `cmake .. -DUSE_COREMODS=OFF -DUSE_CLI=OFF` | ImageStreamIO, processinfo, FPS, computeunit, milkdata, engine TUIs |
| **Core**           | `cmake .. -DUSE_CLI=OFF -DUSE_CFITSIO=OFF`  | + COREMOD_arith, memory, tools, non-FITS plugins and `fpsexec`s    |
| **Core + FITS**    | `cmake .. -DUSE_CLI=OFF`                    | + COREMOD_iofits and FITS-enabled plugins / `fpsexec`s             |
| **Full**           | `cmake ..`                                  | + `milk-cli`, `milk-script`, `milk.bash`, `milk-CTRL`, all plugins |
| **Full + options** | `cmake .. -DUSE_<X_OPTIONAL_DEP>=ON`        | + GPU acceleration (CUDA, MAGMA) or Intel MKL                      |

All tiers are built with:

```bash
$ mkdir _build && cd _build
$ cmake .. [OPTIONS]        # see table above
$ make -j$(nproc)
$ sudo make install
```

<!-- prettier-ignore-start -->
!!! info
    **Quirks on the build variables:**
    - `USE_CLI=ON` automatically enables `USE_COREMODS=ON`
    - Plugins are only built when `USE_COREMODS=ON`
    - `USE_CFITSIO=OFF` excludes `COREMOD_iofits` and compiles remaining modules without cfitsio linkage.

    ## Behavior When `cfitsio` Is Disabled

!!! info
    **when FITS capability is removed from -DUSE_CFITSIO=OFF**
    FITS-dependent code paths are compiled out via `#ifdef USE_CFITSIO` guards:

    | Module           | Effect                                                                               |
    | ---------------- | ------------------------------------------------------------------------------------ |
    | `COREMOD_iofits` | Not built — `iofits.loadfits` / `iofits.saveFITS` unavailable                        |
    | `COREMOD_memory` | `logshmim`, `saveall`, `stream_pixmapdecode` print a warning instead of writing FITS |
    | `COREMOD_arith`  | No impact (never used cfitsio)                                                       |
    | `COREMOD_tools`  | No impact                                                                            |

!!! tip
    The engine and core tiers are ideal for embedded or real-time deployments where only
    shared-memory stream processing is needed and disk I/O to FITS files is unnecessary.
<!-- prettier-ignore-end -->

---

← [Documentation Index](../index.md)
