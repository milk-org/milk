# Build Tiers

<!-- prettier-ignore -->
!!! note
    The milk build system supports multiple build tiers, controlled by CMake options. Lower tiers
    have fewer external dependencies and produce a smaller footprint.

---

## Overview

The default build is the **Full build**, but without extra dependencies.

| Build layer          | Code pulled in                                                                                                    | Dependencies                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Full build**       | CLI (`milk-cli`, `milk-script`, `milk-call`, `milk.bash`, `milk-CTRL`)                                            | readline, tree-sitter (bundled), ncurses                                         |
| **Calculator & JIT** | `milkprop`, `milkcalc` (C-codegen JIT, `milk_calc_ext.h`), `milk-fpsexec-milk-calc`                               | C compiler (`gcc`), optional OpenMP / OpenBLAS / LAPACKE / FFTW                  |
| **Plugins**          | `milk-extra` and `cacao` plugins + standalone `fpsexec` compute units                                             | Per-plugin deps (see [CMake system](../developer/dependency_system.md))          |
| **Core + FITS**      | `COREMOD_iofits`                                                                                                  | cfitsio                                                                          |
| **Core**             | `COREMOD_memory`, `COREMOD_arith`, `COREMOD_tools`                                                                |                                                                                  |
| **Engine**           | `ImageStreamIO`, `libmilkcommon`, `libprocessinfo`, `libfps`, `libcomputeunit`, `libmilkdata`, standalone tools (`milk-fps-*`, `milk-fpsCTRL`, `milk-procCTRL`) | Linux, see [Engine portability](#engine-portability)                             |

## Engine portability

The engine needs Linux. Its libraries keep to POSIX.1-2024 with XSI, except the files listed in
`MILK_ENGINE_NONPOSIX_SOURCES` in `cmake/milk_layer_guards.cmake`, which use Linux or GNU calls.
CI checks this with a strict POSIX build (`-DMILK_STRICT_POSIX=ON`). The strict flags do not
apply to `ImageStreamIO`, a separate repository, or to the standalone tools.

## Optional dependencies

Some optional dependencies are detected by MILK and enable additional features:

| Option           | Default | Description                                                |
| ---------------- | ------- | ---------------------------------------------------------- |
| `USE_READLINE`   | ON      | Enable readline for interactive CLI input                  |
| `USE_NCURSES`    | ON      | Enable ncurses support (CLI help TUI, `milkinfo`)          |
| `USE_TREESITTER` | ON      | Enable tree-sitter syntax highlighting in CLI              |
| `USE_CFITSIO`    | ON      | Enable FITS file I/O (`COREMOD_iofits`)                    |
| `USE_PLUGINS`    | ON      | Build plugin modules (`plugins/*`) and their `fpsexec`s    |
| `USE_CALC`       | ON      | Build the calculator and its JIT (`milkprop`, `milkcalc`)  |
| `USE_OPENBLAS`   | ON      | BLAS: multicore linear algebra on CPU (if found)           |
| `USE_LAPACKE`    | ON      | LAPACKE linear algebra routines (if found)                 |
| `USE_MKL`        | OFF     | Intel MKL compute library                                  |
| `USE_CUDA`       | OFF     | CUDA GPU computations                                      |
| `USE_LTO`        | ON      | Enable Link-Time Optimization (`-flto=auto`)               |
| `USE_STATIC_LTO` | OFF     | Link standalone `fpsexec`s against static archives         |

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

| Tier               | cmake command                                                              | What you get                                                                 |
| ------------------ | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Engine**         | `cmake .. -DUSE_COREMODS=OFF -DUSE_CLI=OFF`                                | ImageStreamIO, processinfo, FPS, computeunit, milkdata, standalone tools     |
| **Core**           | `cmake .. -DUSE_CLI=OFF -DUSE_PLUGINS=OFF -DUSE_CALC=OFF -DUSE_CFITSIO=OFF`| + `COREMOD_arith`, `COREMOD_memory`, `COREMOD_tools` (no cfitsio)            |
| **Core + FITS**    | `cmake .. -DUSE_CLI=OFF -DUSE_PLUGINS=OFF -DUSE_CALC=OFF`                  | + `COREMOD_iofits` (requires cfitsio)                                        |
| **Plugins**        | `cmake .. -DUSE_CLI=OFF -DUSE_CALC=OFF`                                    | + `milk-extra` and `cacao` plugins & standalone `fpsexec`s                   |
| **Calc & JIT**     | `cmake .. -DUSE_CLI=OFF -DUSE_PLUGINS=OFF`                                 | + `milkprop`, `milkcalc` (JIT) & `milk-fpsexec-milk-calc`                    |
| **Standalone**     | `cmake .. -DUSE_CLI=OFF`                                                   | + both Plugins and Calc & JIT (all standalone `fpsexec`s, no CLI)            |
| **Full**           | `cmake ..`                                                                 | + `milk-cli`, `milk-script`, `milk.bash`, `milk-CTRL`, `milk-termview`       |
| **Full + options** | `cmake .. -DUSE_<X_OPTIONAL_DEP>=ON`                                       | + GPU acceleration (CUDA) or Intel MKL                                       |

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
    - `USE_CLI=ON` automatically enables `USE_COREMODS=ON` and `USE_CALC=ON`
    - `USE_PLUGINS` and `USE_CALC` are active when `USE_COREMODS=ON` and can be toggled independently in `-DUSE_CLI=OFF` builds
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
