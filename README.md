[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![REUSE status](https://api.reuse.software/badge/github.com/milk-org/milk)](https://api.reuse.software/info/github.com/milk-org/milk)
[![Documentation](https://img.shields.io/badge/docs-milk--org.github.io-blue?logo=readthedocs)](https://milk-org.github.io/milk/)

Latest Version: [![latesttag](https://img.shields.io/github/tag/milk-org/milk.svg)](https://github.com/milk-org/milk/tree/master)

| Branch   | Build                                                                                                                                                              | Docker Deployment                                                                                                                                                                | Activity                                                                        |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **main** | [![CMake badge](https://github.com/milk-org/milk/actions/workflows/cmake.yml/badge.svg?branch=main)](https://github.com/milk-org/milk/actions/workflows/cmake.yml) | [![CMake badge](https://github.com/milk-org/milk/actions/workflows/docker-image.yml/badge.svg?branch=main)](https://github.com/milk-org/milk/actions/workflows/docker-image.yml) | ![lastcommit](https://img.shields.io/github/last-commit/milk-org/milk/main.svg) |
| **dev**  | [![CMake badge](https://github.com/milk-org/milk/actions/workflows/cmake.yml/badge.svg?branch=dev)](https://github.com/milk-org/milk/actions/workflows/cmake.yml)  | [![CMake badge](https://github.com/milk-org/milk/actions/workflows/docker-image.yml/badge.svg?branch=dev)](https://github.com/milk-org/milk/actions/workflows/docker-image.yml)  | ![lastcommit](https://img.shields.io/github/last-commit/milk-org/milk/dev.svg)  |

[![Docs Lint](https://github.com/milk-org/milk/actions/workflows/docs-lint.yml/badge.svg)](https://github.com/milk-org/milk/actions/workflows/docs-lint.yml)

Code metrics (dev branch) :
[![CodeScene Code Health](https://codescene.io/projects/14777/status-badges/code-health)](https://codescene.io/projects/14777)
[![CodeScene System Mastery](https://codescene.io/projects/14777/status-badges/system-mastery)](https://codescene.io/projects/14777)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1c9a67a8529340359a2047eba5c971bf)](https://www.codacy.com/gh/milk-org/milk/dashboard?utm_source=github.com&utm_medium=referral&utm_content=milk-org/milk&utm_campaign=Badge_Grade)
[![CodeFactor](https://www.codefactor.io/repository/github/milk-org/milk/badge)](https://www.codefactor.io/repository/github/milk-org/milk)

---

# Milk

milk-core for **milk** package

> **📖 [Documentation](https://milk-org.github.io/milk/)** · **🚀 [Getting Started](https://milk-org.github.io/milk/install/compile/)**

## _Looking for something else?_

[CACAO Github repository](https://www.github.com/cacao-org/cacao) | [CACAO Documentation](https://cacao-org.github.io/docs/)

[ImageStreamIO core library](https://www.github.com/milk-org/imagestreamio)

[pyMilk bindings](https://www.github.com/milk-org/pymilk)

## Contents

Module includes key frameworks:

- [**Image streams**](docs/streams.md) — low-latency shared memory streams
- [**processinfo**](docs/procinfo.md) — process management and control
- [**Function Parameter Structure (FPS)**](docs/fps.md) — reading/writing function parameters. See [FPS Standalone and CMD Modes](docs/FPS_Standalone_CMD_Modes.md) for implementation details.

For a comprehensive guide, see the [Documentation Index](docs/index.md).
For a full list of all available documentation in this repository, see the [Markdown Documentation Index](docs/Markdown_Index.md).

## Download

```bash
$ git clone --recursive https://github.com/milk-org/milk.git
$ cd milk
```

## Build

### Quick start (full build)

Builds everything: interactive CLI, TUI, all plugins, and
standalone fpsexec programs.

```bash
$ mkdir _build && cd _build
$ cmake ..
$ make -j$(nproc)
$ sudo make install
```

### Build tiers

The build system supports four tiers with decreasing
dependency requirements:

| Tier               | cmake command                               | External deps              |
| ------------------ | ------------------------------------------- | -------------------------- |
| **Engine**         | `cmake .. -DUSE_COREMODS=OFF -DUSE_CLI=OFF` | POSIX only                 |
| **Core**           | `cmake .. -DUSE_CLI=OFF -DUSE_CFITSIO=OFF`  | POSIX only                 |
| **Core + FITS**    | `cmake .. -DUSE_CLI=OFF`                    | cfitsio                    |
| **Full** (default) | `cmake ..`                                  | cfitsio, readline, ncurses |

For details on each tier, what gets built, and what is
disabled, see [Build Tiers](docs/install/build_tiers.md).

### CMake options

| Option         | Default | Description                                       |
| -------------- | ------- | ------------------------------------------------- |
| `USE_COREMODS` | ON      | Build core modules (COREMOD\_\*)                  |
| `USE_CFITSIO`  | ON      | Build FITS I/O (requires cfitsio)                 |
| `USE_CLI`      | ON      | Build interactive CLI (`milk-cli`), TUI, scripts  |
| `USE_NCURSES`  | ON      | Enable ncurses TUI (`milk-fpsCTRL`, `streamCTRL`) |
| `USE_READLINE` | ON      | Enable readline for CLI input                     |
| `USE_GSL`      | ON      | Enable GSL for plugins                            |
| `USE_CUDA`     | OFF     | Enable CUDA GPU acceleration                      |

### Build with Python module

```bash
$ ./compile.sh $PWD/local
```

### Environment variables

Set in `.bashrc` or equivalent:

| Variable          | Example               | Purpose                 |
| ----------------- | --------------------- | ----------------------- |
| `MILK_ROOT`       | `/home/user/src/milk` | Source code directory   |
| `MILK_INSTALLDIR` | `/usr/local/milk`     | Installation directory  |
| `MILK_SHM_DIR`    | `/milk/shm`           | Shared memory directory |

### Verify installation

```bash
$ milk-check
```

For post-installation steps and dependency details, see
[Installation](docs/install/compile.md).

## Interactive tutorial

Pre-requisites: tmux, nnn

```bash
$ milk-tutorial
```

## Adding plugins

Compile with cacao plugins:

```bash
$ ./fetch_cacao_dev.sh
$ ./compile.sh $PWD/local
```

Compile with coffee plugins:

```bash
$ ./fetch_coffee_dev.sh
$ ./compile.sh $PWD/local
```
