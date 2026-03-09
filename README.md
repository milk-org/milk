[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)


Latest Version: [![latesttag](https://img.shields.io/github/tag/milk-org/milk.svg)](https://github.com/milk-org/milk/tree/master)

| Branch    | Build   | Docker Deployment    |  Activity   |
|-------------|-------------|-------------|-------------|
**main**|[![CMake badge](https://github.com/milk-org/milk/actions/workflows/cmake.yml/badge.svg?branch=main)](https://github.com/milk-org/milk/actions/workflows/cmake.yml)|[![CMake badge](https://github.com/milk-org/milk/actions/workflows/docker-image.yml/badge.svg?branch=main)](https://github.com/milk-org/milk/actions/workflows/docker-image.yml)|![lastcommit](https://img.shields.io/github/last-commit/milk-org/milk/main.svg)|
**dev**|[![CMake badge](https://github.com/milk-org/milk/actions/workflows/cmake.yml/badge.svg?branch=dev)](https://github.com/milk-org/milk/actions/workflows/cmake.yml)|[![CMake badge](https://github.com/milk-org/milk/actions/workflows/docker-image.yml/badge.svg?branch=dev)](https://github.com/milk-org/milk/actions/workflows/docker-image.yml)|![lastcommit](https://img.shields.io/github/last-commit/milk-org/milk/dev.svg)|


Code metrics (dev branch) :
[![CodeScene Code Health](https://codescene.io/projects/14777/status-badges/code-health)](https://codescene.io/projects/14777)
[![CodeScene System Mastery](https://codescene.io/projects/14777/status-badges/system-mastery)](https://codescene.io/projects/14777)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1c9a67a8529340359a2047eba5c971bf)](https://www.codacy.com/gh/milk-org/milk/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=milk-org/milk&amp;utm_campaign=Badge_Grade)
[![CodeFactor](https://www.codefactor.io/repository/github/milk-org/milk/badge)](https://www.codefactor.io/repository/github/milk-org/milk)




***

# Milk

milk-core for **milk** package


### _Looking for something else?_
[CACAO Github repository](https://www.github.com/cacao-org/cacao) | [CACAO Documentation](https://cacao-org.github.io/docs/)

[ImageStreamIO core library](https://www.github.com/milk-org/imagestreamio)

[pyMilk bindings](https://www.github.com/milk-org/pymilk)

## Contents

Module includes key frameworks :

- **image streams** : low-latency shared memory streams
- **processinfo** : process management and control
- **function parameter structure (FPS)** : reading/writing function parameters. See [FPS Standalone and CMD Modes](doc/FPS_Standalone_CMD_Modes.md) for implementation details.

## Download

```bash
git clone https://github.com/milk-org/milk.git
cd milk
```

## Compile

Check required packages in Dockerfile.

### Full build (default)

Builds everything: interactive CLI, TUI, and all standalone fpsexec programs.

**Requires:** cfitsio, readline, ncurses, bison, flex

```bash
mkdir _build && cd _build
cmake ..
make
sudo make install
```

### Standalone-only build (no CLI)

Builds only core libraries and `milk-fpsexec-*` standalone programs.
No interactive CLI, no TUI. Ideal for embedded/headless deployments.

**Requires:** cfitsio only (no readline, ncurses, bison, or flex)

```bash
mkdir _build && cd _build
cmake .. -DUSE_CLI=OFF
make
sudo make install
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_CLI` | ON | Build interactive CLI (`milk-cli`), TUI, scripts |
| `USE_NCURSES` | ON | Enable ncurses TUI (`milk-fpsCTRL`, `streamCTRL`) |
| `USE_READLINE` | ON | Enable readline for CLI input |
| `USE_GSL` | ON | Enable GSL for plugins (`linopt_imtools`) |
| `USE_CUDA` | OFF | Enable CUDA GPU acceleration |

### Build with Python module

```bash
./compile.sh $PWD/local
```

Set environment variables (.bashrc or equivalent):
- MILK_ROOT: Source code directory, for example "/home/coldpenguin/src/milk"
- MILK_INSTALLDIR: Installation directory, for example "/usr/local/milk"
- MILK_SHM_DIR: Shared memory directory, for exmaple "/milk/shm"


Check installation (from any directory) :
```bash
milk-check
```




## Interactive tutorial

Pre-requisites: tmux, nnn
```bash
milk-tutorial
```

## Adding plugins

Compile with cacao plugins:

```bash
./fetch_cacao_dev.sh
./compile.sh $PWD/local
```
Compile with coffee plugins:

```bash
./fetch_coffee_dev.sh
./compile.sh $PWD/local
```
