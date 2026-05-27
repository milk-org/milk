---
description: Set up a development environment from scratch
---

# Set Up Development Environment

Use this workflow for first-time setup of a milk
development environment.

## 1. Install System Dependencies

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y \
  gcc g++ cmake make \
  libcfitsio-dev libncurses-dev \
  libopenblas-dev \
  git tmux \
  pkg-config
```

### Fedora / RHEL

```bash
sudo dnf install -y \
  gcc gcc-c++ cmake make \
  cfitsio-devel ncurses-devel \
  openblas-devel \
  git tmux \
  pkg-config
```

### macOS (Homebrew)

```bash
brew install cmake cfitsio ncurses openblas \
  git tmux pkg-config
```

## 2. Clone the Repository

```bash
cd ~/src
git clone https://github.com/milk-org/milk.git
cd milk
```

## 3. Initialize Submodules

```bash
git submodule update --init --recursive
```

This initializes `ImageStreamIO` and other
submodule dependencies.

## 4. Checkout the Development Branch

```bash
git checkout framework-dev
```

> [!CAUTION]
> Never work directly on `dev` or `main`.
> Create feature branches from `framework-dev`.

## 5. Configure and Build

```bash
mkdir _build && cd _build
cmake .. -DCMAKE_INSTALL_PREFIX=../local
make -j$(nproc)
make install
```

### Optional Build Flags

| Flag                       | Purpose              |
| -------------------------- | -------------------- |
| `-DUSE_STATIC_LTO=ON`      | Static LTO build     |
| `-DVEC_REPORT=ON`          | Vectorization report |
| `-DCMAKE_BUILD_TYPE=Debug` | Debug symbols        |
| `-DUSE_CFITSIO=OFF`        | Build without FITS   |

## 6. Source the Environment

Add to your shell profile:

```bash
source ~/src/milk/local/bin/milk-setup.bash
```

Verify the installation:

```bash
milk-cli -i
```

## 7. Run Tests

```bash
cd ~/src/milk/_build
ctest --output-on-failure
```

## 8. Verify the CLI

```bash
echo "m?" | milk-cli 2>/dev/null
echo "exitCLI" | milk-cli 2>/dev/null
```

## 9. Set Up Git Worktrees (Optional)

For working on multiple features simultaneously:

```bash
cd ~/src/milk
git worktree add ../milk-docs \
  -b docs/current-task framework-dev
git worktree add ../milk-feat \
  -b feat/current-task framework-dev
```

Each worktree needs its own `_build` directory.

## 10. Read the Onboarding Docs

- [`AGENTS.md`](../../AGENTS.md) — if using AI
  coding agents
- [`CONTRIBUTING.md`](../../CONTRIBUTING.md) —
  contribution guidelines
- [`docs/code_assist.md`](../../docs/code_assist.md)
  — rules and workflows index
