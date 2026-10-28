---
render_macros: true
---

# Installation

See also: [Build Tiers](build_tiers.md) · [FAQ & Troubleshooting](faq.md) ·
[CLI Overview](CLI_Overview.md) · [Programmer's Guide](../arch/programmers_guide.md)

---

## 1. Download and install milk

<!-- prettier-ignore -->
!!! info
    This page describes installation of the core package milk.
    If you install an application package (cacao or coffee),
    replace "milk" with "cacao" in these instructions.

??? abstract "Install dependencies"

    "Package installation commands (quickstart)"

    === "Ubuntu / Debian"
        ```bash
        sudo apt-get install gcc cmake pkg-config libcfitsio-dev libreadline-dev libncurses-dev libfftw3-dev libopenblas-dev liblapacke-dev
        ```

    === "CentOS / RHEL / Fedora"
        ```bash
        sudo yum install gcc cmake pkg-config cfitsio-devel readline-devel ncurses-devel fftw3-devel openblas-devel lapacke-devel
        ```

??? note "Optional dependencies"

    ### Core & CLI Options

    | Dependency      | CMake option     | Default | Purpose                                          |
    | --------------- | ---------------- | ------- | ------------------------------------------------ |
    | **cfitsio**     | `USE_CFITSIO`    | ON      | FITS file I/O (`COREMOD_iofits`)                 |
    | **readline**    | `USE_READLINE`   | ON      | Interactive CLI line editing                     |
    | **ncurses**     | `USE_NCURSES`    | ON      | TUI dashboards (`milk-CTRL`, `milk-fpsCTRL`)     |
    | **tree-sitter** | `USE_TREESITTER` | ON      | Bundled CLI syntax highlighting and formatter    |
    | **fftw3**       | —                | auto    | FFT routines (used by `fft` plugin)              |
    | **openMP**      | —                | auto    | Parallel loop execution                          |

    ### Linear Algebra & GPU Options

    | Dependency          | CMake option   | Default | Purpose                                                     |
    | ------------------- | -------------- | ------- | ----------------------------------------------------------- |
    | **OpenBLAS**        | `USE_OPENBLAS` | ON      | Linear algebra on CPU (auto-disabled if not found)          |
    | **LAPACKE**         | `USE_LAPACKE`  | ON      | Linear algebra routines (auto-disabled if not found)        |
    | **MKL** (`mkl-sdl`) | `USE_MKL`      | OFF     | Intel linear algebra; takes priority over OpenBLAS when ON  |
    | **CUDA**            | `USE_CUDA`     | OFF     | NVIDIA GPU computation                                      |
    | **MAGMA**           | `USE_MAGMA`    | OFF     | GPU linear algebra library over CUDA                        |

    **LAPACKE** is often provided by either MKL or OpenBLAS.
    If using neither, standalone LAPACKE can be installed (`apt install liblapacke-dev` or `yum install lapacke-devel`).


!!! tip
    For a minimal POSIX-only build, all optional dependencies can be skipped. See
    [Build Tiers](build_tiers.md) for details.

??? abstract "GPU acceleration (optional)"

    For GPU-accelerated linear algebra:

    - Install **NVIDIA driver** and **CUDA toolkit**
    - Install **MAGMA**

    If cmake reports "No package 'magma' found", add the MAGMA pkg-config path to your shell
    profile:

    ```bash title="~/.bashrc"
    export PKG_CONFIG_PATH=/usr/local/magma/lib/pkgconfig
    ```

**Clone:**

```bash
git clone --recursive https://github.com/milk-org/milk.git
cd milk
```

**Build:**

```bash
mkdir _build && cd _build
cmake ..
make -j$(nproc)
sudo make install # The default install path is /usr/local/milk-<version>
```

For minimal or partial builds (engine-only, core without cfitsio, etc.), see
[Build Tiers](build_tiers.md).

The sections below cover post-installation setup and optional configuration.

---

## 2. Environment variables

Set in `.bashrc` or equivalent:

| Variable          | Example               | Purpose                 |
| ----------------- | --------------------- | ----------------------- |
| `MILK_ROOT`       | `/home/user/src/milk` | Source code directory   |
| `MILK_INSTALLDIR` | `/usr/local/milk`     | Installation directory  |
| `MILK_SHM_DIR`    | `/milk/shm`           | Shared memory directory |

!!! warning "MILK_INSTALLDIR"
    `MILK_INSTALLDIR` should point to the installation.
    Generally we set a symbolic link once and for all:
    ```bash
    sudo ln -sfn "/usr/local/milk-<version>" /usr/local/milk
    ```

---

## 3. tmpfs (optional but quite necessary to get performance)

Create a tmpfs disk for high-performance shared-memory I/O:

```bash
$ echo "tmpfs /milk/shm tmpfs rw,nosuid,nodev" | sudo tee -a /etc/fstab
$ sudo mkdir -p /milk/shm
$ sudo mount /milk/shm
```

---

## 4. Further reading:

- See [Build Tiers](build_tiers.md)
- See [Running multiple versions side by side](../developer/multiversion.md).

---

← [Documentation Index](../index.md)
