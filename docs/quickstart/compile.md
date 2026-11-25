---
render_macros: true
---

# Installation

See also: [Build Tiers](build_tiers.md) · [FAQ & Troubleshooting](faq.md) ·
[CLI Overview](CLI_Overview.md) · [Programmer's Guide](../arch/programmers_guide.md)

______________________________________________________________________

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
        sudo apt-get install gcc cmake pkg-config libcfitsio-dev libreadline-dev libfftw3-dev libgsl-dev tmux
        ```

    === "CentOS / RHEL / Fedora"

        ```bash
        sudo yum install gcc cmake pkg-config cfitsio-devel readline-devel fftw3-devel gsl-devel tmux
        ```

??? note "Optional dependencies"

    ### Optional

    | Dependency   | CMake option   | Default | Purpose                          |
    | ------------ | -------------- | ------- | -------------------------------- |
    | **cfitsio**  | `USE_CFITSIO`  | ON      | FITS file I/O (`COREMOD_iofits`) |
    | **readline** | `USE_READLINE` | ON      | Interactive CLI line editing     |
    | **GSL**      | `USE_GSL`      | ON      | Numerical routines in plugins    |
    | **fftw3**    | —              | auto    | FFT (used by some plugins)       |
    | **openMP**   | —              | auto    | Parallel processing              |

    ### Somewhat optional:

    | Dependency          | Default | Purpose                                                     |
    | ------------------- | ------- | ----------------------------------------------------------- |
    | **MKL** (`mkl-sdl`) | auto    | Intel linear algebra; takes priority over OpenBLAS if found |
    | **OpenBLAS**        | auto    | Linear algebra (plugins); preferred over standalone LAPACKE |
    | **Lapacke**         | auto    | Linear algebra                                              |
    | **CUDA**            | runtime | Nvidia GPU computation                                      |
    | **MAGMA**           | runtime | Linear algebra library over CUDA                            |
    | **tmux**            | runtime | Persistent shells for FPS compute units                     |

    **lapacke** is often provided by either MKL or OpenBLAS.
    If using neither, standalone lapacke is needed: `apt install liblapacke-dev` or `yum install lapacke-devel`.

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
./fetch_cacao_dev.sh  # (1)!
```

1. Clones the cacao plugin into `plugins/cacao-src`, so that the build below includes it. moka
   and the [moka Tutorial](moka_tutorial.md) need cacao.

**Build:**

```bash
mkdir _build && cd _build
cmake ..
make -j$(nproc)
sudo make install # The default install path is /usr/local/milk-<version>
```

!!! note "Installing elsewhere"

    CMake appends `milk-<version>` to `CMAKE_INSTALL_PREFIX` (unless the prefix already ends with
    it) and does not read `MILK_INSTALLDIR`. Set both:

    ```bash
    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/local  # (1)!
    export MILK_INSTALLDIR=$HOME/local/milk-<version>
    ```

    1. Installs to `$HOME/local/milk-<version>`.

    Several versions side by side: [Multiversion](../developer/multiversion.md).

For minimal or partial builds (engine-only, core without cfitsio, etc.), see
[Build Tiers](build_tiers.md).

The sections below cover post-installation setup and optional configuration.

______________________________________________________________________

## 2. Environment variables

Set in `.bashrc` or equivalent:

| Variable          | Purpose                                   |
| ----------------- | ----------------------------------------- |
| `MILK_ROOT`       | Source code directory                     |
| `MILK_INSTALLDIR` | Installation directory                    |
| `PATH`            | milk and cacao executables                |
| `MILK_SHM_DIR`    | Shared memory directory (streams and FPS) |
| `MILK_PROC_DIR`   | Process info directory: must be set       |

```bash title="~/.bashrc"
export MILK_ROOT=$HOME/src/milk
export MILK_INSTALLDIR=/usr/local/milk
export PATH=$MILK_INSTALLDIR/bin:$PATH
export MILK_SHM_DIR=/milk/shm
export MILK_PROC_DIR=$MILK_SHM_DIR
```

`MILK_PROC_DIR` does not default to `MILK_SHM_DIR`: when it is unset, processinfo uses the
compiled-in `/milk/shm`, whatever `MILK_SHM_DIR` is. With a non-default `MILK_SHM_DIR`, processes
and TUIs then no longer see each other.

!!! warning "MILK_INSTALLDIR"

    `MILK_INSTALLDIR` should point to the installation.
    Generally we set a symbolic link once and for all:

    ```bash
    sudo ln -sfn "/usr/local/milk-<version>" /usr/local/milk
    ```

    For a non-default install location, see the note _Installing elsewhere_ in
    [§1](#1-download-and-install-milk).

______________________________________________________________________

## 3. tmpfs (optional but quite necessary to get performance)

Create a tmpfs disk for high-performance shared-memory I/O:

```bash
$ echo "tmpfs /milk/shm tmpfs rw,nosuid,nodev" | sudo tee -a /etc/fstab
$ sudo mkdir -p /milk/shm
$ sudo mount /milk/shm
```

______________________________________________________________________

## 4. Python packages

After milk is installed and the [environment variables](#2-environment-variables) are set:

```bash
PYMILK_ENGINE=SYSTEM pip install "$MILK_ROOT/python/pyMilk[vis]"  # (1)!
pip install $MILK_ROOT/plugins/cacao-src                           # (2)!
pip install $MILK_ROOT/python/                                     # (3)!
```

1. [pyMilk](../guide/pyMilk.md#1-installation): streams, FPS and processinfo from Python.
   Reinstall it after reinstalling milk. The `[vis]` extra brings the viewers (`shmImshow`).
2. pycacao, from the cacao plugin: needed by the moka calibration plans and `[dmhint]`.
3. [moka](../moka/moka.md): `mokainit`, `moka`, `mokashow`.

Add `-e` for a developer (editable) install.

For the pyMilk engine options, requirements and usage, see the full [pyMilk](../guide/pyMilk.md) page.

______________________________________________________________________

## 5. Further reading:

- See [Build Tiers](build_tiers.md)
- See [Running multiple versions side by side](../developer/multiversion.md).

______________________________________________________________________

← [Documentation Index](../index.md)
