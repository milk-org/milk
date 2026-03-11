# Installation

See also: [Build Tiers](build_tiers.md) ·
[FAQ & Troubleshooting](../faq.md) ·
[CLI Overview](../cli/CLI_Overview.md) ·
[Programmer's Guide](../programmers_guide.md)

---

## 1. Download and install milk

> [!WARNING]
> This page describes installation of the core package milk.
> If you install an application package (cacao or coffee),
> replace "milk" with "cacao" in these instructions.

For download, build commands, and CMake options, see the
[Quick Start section in README.md](../../README.md#download).

For configuring minimal or partial builds (engine-only, core
without cfitsio, etc.), see [Build Tiers](build_tiers.md).

The sections below cover post-installation setup, dependencies,
and optional configuration not covered in the README.


## 2. Post-installation

You may need to add `/usr/local/lib` to the linker search
path:

```bash
$ echo "/usr/local/lib" > usrlocal.conf
$ sudo mv usrlocal.conf /etc/ld.so.conf.d/
$ sudo ldconfig -v
```


## 3. tmpfs (optional)

Create a tmpfs disk for high-performance shared-memory I/O:

```bash
$ echo "tmpfs /milk/shm tmpfs rw,nosuid,nodev" | sudo tee -a /etc/fstab
$ sudo mkdir -p /milk/shm
$ sudo mount /milk/shm
```

---

## 4. Dependencies

### 4.1. Required (full build)

| Dependency | Purpose |
|------------|---------|
| **gcc** | C compiler (C17) |
| **cmake** ≥ 3.10 | Build system |
| **pkg-config** | Locates installed libraries |
| **pthreads** | POSIX threading (provided by libc) |

<details>
<summary><b>Optional dependencies</b></summary>

### 4.2. Optional

| Dependency | CMake option | Default | Purpose |
|------------|-------------|---------|---------|
| **cfitsio** | `USE_CFITSIO` | ON | FITS file I/O (`COREMOD_iofits`) |
| **readline** | `USE_READLINE` | ON | Interactive CLI line editing |
| **ncurses** | `USE_NCURSES` | ON | TUI screens (fpsCTRL, procCTRL, streamCTRL) |
| **GSL** | `USE_GSL` | ON | Numerical routines in plugins |
| **fftw3** | — | auto | FFT (used by some plugins) |
| **openMP** | — | auto | Parallel processing |

**💡 Tip:** For a minimal POSIX-only build, all optional
dependencies can be skipped.
See [Build Tiers](build_tiers.md) for details.

</details>

<details>
<summary><b>Package installation commands</b></summary>

**Ubuntu / Debian:**
```bash
$ sudo apt-get install \
    libcfitsio-dev libreadline-dev libncurses5-dev \
    libfftw3-dev libgsl-dev
```

**CentOS / RHEL / Fedora:**
```bash
$ sudo yum install \
    cfitsio-devel readline-devel ncurses-devel \
    fftw3-devel gsl-devel
```

</details>

<details>
<summary><b>cfitsio from source (alternative)</b></summary>

If your distribution does not package cfitsio, install it
from source:

1. Download from [HEASARC](https://heasarc.gsfc.nasa.gov/fitsio/)
2. Build and install:

```bash
$ tar xzf cfitsio-*.tar.gz
$ cd cfitsio-*
$ ./configure --prefix=/usr/local
$ make
$ sudo make install
```

</details>

<details>
<summary><b>GPU acceleration (optional)</b></summary>

For GPU-accelerated linear algebra:

- Install **NVIDIA driver** and **CUDA toolkit**
- Install **MAGMA**

If cmake reports "No package 'magma' found", add the MAGMA
pkg-config path to your shell profile:

```bash
export PKG_CONFIG_PATH=/usr/local/magma/lib/pkgconfig
```

</details>

---

<details>
<summary><b>Running multiple versions side by side</b></summary>

**⚠️ Warning:** Untested — may require tweaking.

Install independent versions by specifying different prefixes:

```bash
$ cd ~/src
$ git clone --recursive https://github.com/milk-org/milk milk-1
$ git clone --recursive https://github.com/milk-org/milk milk-2
```

```bash
$ cd ~/src/milk-1/_build
$ cmake -DCMAKE_INSTALL_PREFIX=/usr/local/milk-1 ..
$ sudo make install

$ cd ~/src/milk-2/_build
$ cmake -DCMAKE_INSTALL_PREFIX=/usr/local/milk-2 ..
$ sudo make install
```

Make version 1 the system default:

```bash
$ sudo ln -s /usr/local/milk-1 /usr/local/milk
```

Run version 2 explicitly:

```bash
$ LD_LIBRARY_PATH=/usr/local/milk-2/lib \
  PATH=/usr/local/milk-2/bin \
  milk-cli
```

Each version can also use its own shared-memory directory:

```bash
$ MILK_SHM_DIR=/milk-2/shm \
  LD_LIBRARY_PATH=/usr/local/milk-2/lib \
  PATH=/usr/local/milk-2/bin \
  milk-cli
```

</details>

---
← [Documentation Index](../index.md)
