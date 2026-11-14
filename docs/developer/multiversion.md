# Running multiple versions side by side

You can run independent versions of milk side-by-side on the same machine, but there's some environment overlap that you must be careful about.

## 1. Installation and environment

Clone two different copies (or, use [git worktrees](./WorkingWithGit.md#3-parallel-development-using-git-worktrees)):

```bash
cd ~/src
git clone --recursive https://github.com/milk-org/milk milk-1
git clone --recursive https://github.com/milk-org/milk milk-2
```

Then compile and install:

```bash
mkdir -p ~/src/milk-1/_build ~/src/milk-2/_build
cd ~/src/milk-1/_build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/milk-1 ..
sudo make install

cd ~/src/milk-2/_build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/milk-2 ..
sudo make install
```

And e.g. make `milk-1` your system default:

```bash
sudo ln -sfn "/usr/local/milk-1/milk-<version>" /usr/local/milk
```

??? note "Legacy linker configuration"
    If you find references to using milk with configuring `LD_LIBRARY_PATH` and using `ldconfig`, these
    are obsolete and should be unnecessary (and conflicting!) with the current build/install workflow.

## 2. Clean environment

???+ warning "The environment conundrum"
    milk relies on environment variables and external tools.
    Expect:

    - `MILK_SHM_DIR` and `MILK_PROC_DIR` collisions
    - `MILK_ROOT` and `MILK_INSTALLDIR` collisions in peripheral tools.

    A unit runs with the environment of the shell that ran its `fpsinit`, saved in
    `NAME.fps.d/env`, whoever starts it later (`milk-fps-info --env NAME` prints it). Set the
    overrides in that shell, and run `fpsinit` again after changing them.

In the installation, we set for the shell:

| Variable          | Example               | Purpose                 |
| ----------------- | --------------------- | ----------------------- |
| `MILK_ROOT`       | `/home/user/src/milk` | Source code directory   |
| `MILK_INSTALLDIR` | `/usr/local/milk`     | Installation directory  |
| `MILK_SHM_DIR`    | `/milk/shm`           | Shared memory directory |

The trick is to not override a variable that may be passed to the shell; this is done by

```bash
export MILK_SHM_DIR="${MILK_SHM_DIR:-/milk/shm}"
export MILK_INSTALLDIR="${MILK_INSTALLDIR:-/usr/local/milk}"
export MILK_ROOT="${MILK_ROOT:-/home/user/src/milk}"
```

A unit's `fpsinit` in that shell saves these values for every later start of the unit.

## 3. Switching Environments (`milk-env.sh`)

To switch your current shell between `framework-dev` (`fdev`) and `dev` installations (updating
`MILK_ROOT`, `MILK_INSTALLDIR`, `PKG_CONFIG_PATH`, `PYTHONPATH`, and `PATH` in place), source
`milk-env.sh`:

```bash
source milk-env.sh fdev     # switch to framework-dev environment
source milk-env.sh dev      # switch to legacy dev environment
source milk-env.sh status   # display active MILK_* variables and binary paths
```
