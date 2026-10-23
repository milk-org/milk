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
cd ~/src/milk-1/_build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/milk-1 ..
sudo make install

cd ~/src/milk-2/_build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/milk-2 ..
sudo make install
```

And e.g. make `milk-1` your system default:
```bash
sudo ln -s /usr/local/milk-1 /usr/local/milk
```

??? note "Legacy linker configuration"
    If you find references to using milk with configuring `LD_LIBRARY_PATH` and using `ldconfig`, these
    are obsolete and should be unnecessary (and conflicting!) with the current build/install workflow.

## 2. Clean environment

???+ warning "The environment conundrum"
    milk relies on environment variables and external tools.
    Expect:

    - tmux naming collisions
    - `MILK_SHM_DIR` and `MILK_PROC_DIR` collisions
    - `MILK_ROOT` and `MILK_INSTALLDIR` collisions in peripheral tools.
    - A lot more trouble ! Even if we override these environment variables in a shell, the following will happen:
    ```text
    Humanic shell:
        override MILK_SHM_DIR
        call fpsexec-<x> -tmux runstart (like this, or through milk pipelines)
     v-------
    fpsexec in humanic shell:
        knows the correct overriden MILK_SHM_DIR
        dispatches fpsexec-<x> runstart (no tmux) into a tmux session
     v-------
    tmux's shell
        Initializes bash and set MILK_SHM_DIR to the default !
        executes <fullpath>/fpsexec-<x> runstart
                The exec, linkage paths are correct, but the environment is the default !
    ```

    At the end of this, the execution environment has NOT received the environment override.

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
to which we add a tmux override, so that another tmux server takes over entirely:
```bash
export TMUX_TMPDIR=/tmp/milk-my-second-devel-version/
mkdir -p ${TMUX_TMPDIR}
unset TMUX
```

Now with all of the above set, the new `tmux` server receives the correct environment, including
the milk variables overriding the defaults.
When `bash` loads in the tmux sessions, it does not override the preferred environment.

!!! note "milk-env-switch ?"
    We still need to switch environments in each top-level terminal interacted with by a human,
    or by tooling that will interact with the tmux.
    We should propose a `milk-env-switch #id` that does that and can be one-lined, or combined with
    a virtualenv of some kind.
