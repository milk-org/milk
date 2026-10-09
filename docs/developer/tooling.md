# Developer tooling

This page explains the preferred tools used to **develop, test and debug** `milk`.

Each section explains prerequisites, setup, and usage.

<!-- prettier-ignore -->
!!! warning "Python environment / virtual environments"
    Several tools below ship as Python packages. How you manage them is your choice: `conda`/`mamba`
    environments, `uv`, `apt` packages+`pip`, `virtualenv` + `pip`, etc...

    My preference goes to a combination of `mamba` and `pip`, which is what the
    examples below use.

    Adapt as needed !



## 1. MILK Documentation

<details class="plain" markdown="1" open>
<summary>Expand contents</summary>

The milk v2 documentation is build from markdown files using [mkdocs](https://www.mkdocs.org).

!!! note "Contributing"
    See [here] # TODO add link.

??? tip "Markdown preview in the IDE/editor"
    The documentation is all markdown files, so any markdown visualization item (e.g. vscode's markdown preview) would work to view content.
    However, some plugins & customizations are non-standard and will not render. For this to work, you'll need to look at the actual documentation website.

??? info "Deployment of the documentation"
    A Github action generates the website into the `gh-pages` branch upon pushes to the `framework-dev` branch.

    A Github deployment deploys it to https://milk-org.github.io/milk/ automatically.


For nontrivial documentation work, it is quite useful to do a full website render
locally on your development machine. **Here's how:**

### 1.1 Installing `mkdocs`

!!! question "Prerequisites"
    - A Python environment (see above) in which you can install packages. The examples below use a combo of `mamba` and `pip`.

    TODO: add an entire page dedicated to that...

```bash
# Optional: create an environment
mamba env create -n mkdocsenv python=3.13
mamba activate mkdocsenv

# install mkdocs
mamba install mkdocs

# install extensions used herein -- some not available on conda-forge, so using pip here.
pip install pymdown-extensions \
    mkdocs-material[plugins] \
    mkdocs-glightbox \
    mkdocs-git-revision-date-localized-plugin \
    mkdocs-minify-plugin
```

### 1.2 Build and serve the documentation

```bash
# Build in strict mode: useful to check for stale links.
mamba run -n mkdocsenv mkdocs build --strict
#                      ^^^^^^^^^^^^^^^^^^^^^-- only this part if `mamba activate mkdocsenv` is still active
```

```bash
# Build and serve the website (here, called directly in my mamba session)
mamba run -n mkdocs mkdocs serve
# Access -- follow the link or:
<browser> http://127.0.0.1:8000/
```

!!! success "You now have the local MILK v2 docs in your browser !"
    You can leave `mkdocs serve` running in its terminal, and watch the browser.
    The website will update continuously as you modify `md` files in the `docs/` folder.

</details>

## 2. Building the Cacao documentation

<details class="plain" markdown="1" open>
<summary>Expand contents</summary>

The Cacao v1 documentation is build from markdown files using [jekyll](https://jekyllrb.com/docs/).

<!-- prettier-ignore -->
!!! warning "Temporary section"
    This section will be removed once the Cacao v2 doc is reorganized and moves to mkdocs.

<!-- prettier-ignore -->
!!! question "Prerequisites"
    - Ruby and Bundler installed (`apt install ruby-full build-essential`, or a version manager).
    - The `cacao` sources, available at `plugins/cacao-src` (a symlink to `~/src/cacao`).
    - The `cacao-docs` sources, on the `gh-pages` branch from `github.com/cacao-org/docs.git`.

```bash
# Into the docs repository
cd cacao-docs # or the cloning folder from cacao-org/docs.git
bundle install
bundle exec jekyll serve

# Access
<browser> http://127.0.0.1:4000/docs/
```

!!! success "You now have the local Cacao v1 docs in your browser !"
    You can leave `jekyll serve` running in its terminal, and watch the render in the browser.
    The website will update as you modify `md` files in the `pages/` folder of the documentation repo
    __and hit refresh in the browser__.

</details>

## 3. Formatting and hooks: pre-commit, clang-format, clangd

<details class="plain" markdown="1" open>
<summary>Expand contents</summary>

Formatting is enforced by [pre-commit](https://pre-commit.com) hooks, configured in
`.pre-commit-config.yaml`, and re-checked indentically in CI (`.github/workflows/pre-commit.yaml`).

| Hook summary:                        | Applies to     |
| ------------------------------------ | -------------- |
| `clang-format`                       | C / C++        |
| `black`                              | Python         |
| `cmake-format`                       | CMake          |
| whitespace, end-of-file, yaml, merge | all text files |
| `check-added-large-files`            | all files      |

**We don't worry too much about defining a style guide in plain text**;
what the formatting hooks outputs is the source of truth.

<!-- prettier-ignore -->
!!! warning "Prerequisites"
    - `git` and `pip` available; `pre-commit` itself downloads pinned versions of `clang-format`,
      `black` and `cmake-format` in its own virtualenv, so none of them needs a system install.

### 3.1 Setup

```bash
# Use your existing python environment
pip install pre-commit
# Or pre-commit is now available through apt
sudo apt install pre-commit

# with cwd == $MILK_ROOT
cd $MILK_ROOT
pre-commit install
```

### 3.2 Usage

The typical usage is the one that is forced upon the developer.
Upon running `git commit`, the hook invokes `pre-commit run`.
By default this applies the pre-commit hooks on all the __staged files__.

| Possible outcomes:                     | Then:                                                |
| -------------------------------------- | ---------------------------------------------------- |
| All hooks pass                         | `git commit` succeeds                                |
| Some "automatic" formatting hook fails | The file is corrected; re-stage it and commit again. |
| Some non-correctible hook fails        | Fix the file, re-stage, commit.                      |

`pre-commit` is also pretty versatile to run specific hooks on specific files at anytime,
or reformat the entire repository if we want to change our preferred formatting.

### 3.3 Example: fixing a failing hook

Formatting hooks fix the files in place and fail the commit; nothing is lost, you just stage the
result and commit again:

```console
$ git commit -m "fix: bounds check in image_crop2D"
clang-format.............................................................Failed
- hook id: clang-format
- files were modified by this hook

$ git diff                      # review what the hook changed
$ git add -u && git commit -m "fix: bounds check in image_crop2D"
```

Hooks that do not auto-fix (e.g. `check-yaml`, `check-merge-conflict`) print the offending file and
line: fix it by hand and commit again.

??? tip "Formatting and curated commits"
    Sometimes we prepare curated commits, where we pick only some files, or only some chunks of the current modifications.
    pre-commit is a little annoying with that; since formatted files are corrected with unstaged modifications,
    we now have to cherry pick all the changes we want to stage, again.

    The good workflow here is to format first, cherry-pick the commit contents second:
    ```bash
    git add .             # stage everything
    pre-commit run        # run the hooks on everything
    git restore --stage . # unstage everything, now format-compliant.
    # Now cherry-pick your commit contents.
    ```

### 3.4 Some more checking...

Ideally, we want a 1-1 mapping between the developer's local check with pre-commit and the CI
stack once the commit or PR reaches github. Sometimes that's not practical, plus, there can be no
actual enforcement of what one actually commits!

Style rules (for C/C++) and best practices that `clang-format` cannot enforce (column-aligned parameters, Kernel-Doc, scope
minimization) are covered in [Coding standards](coding_standards.md).

Documentation changes are additionally checked in CI (`.github/workflows/docs-lint.yml`) with
`markdownlint` (`.markdownlint.yml`), a link checker and a typos checker (`_typos.toml`).

</details>

## 4. Testing through a standard python suite: pytest and nox

<details class="plain" markdown="1" open>
<summary>Expand contents</summary>

The Python test suite lives in `python/tests/` and exercises the installed `milk` binaries and
shared libraries through the `milk` and `pyMilk` Python packages. `pytest` runs it against an
existing install; `nox` builds and installs `milk` in isolated environments, then runs `pytest`.

We're trying to move away from scattered `ctest` and bash testing scripts.

The execution versatility provided by `pytest` is convenient for checking regressions, validating new code, debugging one or few tests efficiently, and even deploying temporary, reproducible work sessions.

`nox` takes virtualization a step further: it creates autonomous sessions (a bit like docker but with narrower reach)
in which we can build milk with different options, then run the tests; or use sessions to run completely different things, e.g. a coverage report, etc. The `nox` sessions are configured to be completely isolated from the production build on the same machine.

<!-- prettier-ignore -->
!!! warning "Prerequisites"
    - A Python environment (see the note at the top of this page).
    - `pytest`, `pytest-timeout`, `nox` and `uv` installed (`nox` sessions use the `uv` backend).
    - An installed `pyMilk` checkout (on the nanobind branch, properly fixed to the same ImageStreamIO version),
      with its path exported as `PYMILK_ROOT`. Some tests use `pyMilk` to interact with SHMs and FPSs.
    - A built and locally installed `milk`, with `MILK_INSTALLDIR` pointing to it.
    - `tmux` and a writable `/tmp`: the tests spawn private tmux servers and shared-memory
      directories.

<!-- prettier-ignore -->
!!! warning "TODO"
    `python/noxfile.py` currently hard-codes the `pyMilk` location. It is being changed to read the
    `PYMILK_ROOT` environment variable.

### 4.1 Setup

```bash
export PYMILK_ROOT=~/src/pyMilk
export MILK_INSTALLDIR=<path to your milk install>

cd python
pip install -e "$PYMILK_ROOT" # pyMilk
pip install pytest pytest-timeout nox
```

### 4.2 pytest

`pytest` runs a series of predefined tests, located in `*_test.py` files in the `python/tests` directory.
Running from the `python/` directory:

```bash
cd $MILK_ROOT/python

# Everything
pytest
# Machine readable
pytest -q --color=no --tb=short

# One file, e.g. check that no standalone executable links CLIcore
pytest tests/trivial/build_sanity_test.py

# Verbose
pytest -v tests/trivial/build_sanity_test.py

# Stop at first failure, drop into debugger
pytest -x --pdb

# Filter by test name
pytest -k

# Or any combination thereof !
```

!!! tip "Fixtures"
    `pytest` provides fixtures, a mechanism to setup a testing context, run one or multiple tests, and perform a clean-up.

    Highly reusable fixtures are put on their own, typically in `python/tests/conftestaux/`.

    Some **test-suite wide** fixture perform setup once before any test and cleanup after all the tests are done;
    these are used to isolate the test environment at each run: `MILK_SHM_DIR` and `MILK_PROC_DIR` are
    redirected to `/tmp/milk_shm_dir_pytest`, and a private tmux server is used (`TMUX_TMPDIR`).

    To connect to the same environment by hand in a own shell, mirror that environment first:

    ```bash
    source $MILK_ROOT/python/tests/env.bash
    ```

!!! tip "Using pytest for reproducible testing / development configurations"
    The `tests/mains/` folder is meant to contain **pytest-excluded files** (just omit `_test` in their name).

    A test can be used to setup a reproducible debug environment, either directly or with a fixture:

    === "Plain test"

        ```python
        def test_some_env():
            deploy_something()

            input('Test environment is ready')

            undeploy_something()
        ```

    === "With a fixture"

        ```python
        @pytest.fixture
        def my_fixture():
            setup_something()
            yield None # <-- this is what's passed to the test function.
            teardown_something()

        def test_some_env(my_fixture):
            input('Test environment is ready')
        ```

    and run with
    ```bash
    pytest -s mains/some_file.py -k test_some_env
    # -s provides stdin/stdout for the test and is needed to hold on at the input() call.
    ```
    and you just need to revisit this terminal and hit any key to proceed through `input()` and terminate the test.

    This can be useful in many ways: debugging engineering TUIs behavior when underlying data is created or destroyed;
    tracking a bug; deploying a piece of network code on machine A and experimenting with its sibling on machine B...

    Any terminal can join the pytest milk environment with, as above:
    ```bash
    source $MILK_ROOT/python/tests/env.bash
    ```

    We also have fixtures that deploy, then cleanup an entire Cacao loop from scratch, so this allows deployment,
    toying around with AO, and finally shutting down.

### 4.3 nox

`nox` creates clean, isolated python environment, that can be used for any relevant purpose that requires
an isolated tree and a clean set of packages.

We use it to run the milk tests with various build options, or to run coverage reports.
The default sessions install `milk` and `pyMilk`, configures and builds milk
in a temporary directory, installs it there, and runs `pytest`. The build variants are:

| Default sessions               | `USE_CLI` | `USE_STATIC_LTO` |
| ------------------------------ | --------- | ---------------- |
| `build_and_pytest_nocli_nolto` | OFF       | OFF              |
| `build_and_pytest_nocli_lto`   | OFF       | ON               |
| `build_and_pytest_cli_nolto`   | ON        | OFF              |
| `build_and_pytest_cli_lto`     | ON        | ON               |


```bash
cd $MILK_ROOT/python

nox -l                              # list sessions
nox -s build_and_pytest_nocli_nolto # one session, including those excluded by default

nox -s tests_run_coverage           # run the coverage session, which is not in the defaults; See #5 below.
```

!!! tip
    The nox session setups try to fetch the required packages from the various archives.

    If you're offline, that is bypassed with:
    ```bash
    UV_OFFLINE=1 nox ...
    ```

<!-- prettier-ignore -->
!!! warning
    A full session rebuilds all of milk, even if nox is configured to reuse sessions and only do incremental builds.
    To force a full session rebuild (sometimes needed to purge stale artifacts):

    ```
    nox -N [-s <session>]
    ```

    Rebuilding the 4 default sessions from scratch takes ~5 minutes on a 2020 6-core 4 GHz laptop.
    Running them with no building (zero-diff incremental build).

</details>

## 5. Coverage: gcov and gcovr

<details class="plain" markdown="1" open>
<summary>Expand contents</summary>

Coverage tells you which C code the Python test suite did not execute. It is a validation tool for
the tests, not a pass/fail gate.

<!-- prettier-ignore -->
!!! warning "Prerequisites"
    - `gcc` with `gcov`, and `gcovr` (`pip install gcovr`).
    - Everything from [section 4](#4-python-tests-pytest-nox-and-pymilk).
    - `pyMilk` built with coverage too; otherwise its session-end hook prints
      `_gcov_dump not available` and its counters are not flushed.

### 5.1 Build, test, report

It is possible to run the coverage build directly in the main shell environment (build with coverage flags,
install, run the tests, etc), but inconvenient due to managing the gcc tracking and counter files.

We packaged all of it in a nox session, that produces two html reports, for python and for C/C++.
```bash
nox -s tests_run_coverage           # run the coverage session, which is not in the defaults; See #5 below.
```

### 5.2 Reading the report

!!! warning
    Fix in progress, nox session is broken.

The python coverage report:
```
<browser> $MILK_ROOT/cov_py/index.html
```
and the C coverage report:
```
<browser> $MILK_ROOT/cov_c/index.html
```

<!-- prettier-ignore -->
!!! warning "TODO"
    The `tests_run_coverage` nox session runs the same `gcovr` command, but its CMake configuration
    does not yet pass `-DCMAKE_BUILD_TYPE=Coverage`, so it currently yields empty reports.

</details>

## 6. Dev installs without sudo

<details class="plain" markdown="1" open>
<summary>Expand contents</summary>

Building and testing never requires root: install into a staging directory under `_build/`, and point
your environment at it.

<!-- prettier-ignore -->
!!! warning "Prerequisites"
    - A configured and built `_build` directory (`./compile.sh`, or `cmake` and `cmake --build`).
    - Do **not** run `make install` or `cmake --install` without `--prefix`: it targets the system
      prefix (`/usr/local`) and would need root.
    - Do **not** run `milk-setup-caps`: it needs root to set capabilities.

### 6.1 Install and use

```bash
cd _build
cmake --install . --prefix _install

export MILK_INSTALLDIR="$PWD/_install"
export PATH="$MILK_INSTALLDIR/bin:$PATH"
export LD_LIBRARY_PATH="$MILK_INSTALLDIR/lib:$LD_LIBRARY_PATH"

milk-check        # verifies MILK_ROOT, MILK_INSTALLDIR, MILK_SHM_DIR and PATH
```

To remove it: `rm -rf _build/_install`.

<!-- prettier-ignore -->
!!! note
    Alternatively, configure with `-DCMAKE_INSTALL_PREFIX=$PWD/_install`. The libraries' `RPATH` then
    points to the install, at the cost of the layout becoming `_install/milk-<version>/`.

### 6.2 Global install without sudo

Follow the process for installing milk system-wide: the default recommendation is an install under `/usr/local/milk`,
which is a symbolic link to `/usr/local/milk-<version>`.

By changing ownership of that folder:
```bash
sudo chown -R <user>:<user> /usr/local/milk-<version>
```
you can now `make install`/`ninja install` milk without sudo permissions. The only bit that is missing and actually requires `sudo` is granting root capability to `milk-makecsetandrt`.

!!! warning
    This trick is useful for development when you build, install, and remove the global install very often.

    On a security note, it places non-root sanctionned code in `/usr/local`, which is non-critical but also not expected.

</details>

## 7. Debugging a crashing fpsexec: gdb and valgrind

<details class="plain" markdown="1" open>
<summary>Expand contents</summary>

For attaching to running processes, tmux log inspection and core dumps, see
[Debugging](../operations/debugging.md). This section adds the development workflow around a
standalone executable (see [FPS standalone modes](../arch/FPS_Standalone_CMD_Modes.md)).

<!-- prettier-ignore -->
!!! warning "Prerequisites"
    - `gdb` and `valgrind` installed (`apt install gdb valgrind`).
    - A **Debug** build (`MILK_BUILD_TYPE=Debug ./compile.sh`, or `-DCMAKE_BUILD_TYPE=Debug`): in
      the default `Release` build (`-Ofast`) backtraces are missing frames and variables.
    - Reproduce in the foreground: do not use `-tmux`, which runs the process inside a tmux session
      where `gdb`/`valgrind` do not wrap it. The `exec` command (auto-init, set arguments, run) does
      this in one process.
    - A stale shared-memory state can mask or fake a crash: delete leftover
      `$MILK_SHM_DIR/fps.<name>.*` before retrying.

### 7.1 gdb

```bash
gdb --args milk-fpsexec-mymodule exec <arg1> <arg2>
(gdb) run
# ... crash ...
(gdb) bt full                    # backtrace with local variables
(gdb) frame 2
(gdb) print img.im->md->size[0]
(gdb) info threads
(gdb) thread apply all bt
```

For an already-running loop, attach instead (`gdb -p <PID>`, from `milk-procinfo-list`).

### 7.2 valgrind

`valgrind` is the tool of choice for invalid reads and writes, uninitialized values and leaks, at
a 10 to 50 times slowdown. Use few iterations.

```bash
# The interactive CLI, with a ready-made wrapper: output goes to milk-cli.memcheck.log
milk-debug
tail -f milk-cli.memcheck.log

# A standalone executable
valgrind --leak-check=full --track-origins=yes --max-stackframe=4442392 \
    milk-fpsexec-mymodule exec <arg1> <arg2>
```

In the output, read the first error of each block first: the "Invalid write of size 4" stack is the
faulting access, the "Address ... is N bytes after a block of size M alloc'd" stack is where the
buffer came from.

<!-- prettier-ignore -->
!!! warning "TODO"
    `milk-debug` passes `--suppressions=$MILK_ROOT/milk-cli.memcheck.supp`, a file that is not in the
    repository. Valgrind may refuse to start; add the file or drop the option.

</details>

## 8. License compliance: REUSE

<details class="plain" markdown="1" open>
<summary>Expand contents</summary>

The project follows the [REUSE](https://reuse.software) specification: every file carries its
copyright and license as SPDX tags, and the license texts are in `LICENSES/`.

<!-- prettier-ignore -->
!!! warning "Prerequisites"
    - `reuse` installed (`pip install reuse`).
    - The license text of every identifier you use must be in `LICENSES/` (currently
      `LGPL-3.0-or-later`, `GPL-3.0-or-later` and `MIT`). Fetch a missing one with
      `reuse download <identifier>`.
    - Run from the repository root.

### 8.1 Usage

```bash
# Validation: reports files lacking copyright/license info, unused or missing license texts
reuse lint

# Add the standard header to a new file
reuse annotate --copyright "Olivier Guyon et al" --year 2026 \
    --license LGPL-3.0-or-later src/my_module/my_func.c
```

The result matches the existing sources:

```c
// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later
```

Run `reuse lint` before opening a PR that adds files. For files that cannot hold a header
(binaries, FITS files), declare them in a `REUSE.toml` at the repository root.

<!-- prettier-ignore -->
!!! warning "TODO"
    There is no `REUSE.toml` yet, so `reuse lint` will flag such files. Add it, and add `reuse lint`
    to the pre-commit configuration or CI.

</details>

## 9. Where to go next

- [Debugging](../operations/debugging.md): running processes, tmux, core dumps.
- [Performance tuning](../operations/performance.md) and
  [PGO and LTO](../operations/pgo.md): `perf`, `milk-perfbench`, optimized builds.
- [Code assist tools](code_assist.md): agent rules, skills and workflows.
- [Working with git](WorkingWithGit.md): branches, PRs and worktrees.
