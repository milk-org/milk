# Implementation Plan: Import safety + errors

**Spec**: [spec.md](./spec.md) · **Roadmap slice**: R1a · **Tests**: py

**Design**: [design-v7.md](../pipelines/design-v7.md) §4.5, §4.7

## Technical Context

- Python ≥3.11, no new dependencies. Package root `milk/`, tests in `tests/` (`pytest` from
  `python/`).
- `MILK_*` reads that exist today: `MILK_ROOT` (`infra/mains_pipeline_mgmt.py`, at import),
  `MILK_INSTALLDIR` (`cliwrap.py`, at import, plus `dlopen` of `libmilkcommon.so`), and
  `MILK_SHM_DIR` (`infra/deploy_tasks.py` ×2 via `os.environ`; `pipeline.py` via pyMilk).
- pyMilk's `MILK_SHM_DIR()` is a lazy `os.environ["MILK_SHM_DIR"]` with no default.
- The `os.environ` copies passed to subprocesses (`pipeline._git`, `CLICommands`) are not `MILK_*`
  reads and stay as they are.

## Decisions (Vincent, 2026-10-27)

1. **`Pipeline.nuke`**: R1a changes only its `attempt()` loop so that it collects failures and
   raises one aggregate error (FR-010). The rest of `nuke` stays for R1c.
2. **`MILK_SHM_DIR` unset**: `milk.env.shm_dir()` raises `EnvError("MILK_SHM_DIR")`. When the
   variable is set, it returns pyMilk's `MILK_SHM_DIR()`.
3. **`deploy_tasks.py` prints**: removed (FR-010; this module is touched).

## New modules

### `milk/errors.py`

```text
MilkError(Exception)          kw-only context: pipeline, compu, task, path, cmd, stderr
├── EnvError
├── ConfigError
├── TaskError
│   ├── TaskCannotRunError
│   └── TaskDidNotSucceedError
├── ComputeUnitError
├── LinkageError
└── BestEffortError           failures: list[tuple[str, Exception]]
```

- `MilkError.__init__(message, *, pipeline=None, compu=None, task=None, path=None, cmd=None,
  stderr=None)` stores every field as an attribute. `__str__` appends the non-empty ones, one per
  line, as `name: value`.
- `BestEffortError` is not listed in §4.7 rule 1. It is the "one aggregate error" that rule 2
  requires. Its message lists each `label: error`, and it chains nothing because there are several
  causes. ⚠ Flag: this is the only class added beyond the design's list.
- `ProvisionError` and `StateError` are not needed by this slice and are not added.

### `milk/env.py`

The single place that reads `MILK_*`. Every function is lazy and has no cache, so tests can
monkeypatch the environment.

| Function | Returns | Raises `EnvError` (context `path=`) when |
|----------|---------|------------------------------------------|
| `root() -> Path` | `$MILK_ROOT` | unset, empty, or not a directory |
| `installdir() -> Path` | `$MILK_INSTALLDIR` | unset, empty, or not a directory |
| `shm_dir() -> Path` | pyMilk `MILK_SHM_DIR()` | `MILK_SHM_DIR` unset, empty, or not a directory |
| `lib(name) -> Path` | `installdir()/lib/<name>` | the file is missing (the message names the library) |

A private `_require_dir(var)` guard performs the unset/empty/missing check (early returns).

## Exception mapping

| Old class (module) | New class | Context set at the raise site | Why |
|--------------------|-----------|-------------------------------|-----|
| `PipelineConfFolderNotFoundException` (infra/exceptions) | `ConfigError` | `path=conf_folder` | The test checks the raise only; `path` tells it apart from the TOML case |
| `PipelineTomlNotFoundException` (infra/exceptions) | `ConfigError` | `path=toml_path` | same |
| `UnknownSystemLinkageException` (infra/exceptions) | `LinkageError` | `pipeline=short_name` | The only catcher (`mains.milk_pipes_sys_function`) handles both linkage classes the same way |
| `SystemLinkageException` (infra/exceptions) | `LinkageError` | `pipeline=short_name` | same |
| `FPSExecException` (infra/exceptions) | `ComputeUnitError` | `compu=fps name, cmd=proc.args, stderr=decoded stderr` | No caller distinguishes it. The proc-reading logic moves into `ComputeSession._subprocess_complete` |
| `NoCanTaskError` (infra/task_models) | `TaskCannotRunError` | `task=type name, pipeline=short_name` | It is a design family; the message keeps `failure_reason` |
| `NoSuccessTaskError` (infra/task_models) | `TaskDidNotSucceedError` | `task=…, pipeline=…` | It is a design family |
| `MilkBuildException` (cliwrap) | `EnvError` | none (the message names `-DUSE_CLI=ON` / `milk-cli`) | A milk build without CLI is a missing part of the environment |
| `CLICrashError` (cliwrap) | `MilkError` (base, no subclass) | `cmd=proc.args, stderr=stderr` | No §4.7 family fits a crash of the `milk-cli` REPL, and no caller or test catches it by type |

No dedicated subclass is kept, because nothing in `milk/` or `tests/` tells two mapped classes
apart by type. The old classes subclassed `FileNotFoundError`, `KeyError` and `RuntimeError`, but
no caller catches those bases, so dropping them changes no behaviour.
`infra/exceptions.py` is deleted, with no aliases.

## Per-file changes

- **`cliwrap.py`**
  - `_find_build_tags()` uses `env.lib("libmilkcommon.so")`, wraps the ctypes `OSError` into
    `EnvError(path=…) from`, and is cached with `functools.cache`.
  - `_clicore_has_treesitter()` uses `env.installdir()`.
  - New cached helpers: `have_cli()` returns `False` on `EnvError` (spec edge case),
    `have_treesitter()` and `cli_exec()`.
  - A module `__getattr__` (PEP 562) maps `HAVE_CLI`, `HAVE_TREESITTER`, `MILK_CLI_EXEC` and
    `MILK_BUILD_TAGS` to those helpers. Existing `from milk.cliwrap import HAVE_CLI, …` imports in
    the tests keep working. This is the only indirection added; it is needed because a plain
    constant cannot be lazy. It uses the lazy-loading exception in
    [constitution IV](../../.specify/memory/constitution.md#iv-lean-and-readable), approved by
    Vincent on 2026-10-27.
  - `CLI` and `CLICommands` call the helpers. Exception classes are removed as mapped above.
  - `close()` keeps its `except (BrokenPipeError, ValueError): pass`, because writing `exit` to a
    REPL that is already dead is the expected outcome, not a failure.
- **`infra/mains_pipeline_mgmt.py`** (CLI entry point, rule 2 exempt)
  - Remove `MILK_SAMPLES_DIR` and `CACAO_SAMPLES_DIR`. Add `_milk_samples_dir()` and
    `_cacao_samples_dir()`, built on `env.root()`, and call them where the constants were used.
  - Catch `LinkageError` in `milk_pipes_sys_function`.
  - `milk_pipes_nuke_function` catches `BestEffortError` and `click.echo`s one warning line per
    failure, keeping today's exit status 0.
- **`pipeline.py`**
  - Raise sites as mapped above. `_linkage_paths` uses `env.shm_dir()` instead of pyMilk's
    `MILK_SHM_DIR`.
  - `nuke.attempt()`: no `print`. It collects `(label, error)` pairs and raises `BestEffortError`
    after all steps if any failed. Everything else in `nuke` is unchanged (R1c).
  - Untouched `ValueError` raise sites (`_has_local_git`, `check_clone_target`,
    `update_local_git`, `task_do`) stay as they are, since FR-007 covers only the code this slice
    touches.
- **`session.py`**: `_subprocess_complete` reads stderr and raises `ComputeUnitError`. It keeps
  today's kill-if-still-running guard.
- **`infra/deploy_tasks.py`**: drop the unused `FPSExecException` import and the three `print`s.
  Both `os.environ["MILK_SHM_DIR"]` reads become `env.shm_dir()`.
- **`infra/task_models.py`**: delete both exception classes.
- **`infra/toml_manipulation.py`**: raise `ConfigError(path=toml_path)`.
- **`infra/exceptions.py`**: delete.

pyMilk calls are left unwrapped everywhere (FR-008); no `try` is added around them.

## Tests (py)

- **Update**:
  - `tests/pipeline/deploy_pipeline_test.py`: `NoCanTaskError` becomes `TaskCannotRunError`, and
    both not-found cases expect `ConfigError`.
  - `tests/pipeline/system_linkage_test.py`: expect `LinkageError`.
  - Both import from `milk.errors`.
- **New `tests/trivial/import_safety_test.py`**:
  - `test_imports_without_milk_env`: `monkeypatch.delenv` every `MILK_*` variable. Then, for each
    module from `pkgutil.walk_packages(milk.__path__, "milk.")`, run
    `monkeypatch.delitem(sys.modules, name, raising=False)` and `importlib.import_module(name)`.
    Also clear the `cliwrap` caches, then assert `cliwrap.HAVE_CLI is False`.
  - `test_env_error_names_missing_var`: with `MILK_ROOT` unset, `env.root()` raises `EnvError`
    and the message contains `MILK_ROOT`.
  - `test_shm_dir` (SC-005): `env.shm_dir()` raises `EnvError` containing `MILK_SHM_DIR` when it
    is unset, empty or not a directory, and returns the directory as a `Path` when valid.
- **New `tests/trivial/errors_location_test.py`**:
  - `test_only_errors_module_defines_exceptions`: `ast`-walk every `milk/**/*.py` except
    `errors.py`, then assert that no `ClassDef` base is named `Exception`, `BaseException`,
    `*Error` or `*Exception`.
  - Small unit checks on `MilkError` context in `str()` and on the `BestEffortError` message.
- **Run**: `pytest` from `python/`. The full suite must stay green; the milk-tagged tests need
  `MILK_INSTALLDIR`.

## Docs & roadmap (closing step)

- `../docs/pipelines.md`: add a short "Environment and errors" section. It says that `milk.env` is
  the only `MILK_*` reader and that `milk.errors` holds the hierarchy, and it links design-v7 §4.7
  instead of copying it.
- `specs/pipelines/ROADMAP.md` R1a: set Status to `done` and Sub-spec to
  `specs/001-r1a-import-safety/`.
