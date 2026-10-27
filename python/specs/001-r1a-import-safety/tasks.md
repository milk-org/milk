---
description: "Task list for R1a: Import safety + errors"
---

# Tasks: Import safety + errors

**Input**: [spec.md](./spec.md), [plan.md](./plan.md) · **Roadmap slice**: R1a · **Tests**: py

**Tests**: Required (SC-002, SC-003, SC-005). Run `pytest` from `python/`; the milk-tagged tests need
`MILK_INSTALLDIR`.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: different files, no dependency on an unfinished task
- **[Story]**: US1 (import without a milk environment), US2 (uniform failures)
- Paths are relative to `python/`

---

## Phase 1: Setup

- [x] T001 Run `pytest` from `python/` and record the baseline (which tests pass, which are
  skipped without `MILK_INSTALLDIR`), so that later phases can be compared against it

---

## Phase 2: Foundational (blocks both stories)

**Purpose**: the two new modules every later task imports.

- [x] T002 [P] Create `milk/errors.py`: `MilkError` (kw-only `pipeline`, `compu`, `task`, `path`,
  `cmd`, `stderr`, stored as attributes, non-empty ones appended to `str()` as `name: value`
  lines), then `EnvError`, `ConfigError`, `TaskError` → `TaskCannotRunError` /
  `TaskDidNotSucceedError`, `ComputeUnitError`, `LinkageError`, and `BestEffortError`
  (`failures: list[tuple[str, Exception]]`, message lists `label: error`). Docstrings say why.
- [x] T003 [P] Create `milk/env.py`: `_require_dir(var)` guard (unset, empty or missing path →
  `EnvError(path=…)` naming the variable), `root()`, `installdir()`, `shm_dir()` (requires
  `MILK_SHM_DIR` set, then returns pyMilk's `MILK_SHM_DIR()`), `lib(name)` (missing file →
  `EnvError` naming the library). No caching.
- [x] T004 [P] Add the `MilkError` context and `BestEffortError` message unit checks in
  `tests/trivial/errors_location_test.py` (the file is completed in T024)
- [x] T005 [P] Add `test_env_error_names_missing_var` (`MILK_ROOT` unset → `env.root()` raises
  `EnvError` containing `MILK_ROOT`; also empty and missing-path cases) in
  `tests/trivial/import_safety_test.py` (the file is completed in T011). Add the SC-005 test
  `test_shm_dir` there too: `env.shm_dir()` raises `EnvError` containing `MILK_SHM_DIR` when it is
  unset, empty or not a directory, and returns the directory as a `Path` when valid.

**Checkpoint**: `milk.errors` and `milk.env` import and their unit checks pass.

---

## Phase 3: User Story 1 — Import without a milk environment (P1) 🎯 MVP

**Goal**: every module imports with no `MILK_*` variables and no milk shared libraries; a function
needing them raises `EnvError`.

**Independent Test**: `tests/trivial/import_safety_test.py` passes.

- [x] T006 [US1] `milk/cliwrap.py`: make `_find_build_tags()` use `env.lib("libmilkcommon.so")`,
  wrap the ctypes `OSError` into `EnvError(path=…)` with `raise … from`, and cache it with
  `functools.cache`. Make `_clicore_has_treesitter()` use `env.installdir()`.
- [x] T007 [US1] `milk/cliwrap.py` (after T006): add cached `have_cli()` (returns `False` on
  `EnvError`), `have_treesitter()`, `cli_exec()`; add the PEP 562 module `__getattr__` mapping
  `HAVE_CLI`, `HAVE_TREESITTER`, `MILK_CLI_EXEC`, `MILK_BUILD_TAGS` to them; point `CLI` and
  `CLICommands` at the helpers. Remove all import-time env reads and `dlopen`.
- [x] T008 [P] [US1] `milk/infra/mains_pipeline_mgmt.py`: remove `MILK_SAMPLES_DIR` /
  `CACAO_SAMPLES_DIR`; add `_milk_samples_dir()` and `_cacao_samples_dir()` built on `env.root()`
  and call them where the constants were used.
- [x] T009 [P] [US1] `milk/infra/deploy_tasks.py`: replace both `os.environ["MILK_SHM_DIR"]` reads
  with `env.shm_dir()`.
- [x] T010 [P] [US1] `milk/pipeline.py`: `_linkage_paths` uses `env.shm_dir()` instead of pyMilk's
  `MILK_SHM_DIR`.
- [x] T011 [US1] Complete `tests/trivial/import_safety_test.py` (after T005–T010):
  `test_imports_without_milk_env` — `monkeypatch.delenv` every `MILK_*`, then for each module of
  `pkgutil.walk_packages(milk.__path__, "milk.")` do `monkeypatch.delitem(sys.modules, name,
  raising=False)` + `importlib.import_module(name)`; clear the `cliwrap` caches and assert
  `cliwrap.HAVE_CLI is False`.
- [x] T012 [US1] Run `pytest tests/trivial tests/cli/cliwrap_test.py`; confirm `HAVE_CLI` still
  works for the existing tests in a full environment (acceptance scenario 3).

**Checkpoint**: US1 is testable on its own; exception classes are still in their old places.

---

## Phase 4: User Story 2 — Handle failures uniformly (P1)

**Goal**: one hierarchy under `MilkError`; old classes gone; raise sites carry context.

**Independent Test**: `tests/trivial/errors_location_test.py` passes.

- [x] T013 [P] [US2] `milk/infra/task_models.py`: delete `NoCanTaskError` / `NoSuccessTaskError`
  (the raise sites are in `Pipeline.task_do`, migrated in T015).
- [x] T014 [P] [US2] `milk/infra/toml_manipulation.py`: raise `ConfigError(path=toml_path)`.
- [x] T015 [US2] `milk/pipeline.py` (after T010, T013): raise `ConfigError(path=conf_folder)` /
  `ConfigError(path=toml_path)` for the two not-found cases and `LinkageError(pipeline=short_name)`
  for both linkage cases. Update the imports and both raise sites in `Pipeline.task_do` to
  `TaskCannotRunError` / `TaskDidNotSucceedError`, with `task=` the task type name and
  `pipeline=` the pipeline short name; the cannot-run message keeps `failure_reason`. Untouched
  `ValueError` sites stay.
- [x] T016 [US2] `milk/pipeline.py` `nuke.attempt()` (after T015): no `print`; collect
  `(label, error)` pairs, run every step, then raise `BestEffortError` if any failed. Leave the
  rest of `nuke` for R1c.
- [x] T017 [P] [US2] `milk/session.py`: `_subprocess_complete` reads and decodes stderr and raises
  `ComputeUnitError(compu=…, cmd=proc.args, stderr=…)`, keeping the kill-if-still-running guard.
- [x] T018 [US2] `milk/infra/deploy_tasks.py` (after T009): drop the `FPSExecException` import and
  the three `print`s.
- [x] T019 [US2] `milk/cliwrap.py` (after T007): delete `MilkBuildException` and `CLICrashError`;
  raise `EnvError` (message names `-DUSE_CLI=ON` / `milk-cli`) and `MilkError(cmd=proc.args,
  stderr=stderr)`. `close()` keeps its `except (BrokenPipeError, ValueError): pass`.
- [x] T020 [US2] `milk/infra/mains_pipeline_mgmt.py` (after T008): catch `LinkageError` in
  `milk_pipes_sys_function`; `milk_pipes_nuke_function` catches `BestEffortError` and
  `click.echo`s one warning line per failure, keeping exit status 0.
- [x] T021 [US2] Delete `milk/infra/exceptions.py` (after T013–T020) and grep `milk/` and `tests/`
  for any leftover import of it or of the removed class names; no aliases.
- [x] T022 [P] [US2] Update `tests/pipeline/deploy_pipeline_test.py`: `TaskCannotRunError`, and
  `ConfigError` for both not-found cases, imported from `milk.errors`. Also check both task
  failure paths (`TaskCannotRunError`, `TaskDidNotSucceedError`): the `task` and `pipeline`
  attributes are set, and the cannot-run message contains `failure_reason`.
- [x] T023 [P] [US2] Update `tests/pipeline/system_linkage_test.py`: expect `LinkageError`, imported
  from `milk.errors`.
- [x] T024 [US2] Complete `tests/trivial/errors_location_test.py` (after T021):
  `test_only_errors_module_defines_exceptions` — `ast`-walk every `milk/**/*.py` except
  `errors.py` and assert no `ClassDef` base is named `Exception`, `BaseException`, `*Error` or
  `*Exception`.

**Checkpoint**: US1 and US2 both pass; `infra/exceptions.py` is gone.

---

## Phase 5: Polish and closing (constitution VIII)

- [x] T025 Run the full `pytest` from `python/` (with `MILK_INSTALLDIR` for the milk-tagged
  tests) and compare with the T001 baseline; SC-001 to SC-005 must hold.
- [x] T026 [P] Add a short "Environment and errors" section to `../docs/pipelines.md`: `milk.env`
  is the only `MILK_*` reader, `milk.errors` holds the hierarchy; link design-v7 §4.7, do not copy
  it.
- [x] T027 [P] `specs/pipelines/ROADMAP.md`, row R1a: set Status to `done` and Sub-spec to
  `specs/001-r1a-import-safety/` (edit only those two cells).
- [x] T028 Verify lines ≤ 100 characters and no `os.environ["MILK_` reads outside `milk/env.py`
  (`grep` in `milk/`).

---

## Dependencies & Execution Order

- Phase 1 → Phase 2 (T002, T003 first; T004, T005 can follow in parallel) → Phase 3 → Phase 4 →
  Phase 5.
- Phases 3 and 4 touch the same files (`cliwrap.py`, `pipeline.py`, `deploy_tasks.py`,
  `mains_pipeline_mgmt.py`), so US2 follows US1 rather than running in parallel. Each story is
  still tested on its own.
- File chains: `cliwrap.py` T006 → T007 → T019; `pipeline.py` T010, T013 → T015 → T016;
  `deploy_tasks.py` T009 → T018; `mains_pipeline_mgmt.py` T008 → T020.
- T021 waits for every task that removes a use of the old classes (T013–T020).

### Parallel opportunities

- T002 ∥ T003; T004 ∥ T005.
- US1: T008 ∥ T009 ∥ T010 (different files; independent of T006–T007).
- US2: T013 ∥ T014 ∥ T017; T022 ∥ T023.
- T026 ∥ T027.

## Implementation Strategy

MVP = Phases 1–3 (import safety). Then Phase 4 completes the exception contract and removes the
old module. The slice ships in one pass; nothing in "anticipated" or out-of-scope lists
(CLI top-level error handler, R1b renames, R1c `nuke` rework) is implemented.

---

## Phase 6: Convergence

- [x] T029 `milk/session.py` `_subprocess_complete`: set `compu=self.fpsname` (the FPS name, not
  the subcommand) on the raised `ComputeUnitError`, and add a py test asserting `compu`, `cmd` and
  `stderr` are set and appear in `str(err)` per FR-007 (partial)
- [x] T030 `milk/session.py` `_subprocess_complete`: restore the kill-if-still-running guard
  (`proc.poll() is None` → `kill()` + `wait()`) from the deleted `FPSExecException` per plan:
  per-file changes `session.py` (contradicts)
