# Feature Specification: Import safety + errors

**Roadmap slice**: R1a ([ROADMAP.md](../pipelines/ROADMAP.md)) · **Tests**: py

**Design**: [design-v7.md](../pipelines/design-v7.md) §4.5, §4.7; defects B3, X8

**Created**: 2026-10-27 · **Status**: Draft

## Overview

Make the `milk` package safe to import without a milk environment, and give it one exception
contract.

Why: some modules cannot be imported without a full milk environment. `mains_pipeline_mgmt` reads
`MILK_ROOT` at import; `cliwrap` reads `MILK_INSTALLDIR` and loads `libmilkcommon.so` at import
(B3). Exception classes are also scattered across modules with no common base, so callers cannot
handle failures uniformly (X8).

## Clarifications

### Session 2026-10-27

- Q: Must `milk` modules import with pyMilk absent, or only with `MILK_*` variables and the
  `dlopen`'d milk libraries absent? → A: pyMilk stays installed and importable. Only `MILK_*`
  variables and the `dlopen`'d milk libraries (`libmilkcommon.so`, `libCLIcore.so`) are absent.
- Q: Does R1a also apply design-v7 §4.7 rule 2 (library code never prints, exits or swallows
  errors) to the code it touches? → A: Yes, in library code of the modules this slice modifies.
  CLI entry points (`milk/infra/mains_pipeline_mgmt.py`) keep their user-facing prints and catches.
- Q: How does a `MilkError` carry structured context? → A: As optional keyword-only attributes on
  `MilkError` (`pipeline`, `compu`, `task`, `path`, `cmd`, `stderr`), with the non-empty ones also
  shown in `str()`.
- Q: Where does `milk.env` get `MILK_SHM_DIR`, and what if it is unset? → A: `milk.env.shm_dir()`
  raises `EnvError` naming `MILK_SHM_DIR` when it is unset, empty or not a directory; otherwise it
  returns pyMilk's `MILK_SHM_DIR()`. pyMilk has no default, and no other `milk` module reads it.

## User Scenarios & Testing

### User Story 1 - Import without a milk environment (Priority: P1)

A developer or CI job imports any `milk` module on a machine with pyMilk installed, but with no
`MILK_*` variables and no `dlopen`'d milk libraries.

**Independent Test**: With every `MILK_*` variable removed, import each package module.

**Acceptance Scenarios**:

1. **Given** no `MILK_*` variables and no `dlopen`'d milk libraries, **When** any package module
   is imported, **Then** the import succeeds.
2. **Given** that same environment, **When** a function needing a variable or library is called,
   **Then** it raises `EnvError` naming what is missing.
3. **Given** a full milk environment, **When** existing tests read `HAVE_CLI` from `cliwrap`,
   **Then** they get the same value as before.

### User Story 2 - Handle failures uniformly (Priority: P1)

A caller catches `MilkError` to handle any package failure, or a specific family to handle one
kind.

**Independent Test**: Walk `milk/` and list every `Exception` subclass definition.

**Acceptance Scenarios**:

1. **Given** the package, **When** its modules are scanned, **Then** only `milk/errors.py` defines
   `Exception` subclasses.
2. **Given** any exception raised by package code touched by this slice, **When** caught as
   `MilkError`, **Then** its context attributes (pipeline, CompU, task, path, subprocess command
   and stderr, as applicable; design-v7 §4.7 rule 3) are set and appear in its message.
3. **Given** a pyMilk call that fails, **When** the error reaches the caller, **Then** it is the
   original pyMilk exception, unwrapped.

### Edge Cases

- A `MILK_*` variable is set but empty, or points to a missing path: treated as missing,
  `EnvError` names it.
- `libmilkcommon.so` absent while `MILK_INSTALLDIR` is set: `EnvError` names the library.
- `HAVE_CLI` is read in an environment without milk: it reports no CLI instead of raising at
  import.
- Code that caught a removed class: it must now catch the family class; no aliases remain.
- A best-effort loop in library code (e.g. the per-item `try/except` that prints a warning in
  `milk/pipeline.py`) hits several failures: every item is still attempted, then one aggregate
  error listing all failures is raised.

## Requirements

### Functional Requirements

- **FR-001**: Every module of the package MUST import with no `MILK_*` variables and none of the
  milk shared libraries it `dlopen`s (`libmilkcommon.so`, `libCLIcore.so`) present. pyMilk is a
  dependency of the py environment and stays importable; top-level pyMilk imports are unchanged.
- **FR-002**: A function needing a missing variable or library MUST raise `EnvError` when called,
  and the error MUST name what is missing.
- **FR-003**: All `MILK_*` environment access MUST go through one accessor module, `milk/env.py`.
  The shared-memory directory is exposed by `milk.env.shm_dir()`: it raises `EnvError` naming
  `MILK_SHM_DIR` when the variable is unset, empty or not a directory, and otherwise returns
  pyMilk's `MILK_SHM_DIR()` as a `Path`. No other `milk` module reads it.
- **FR-004**: `cliwrap` MUST discover build tags lazily, and MUST keep a `HAVE_CLI` accessor that
  existing tests (`tests/cli/cliwrap_test.py`) can use.
- **FR-005**: `milk/errors.py` MUST define the whole exception hierarchy under `MilkError`, with
  the families of design-v7 §4.7 rule 1 that this slice needs: at least `EnvError`, `ConfigError`,
  `TaskError` (with `TaskCannotRunError` and `TaskDidNotSucceedError`), `ComputeUnitError`,
  `LinkageError`.
- **FR-006**: Every existing exception class of the package MUST move into `milk/errors.py`.
  No other module MAY define exception classes. No compatibility aliases remain.
- **FR-007**: Exceptions raised from code touched by this slice MUST carry structured context
  (design-v7 §4.7 rule 3), and native exceptions MUST be wrapped with `raise … from`. `MilkError`
  accepts optional keyword-only context fields `pipeline`, `compu`, `task`, `path`, `cmd` and
  `stderr`. It stores them as attributes (default `None`) and appends the non-empty ones to
  `str(err)`.
- **FR-008**: pyMilk exceptions MUST propagate unwrapped.
- **FR-009**: Behaviour beyond FR-001…FR-010 MUST NOT change.
- **FR-010**: Library code in modules this slice modifies MUST NOT print, exit or swallow errors
  (design-v7 §4.7 rule 2). Best-effort loops collect failures and raise one aggregate `MilkError`
  subclass at the end. CLI entry points (`milk/infra/mains_pipeline_mgmt.py`) are exempt and keep
  their user-facing output.

### Key Entities

- **`milk.env`**: the single accessor for `MILK_*` variables; raises `EnvError` naming the missing
  variable or library. The shared-memory directory value comes from pyMilk.
- **`MilkError`**: root of the package hierarchy; families per design-v7 §4.7 rule 1. Context
  attributes: `pipeline`, `compu`, `task`, `path`, `cmd`, `stderr` (all optional).

## Out of Scope

CLI top-level error handler (anticipated, not scoped); rule 2 in modules this slice does not
modify; renames (R1b); `Pipeline.nuke` (R1c); any behaviour change beyond the above.

## Success Criteria

- **SC-001**: The existing pytest suite is green.
- **SC-002**: A test imports every package module with all `MILK_*` variables removed, and passes.
- **SC-003**: A test asserts that no module other than `milk/errors.py` defines an `Exception`
  subclass, and passes.
- **SC-004**: `infra/exceptions.py` no longer exists.
- **SC-005**: A test asserts that `milk.env.shm_dir()` raises `EnvError` naming `MILK_SHM_DIR` when
  it is unset, empty or not a directory, and returns the directory when valid.

## Assumptions

- Python ≥3.11; no new dependencies.
- "Every existing exception class" means those currently in `infra/exceptions.py`,
  `infra/task_models.py` and `cliwrap.py`; the plan maps each to a family and says where a
  dedicated subclass is kept.
- Completion also updates `../docs/` (short note on `milk.env` and `milk/errors.py`, linking
  design-v7 §4.7) and ROADMAP R1a Status/Sub-spec, per the constitution.
