# R1a: Import safety + errors

Feature directory: `specs/001-r1a-import-safety` · Tests: py

## /speckit-specify

```text
Feature directory: specs/001-r1a-import-safety
Parent roadmap: specs/pipelines/ROADMAP.md → entry R1a. Design: specs/pipelines/design-v7.md (§4.5, §4.7; defects B3, X8).

Make the milk Python package safe to import, and give it a single exception contract.

Why: some modules cannot be imported without a full milk environment: mains_pipeline_mgmt reads MILK_ROOT at import, and cliwrap reads MILK_INSTALLDIR and loads libmilkcommon.so at import (B3). Exception classes are also scattered across modules, with no common base, so callers cannot handle failures uniformly (X8).

Requirements:
- Every module of the package imports without MILK_* environment variables or milk shared libraries. A missing variable or library raises EnvError when a function that needs it is called, and the error names what is missing.
- All MILK_* environment access goes through one accessor module.
- cliwrap discovers build tags lazily. The HAVE_CLI accessor used by existing tests keeps working.
- One module, milk/errors.py, defines the whole hierarchy under MilkError, with the families of design-v7 §4.7 rule 1 that this slice needs (at least EnvError, ConfigError, TaskError with TaskCannotRunError/TaskDidNotSucceedError, ComputeUnitError, LinkageError). Every existing exception class of the package moves into it. No other module defines exception classes. There are no compatibility aliases.
- Exceptions raised from code touched by this slice carry structured context (design-v7 §4.7 rule 3).
- pyMilk exceptions propagate unwrapped.

Out of scope: the CLI top-level error handler, renames (R1b), Pipeline.nuke (R1c), and any behaviour change beyond the above.

Success criteria: the existing suite is green; a test imports every package module with all MILK_* variables removed; a test asserts that no module other than milk/errors.py defines an Exception subclass.
```

## /speckit-plan

```text
Python ≥3.11. Add no new dependencies. Package root is milk/; tests are in tests/.
New modules: milk/env.py (the accessor; raises EnvError) and milk/errors.py.
Existing classes to fold in: infra/exceptions.py (PipelineConfFolderNotFoundException, PipelineTomlNotFoundException, UnknownSystemLinkageException, SystemLinkageException, FPSExecException), infra/task_models.py (NoCanTaskError, NoSuccessTaskError) and cliwrap.py (MilkBuildException, CLICrashError). Map each one to a family class. Keep a dedicated subclass only where a caller or test distinguishes it, and say which in the plan. Delete infra/exceptions.py and update imports, tests included.
Module-level reads (MILK_SAMPLES_DIR/CACAO_SAMPLES_DIR in mains_pipeline_mgmt; MILK_BUILD_TAGS/HAVE_CLI/HAVE_TREESITTER/MILK_CLI_EXEC in cliwrap) become lazy functions or cached accessors. Keep HAVE_CLI importable for tests/cli/cliwrap_test.py.
Follow the constitution: early returns, small functions, no aliases, docstrings say why.
Tests: py. Use monkeypatch.delenv for the clean-env import test. The "no exception classes elsewhere" test walks milk/ with ast.
```

## /speckit-implement

Run it in one pass; the slice is small.

## Done when

- The suite is green, the two new tests pass, and `infra/exceptions.py` is gone.
- ROADMAP R1a is set to done and Sub-spec to `specs/001-r1a-import-safety/`.
- `../docs/` has a short note on `milk.env` and `milk/errors.py` (link to design-v7 §4.7; don't copy it).
