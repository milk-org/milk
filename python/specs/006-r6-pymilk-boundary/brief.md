# R6 brief: pyMilk boundary

> **Status: deferred (Vincent, 2026-10-27).** It moves to the future milk/pyMilk repo-merge round,
> under a unified naming and architecture. That round consolidates the engine libs, the bindings
> build and the version checks on the SHM/FPS structs, all of which involve repo architecture
> and C changes. This brief and `draft_pymilk_boundary_test.py` stay as input for that round. Do
> not implement them now.

Roadmap: R6 · Design: design-v7 §2.3, §4.5, D-13 · Depends on: R1a · Tests: py + milk
Workflow: lean; prompts in `specs/pipelines/lean-prompts.md`. This brief and its tests replace
spec/plan/tasks, which overrides constitution §VIII for this slice.
**Vincent reviews this brief before P1.** See "Review points" at the end.

## Goal
1. `milk/` reaches pyMilk through one thin module.
2. Config validation never loads pyMilk's native bindings.
3. A contract test fails loudly when milk's installed ImageStreamIO and pyMilk's vendored copy
   disagree ("got burned AGAIN", WORKLIST.md).

## What exists today
- pyMilk is imported in 5 places in `milk/`:
  - `compute_unit.py`: `FPS`, `FPSDoesntExistError`
  - `env.py`: `MILK_SHM_DIR`, which is just `os.environ["MILK_SHM_DIR"]`
  - `infra/deploy_tasks.py`: `SHM`, imported lazily twice
  - `infra/pipeline_models.py`: `check_SHM_name`, which pulls in `ImageStreamIOWrap` (native)
  - `infra/toml_manipulation.py`: `FPVal`, under TYPE_CHECKING only
- **ISIO has a version; FPS has none.** `IMAGESTRUCT_VERSION` is "2.01" on both sides today.
  It is written into every stream's `md.version`, and milk's
  `ImageStreamIO_read_sharedmem_image_toIMAGE` refuses a stream whose version differs. The
  FPS structs (`fps_types.h`) carry no version field. The only FPS guard is behavioural:
  `tests/infra/compute_unit_fps_runtime_sync_test.py`, where milk writes and pyMilk reads all
  17 parameter types.
- pyMilk exposes no compiled-version constant, but `SHM(...).md.version` gives the same
  information at runtime. **No pyMilk change is needed.**

## Interface (the spec)

```python
# milk/pymilk_api.py  (new; the only module in milk/ that imports pyMilk)
"""Thin seam to pyMilk (design-v7 §4.5). Re-exports only: no wrappers, and errors propagate unwrapped (D-13)."""
from pyMilk.interfacing.fps import FPS, FPSDoesntExistError
from pyMilk.interfacing.shm import SHM
```

| Site | Change |
|------|--------|
| `compute_unit.py`, `deploy_tasks.py` | import from `milk.pymilk_api`. The lazy `SHM` import stays lazy (`from ..pymilk_api import SHM` inside the method). |
| `env.py` | `shm_dir()` returns `Path(os.environ["MILK_SHM_DIR"])` after the existing `_require_dir` check. Drop the pyMilk import and update the docstring. |
| `pipeline_models.py` | Replace `check_SHM_name` with a private `_shm_name(name)`: refuse any `/` (ValueError), then strip a `.im.shm` suffix. Keep the existing empty/equal checks. |
| `toml_manipulation.py` | Define `FPVal = str \| bool \| int \| float` locally under TYPE_CHECKING and drop the pyMilk import. |

## Tests
- **New:** `tests/trivial/pymilk_boundary_test.py` (py + milk):
  - Config validation in a subprocess with `ImageStreamIOWrap`, `FpsWrap` and
    `ProcessInfoWrap` blocked.
  - Stream-name suffix stripping and path refusal.
  - The ISIO contract: pyMilk writes a stream, milk's `isio-stream-info` reads it, and the
    test compares cnt0, datatype, naxis and size. Every failure message carries pyMilk's
    `md.version` and the tool's output.
- **Allowed test edit:** if a pyMilk accessor in the new test has a different spelling (for
  example `int(md.datatype)`), fix the accessor only, never the intent.
- **Do not touch** any test file that R7 is editing: `tests/compute_unit/*`, `tests/infra/*`,
  `tests/conftestaux/compute_unit.py` and `tests/conftestaux/polling.py`. Tests keep importing
  pyMilk directly; the seam is a rule for `milk/` only.

## Not doing
- An FPS version field. It needs C changes in milk and in pyMilk's vendored engine. Logged in
  LOOSE_ENDS.
- Any change to pyMilk, `fetch_milk_engine.sh`, or the repo merge (D-13).
- A runtime version check inside `milk` (for example at deploy). The contract lives in tests
  only, as design §4.5 says.
- A simulated-mismatch test. Faking a version needs either pyMilk changes or byte-poking the
  `.im.shm` file. Vincent checks the failure path once by hand instead (see below).

## Done when
- The suite is green, including the new milk-tagged contract test.
- `grep -rn "pyMilk" milk/ --include=*.py` shows only `milk/pymilk_api.py`, plus docstrings
  and comments. This is a done-check, not a test.
- Net LOC in `milk/` ≤ +30.
- ROADMAP R6 is `done`. `docs/` gets one line saying the pyMilk seam is `milk/pymilk_api.py`
  and the contract test is `pymilk_boundary_test.py`.

## Review points (Vincent)
1. **`_shm_name` refuses paths.** Today `check_SHM_name` accepts
   `$MILK_SHM_DIR/x.im.shm`-style paths in `[systemlinkage]`. The brief makes config names
   bare only, with `.im.shm` tolerated. Is any real conf using paths?
2. **The contract direction is pyMilk writes, milk reads**, via `isio-stream-info`.
   - Milk writes, pyMilk reads is already exercised by the streamdelay tests (imout). They
     don't name versions, but they do break on drift.
   - Is `isio-stream-info` always built in your milk installs? The test **fails** when it is
     missing rather than skipping, on purpose. A skipping contract test is how you got
     burned.
3. **FPS has no version.** The fpssynctest round-trip is the FPS contract for now. Do you want
   the FPS version field (milk C + pyMilk vendored) as a follow-up slice? It's in LOOSE_ENDS.
4. **Version equality alone is weak.** A struct change without a version bump passes the
   version check. The field-by-field comparison in the contract test covers the metadata
   layout but not the data region. That's acceptable to me; say if not.
5. **Failure path, checked once by hand:** build pyMilk against an older ISIO (or bump
   `VERSION_MINOR` locally in pyMilk's copy), run the contract test, and confirm the message
   names both versions. Then revert.
