# R1c brief: CompU views + bug fixes

Roadmap: R1c · Design: design-v7 §4.1, X3, B1, B4–B7 · Tests: milk
**Workflow trial:** this brief and `tests/pipeline/compu_views_test.py` replace spec/plan/tasks.
Vincent has sanctioned this trial, which overrides constitution §VIII for R1c only.

## Goal
Compute units are obtained in exactly one way, and the known task bugs are fixed.

## Touch
`milk/pipeline.py`, `milk/infra/deploy_tasks.py`, `milk/infra/mains_pipeline_mgmt.py` (nuke help
text only), and the existing tests that read the old attributes. Nothing else.

## Interfaces (the spec)

```python
# milk/pipeline.py
class Pipeline:
    # __init__: delete `self.compute_units = self.config.compute_units`.
    # Specs are read via self.config.compute_units.

    @property
    def compute_units(self) -> dict[str, ComputeUnit]:
        """Fresh views keyed like config.compute_units. The only ComputeUnit construction in milk/."""
        return {cu: ComputeUnit(spec.exec, self.config.fps_name(cu))
                for cu, spec in self.config.compute_units.items()}

    def any_run_active(self) -> bool: ...   # replaces is_running(), which is deleted
```

Building fresh views on every access is the refresh behaviour: an invalid or stale FPS link
cannot survive. Do not cache them.

## Fixes
| Bug | Change |
|-----|--------|
| B1 | `StartConfProcesses.can`: `return False` after the invalid-FPS failure_reason. |
| B4 | `InitialFolderSetup.can`: if `root_folder` exists (`task_do` has already found success False), set failure_reason naming the path ("partial rootdir; remove it") and return False. |
| B5 | `LoadDataFiles.success`: each target stream exists, and its data shape and dtype equal the FITS source's. Ignore byte order (FITS is big-endian). |
| B6 | `StartConfProcesses.forward`: on timeout, raise `TaskDidNotSucceedError` listing the FPS names whose conf process is not running, plus the hint "tmux session busy?". |
| B7 | `nuke`: FPS destroy goes through `attempt(f"destroy {cu}", …)`. Add the step to the CLI help text. |
| X3 | Every `ComputeUnit(...)` in tasks and in `nuke`/`any_run_active` uses `self.compute_units`. |

## Not doing
State model, rundir, teardown and plans (R4). Deleting `ReversibleTask`/`CompoundTask`/`TestConfig`/`task_undo` (R4).
The `DMCombPipeline` stubs (X5). Detecting a busy tmux in `can()`: logged in LOOSE_ENDS and
deferred to R4, where teardown owns tmux. Any new abstraction, option or helper module.

## Done when
- `tests/pipeline/compu_views_test.py` passes and the whole suite is green.
- The net diff in `milk/` is at most about +60 LOC (tests excluded). Being over budget means stop and ask.
- ROADMAP R1c is `done`. `docs/` is updated only where it names `is_running` or the old attribute.
