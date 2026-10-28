# R7 brief: waitable states in tests (no guessed sleeps)

Roadmap: R7 (test infrastructure, outside design-v7) · Depends on: R1c · Tests: milk
Workflow: lean; prompts in `specs/pipelines/lean-prompts.md`. This brief replaces
spec/plan/tasks, which overrides constitution §VIII for this slice.

## Goal
Tests wait for an observable state, with a generous upper bound, instead of sleeping for a
guessed duration. A passing test returns as soon as the state is reached. A broken one fails
with a message that names the state it waited for.

## The rule
- **Positive state** ("X becomes true"): use
  `poll_until_or_fail(predicate, t=<upper bound>, msg=...)` from `tests/conftestaux/polling.py`.
  It already exists; reuse it and do not write a second one. Upper bounds are generous (about
  5 s): they cost nothing when the test passes.
- **Negative window** ("nothing happens"): this is the only legitimate fixed wait. Keep it
  short, give it an explicit bound (for example `get_data(..., timeout=0.1,
  return_none_on_timeout=True)`), and add a one-line comment saying why.
- **No `time.sleep` in tests** except inside `polling.py`.

## Sites

| File | Now | Becomes |
|------|-----|---------|
| `compute_unit/deployment_test.py` `test_fps_lifecycle` | `sleep(0.1)` after confstart; `sleep(1.0)` after runstart | poll `fps.conf_isrunning`; poll `fps.run_isrunning` |
| `compute_unit/deployment_test.py` `test_loops` (6 params) | `sleep(0.1)`, `sleep(1.0)`, `sleep(0.1)` between posts; 100×`sleep(0.01)` loop | see "test_loops" below |
| `infra/compute_unit_deployment_test.py` | duplicate of `test_fps_lifecycle` | **delete the file** (same behaviour is covered above) |
| `infra/compute_unit_fixtures.py` cleanup | `run_stop(timeoutsync=5)` then `sleep(0.1)` before asserting the pids are dead | poll until `os.kill(pid, 0)` raises `ProcessLookupError` |
| `compute_unit/deployment_test.py` local `func_fpsinit` / cleanup / `tp` | duplicates `infra/compute_unit_fixtures.py` | move the fixture factory to `tests/conftestaux/compute_unit.py`; both CompU test files use it; delete the copies |

### test_loops
The 1 s sleep after runstart hides a race: a trigger posted before the run loop is ready may
be flushed. Do not guess the timing. Instead:
1. Poll `run_isrunning`.
2. **Prime:** repost the input until `out.md.cnt0 >= 1`, polling every 10 ms with a 5 s bound.
   This proves the loop is consuming triggers.
3. Assert on **increments from that point:**
   - **mode 0:** poll `not run_isrunning`, then assert `cnt0 == 1` and `loopcntMax == 1`.
   - **mode 1:** `c = cnt0`; post; poll `cnt0 == c + 1`; assert it is still running.
   - **mode 2 (`loopd`):** poll `cnt0 >= 50`.

If pyMilk's `ProcessInfo` exposes a real "waiting on trigger" state, use that for readiness
instead of priming. Don't spend more than a quick look on finding out.

## Not doing
- `cli/cli_hardening_test.py` sleeps (pty/FIFO pacing). They are about 0.5 s in total and are
  milk-cli tests; logged in LOOSE_ENDS.
- A library-side wait primitive. StartConfProcesses keeps its own loop; R4 (RunStart, RunStop,
  ConfStop) is where a shared `milk` wait earns its callers. Logged in LOOSE_ENDS.
- `tests/mains/`, which pytest ignores.
- Changing what any test asserts, beyond the increment reformulation above.

## Done when
- The suite is green, run **3 times in a row** to catch flakiness.
- Wall time drops by about 10 s. The removed sleeps total about 11.5 s. Measure with
  `pytest --durations=15 tests/compute_unit tests/infra tests/pipeline` before and after, and
  record both totals here.
- `grep -rn "time.sleep" tests/ --include=*_test.py | grep -v tests/cli/` returns nothing. This
  is a done-check for the slice, not a test.
- Net LOC in `tests/` ≤ 0 (the deletions should outweigh the additions). No change to `milk/`.

| | Before | After |
|--|--|--|
| durations total (s) | | |
