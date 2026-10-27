# R1c prompts (lean workflow trial)

Paste these into Claude Code, opened in `milk/python/`. Use one fresh session per prompt. No
Spec Kit commands.

## P1: Implement

```text
Implement roadmap slice R1c.
The spec is specs/003-r1c-compu-views-fixes/brief.md plus the tests in
tests/pipeline/compu_views_test.py and the fixtures in tests/conftestaux/pipeline.py. Read those three files, then
the code they touch. Do not read design-v7 unless the brief is ambiguous.

Rules:
- Make the R1c tests pass and keep the whole suite green (`pytest`).
- Do not edit the R1c tests. If one of them looks wrong, stop and tell me why.
- Update existing tests only where they read the old attributes: specs move to
  pp.config.compute_units, is_running becomes any_run_active. Prefer the views
  (pp.compute_units[cu].fps) over building a ComputeUnit in tests.
- Follow the test rules in .github/copilot-instructions.md (no meta-testing of source text).
- Lean: no new module, class, helper or option. Early returns. Comments only where the why is not
  obvious. Validate only where the brief says to.
- If you stop short of something, add a line to specs/pipelines/LOOSE_ENDS.md instead of
  designing ahead. Do not implement R4 material.
- Budget: net +60 LOC in milk/. Report `git diff --stat` at the end.
- Finish with the pre-commit sequence from .github/copilot-instructions.md.
No plan document. Go straight to code.
```

## P2: Tighten (fresh session)

```text
Review the uncommitted diff in milk/ (git diff -- milk/) for slice R1c.
Make it smaller and plainer without changing behaviour:
- inline single-use helpers and variables;
- remove defensive checks that are not at a boundary, and comments that restate the code;
- remove imports that are now unused (for example ComputeUnit in deploy_tasks.py, if only typing uses it);
- prefer early returns.
The tests must stay green. Do not touch tests/. Report the LOC before and after.
```

## P3: Close out (same session as P2, or by hand)

```text
Set R1c to done in specs/pipelines/ROADMAP.md (Status and Sub-spec columns only; Sub-spec =
specs/003-r1c-compu-views-fixes/brief.md). Grep docs/ for is_running and for compute_units used as
specs, and fix only those. List the open entries of specs/pipelines/LOOSE_ENDS.md.
```

## Metrics to log (for comparison with R1a/R1b)

| | R1b (Spec Kit) | R1c (lean) |
|--|--|--|
| Wall clock, first prompt → green | | |
| Prompts you sent | | |
| Times you had to intervene or correct | | |
| Spec artifacts (bytes) | 37 k (spec+plan+tasks) | ~3 k brief + tests |
| `git diff --stat` milk/ (+/−) | | |
| Net LOC after tighten | | |
| Your read of the code (1–5) | | |
