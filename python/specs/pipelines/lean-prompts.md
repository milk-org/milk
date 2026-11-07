# Lean prompts (any slice)

Replace `<ID>` and `<BRIEF>` (for example `R2` and `specs/004-r2-confresolve/brief.md`). Run them
in Claude Code opened in `milk/python/`, one fresh conversation per prompt (P3 may share P2's).
Do not use Spec Kit commands.

## P1: Implement

```text
Implement roadmap slice <ID>. The spec is <BRIEF> plus the test files listed in its Tests section.
Read those, then the code they touch. Read design-v7 only if the brief is ambiguous.

- Make the slice tests pass and keep the whole suite green (`pytest`).
- Do not edit the slice tests. If one looks wrong, stop and tell me why.
- Edit existing tests only as the brief's Touch section allows.
- Follow the code rules, test rules and stop rule in .github/copilot-instructions.md.
- If a failure looks external (milk/pyMilk, binaries, tmux, OS, races in external processes, the
  test environment), stop after at most two attempts: no retries, sleeps, loosened asserts,
  skips/xfails or redesigns. Report the symptom, the evidence, what you tried and 2-3 options,
  then wait for me.
- Settle the brief's "Decisions for Vincent" as written; if one does not hold up in the code,
  stop and ask.
- If you stop short of something, add one line to specs/pipelines/LOOSE_ENDS.md instead of
  designing ahead. Implement nothing from later slices.
- Stay within the brief's budget. If you will exceed it, stop and ask. Report `git diff --stat`
  and the net line count at the end.
- Finish with the pre-commit sequence from .github/copilot-instructions.md.
No plan document. Go straight to code.
```

## P2: Tighten (fresh conversation)

```text
Review the uncommitted diff in milk/ (git diff -- milk/) for slice <ID>.
Make it smaller and plainer without changing behaviour: inline single-use helpers and variables unless they have multiple use-points. Favor human readability, do not use abstractions for two uses or less.
Remove defensive checks that are not at a boundary, comments that restate the code, and unused
imports; prefer early returns. Apply the code rules in .github/copilot-instructions.md.
The tests must stay green. Do not touch tests/. Report the net line count before and after.
If something fails for an external reason, stop and report (stop rule); do not work around it.
```

## P3: Close out

```text
Set <ID> to done in specs/pipelines/ROADMAP.md (Status and Sub-spec columns only;
Sub-spec = <BRIEF>). Make the docs/ update the brief's "Done when" asks for, and nothing else.
List the open entries of specs/pipelines/LOOSE_ENDS.md.
```
