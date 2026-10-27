# milk python pipeline layer Constitution

Scope: package `milk` in `python/`. This constitution points to its sources; it does not restate them.

## Core Principles

### I. Design Contract
[design-v7.md](../../specs/pipelines/design-v7.md) and
[design-v7-appendix-a.md](../../specs/pipelines/design-v7-appendix-a.md) are normative and frozen.
Terms are used exactly as defined in design-v7 §0, with no synonyms (notably "compute unit"/CompU,
never "session" except for tmux). Decisions D-1…D-26 (design-v7 §3) are binding. Only Vincent can
amend the design (it becomes v8); agents never edit it.

### II. Repository Rules
The Critical Rules in [CLAUDE.md](../../../CLAUDE.md) apply to every agent: no git operations that
move branches/HEAD/tags, touch GitHub, or overwrite working-tree files; no scope creep; ask when
ambiguous; flag duplicate infrastructure, features or docs. [docs/](../../../docs/) is the human
source of truth: link to it, never duplicate it.

### III. Scope Discipline
One feature = one slice of [ROADMAP.md](../../specs/pipelines/ROADMAP.md), and its spec cites the
slice ID. Items listed as "not to be specified" or "anticipated" (ROADMAP, design-v7 §5) must not
be implemented. A conflict between any artifact and design-v7 stops work until Vincent decides.

### IV. Lean and Readable
Per design-v7 §4.8.1–2: low-volume docs with clear intent; docstrings say why. Flat control flow,
early returns and guard clauses, small intent-named functions. Indirection only at the declared
protocols of design-v7 §4.9 (Task, ConfHook, ConfArtifact, PipelineKind), except for lazy loading
needed to keep imports safe while preserving existing accessors. Vincent approved this exception
on 2026-10-27; it permits module-level `__getattr__` for that purpose, not unrelated indirection.

### V. No API Duplication
Per design-v7 §4.8.3: reuse or extend before adding. One construction point for compute units
(`Pipeline.compute_units`), one way to get a config (confresolve), one exception module, one
registry. CLIs only parse arguments and call the API.

### VI. Exception Contract
Per design-v7 §4.7 rules 1–4: all exception classes live in `milk/errors.py` under `MilkError`.
Library code never prints, exits or swallows errors; best-effort operations raise one aggregate
error. Exceptions carry structured context. pyMilk exceptions propagate unwrapped.

### VII. Tests
The existing pytest suite stays green at every slice. New behaviour ships with tests in the
environment the ROADMAP names (py or milk). The test rules in
[copilot-instructions](../../.github/copilot-instructions.md) bind (no meta-testing of source text,
no systematic `ast`, shared fixtures in `tests/conftestaux/`).

### VIII. Delivery
Lean workflow: `spec.md`, `plan.md` and `tasks.md` per feature. Flow-forward persistence: finished
feature folders are frozen records. Each slice ends by updating [docs/](../../../docs/) and the
slice's Status/Sub-spec cells in the ROADMAP.

### IX. Formatting
Formatting is delegated to pre-commit. Do not spend effort formatting Python by hand: no manual line
wrapping, import sorting or style passes, and no formatting-only edits or tasks. At the end of each
implement phase, run from this folder, in order: `git add .`, `pre-commit run`,
`git restore --staged .`. The repository's hooks (configured in ../.pre-commit-config.yaml) then
format what changed. A pre-commit failure that only reports "files were modified by this hook" is
expected and needs no action. Report any other hook failure to Vincent instead of looping on it.
This sequence is a sanctioned exception to the ../CLAUDE.md git rules: it touches only the index and
lets hooks rewrite files, and it never commits, moves branches/HEAD/tags or contacts GitHub. Use no
other git commands.

## Governance

Vincent ratifies and amends this constitution.

**Version**: 1.3.0 | **Ratified**: 2026-10-27 | **Last Amended**: 2026-10-27
