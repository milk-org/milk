# Copilot instructions: `milk/python`

This workspace is the `python/` folder of the milk repository. Before any task:

1. Read and follow the **Critical Rules** in `../CLAUDE.md`. They apply to every agent, including
   you. In particular: no git operations that move branches/HEAD/tags, touch GitHub, or overwrite
   working-tree files; no scope creep; flag duplicates.
2. Read `../AGENTS.md` for repository context. `../docs/` is the human source of truth. Link to it;
   never duplicate it.
3. Pipeline work is governed by `specs/pipelines/design-v7.md` (frozen) and its Appendix A.
   Use the terms in its §0 exactly. Never edit these two files. In
   `specs/pipelines/ROADMAP.md`, edit only the Status and Sub-spec columns.
4. Implement exactly one roadmap slice per feature. Items listed as "not to be specified" or
   "anticipated" must not be implemented.
5. If a spec, plan or task conflicts with design-v7, stop and ask Vincent. Do not resolve the
   conflict yourself.
6. Formatting is delegated to pre-commit. Do not spend effort formatting Python by hand: no manual
   line wrapping, import sorting or style passes, and no formatting-only edits or tasks. At the end
   of each implement phase, run from this folder, in order: `git add .`, `pre-commit run`,
   `git restore --staged .`. The repository's hooks (configured in ../.pre-commit-config.yaml)
   then format what changed. A pre-commit failure that only reports "files were modified by this
   hook" is expected and needs no action. Report any other hook failure to Vincent instead of
   looping on it. This sequence is a sanctioned exception to the ../CLAUDE.md git rules: it
   touches only the index and lets hooks rewrite files, and it never commits, moves
   branches/HEAD/tags or contacts GitHub. Use no other git commands.

Tests: `pytest` from this folder. The **milk**-tagged slices also need `MILK_INSTALLDIR` and a
milk build (see `tests/conftestaux/milk.py`). The existing suite must stay green.

**Code rules (important):**
- No new abstraction (class, protocol, registry, factory) unless it has two real callers now.
- Validate only at boundaries: config load, the CLIs, the pyMilk seam. Internal code trusts its
  inputs.
- No options, flags or hooks that nothing in the current slice uses.
- Prefer deleting to adding. Report the net line count of every change. The slice brief sets the
  budget (default +300 net LOC); if you will exceed it, stop and ask.

**Test rules (important):**
- **No meta-testing.** Tests exercise behaviour through the Python API, the CLIs or the live system.
  Never read, grep or analyze the text of the source tree to assert on code. That rules out checks
  such as which arguments an FPS declaration passes, where a constructor is called, where a class is
  defined, or which module imports what. Structural rules (one construction point, one exception
  module, no duplication) are enforced in review, not in tests.
- `ast` is off-limits for any systematic use. It may occasionally parse a single expression under
  test, never the source tree.
- Fixtures shared across test files live in `tests/conftestaux/` and are registered in
  `tests/conftest.py` (`pytest_plugins`). Do not add `conftest.py` files in subfolders. Reuse an
  existing fixture before writing a new one.
