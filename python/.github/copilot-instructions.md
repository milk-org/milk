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

Tests: `pytest` from this folder. The **milk**-tagged slices also need `MILK_INSTALLDIR` and a
milk build (see `tests/conftestaux/milk.py`). The existing suite must stay green.
