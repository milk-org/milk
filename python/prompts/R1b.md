# R1b: One config model + compute-unit vocabulary

Feature directory: `specs/002-r1b-model-vocabulary` · Tests: milk

## /speckit-specify

```text
Feature directory: specs/002-r1b-model-vocabulary
Parent roadmap: specs/pipelines/ROADMAP.md → entry R1b. Design: specs/pipelines/design-v7.md (§0, §4.4, D-15, D-19; defects B2, X1, X2, X9) and design-v7-appendix-a.md §A.1.

Give pipeline configuration one model and one vocabulary.

Why: the config exists in three representations (X1), the same FPS-naming expression is repeated in five places (X2), unknown TOML keys are silently ignored (B2), and "session" names a concept the design calls a compute unit (X9).

Requirements:
- One validated configuration model (PipelineConfig) with typed compute units (ComputeUnitSpec: exec, FPS params, procinfo). The plain duplicate class and LOOP_INFO are removed.
- conf.toml uses the Appendix A.1 shapes for the keys implemented so far: identity keys (long_name, name, loop_number), nested [compute_units.<cu>] with exec plus a .fps sub-table (procinfo under .fps.procinfo, schema-checked), table-only [[dataloads]] (source/target), and [systemlinkage.<mode>].
- The top-level key set is closed. Any other key, including the old `tasks` and `[sessions]`, raises ConfigError naming the key. (Later slices add includes, kind, hooks, plans.)
- The FPS name of a compute unit is <name>_<cu>, computed in exactly one place.
- "Session" no longer names compute units anywhere: classes, functions, attributes, CLI help, test helpers and example configs. The word is kept only for tmux sessions.
- Every conf.toml in the repo (test resources, milk-examples) is migrated to the new shapes.
- The §0 glossary of design-v7 is published in ../docs/ (linked from the pipelines page), and the "CU" abbreviation in ../docs and ../AGENTS.md becomes "CompU".

Out of scope: includes, kinds, overrides and substitution changes (R2); Pipeline.compute_units as a view property (R1c).

Success criteria: the suite is green after the migration; a conf.toml with an unknown top-level key fails with ConfigError naming it; FPS names in tests follow <name>_<cu>; a search finds no compute-unit-related "session" identifiers.
```

## /speckit-plan

```text
Python ≥3.11, pydantic 2, tomli. No new dependencies.
Collapse infra/pipeline_models.py (PipelineConfigModel, the plain PipelineConfig, LOOP_INFO) into one frozen pydantic PipelineConfig with compute_units: dict[str, ComputeUnitSpec]. Use extra="forbid" at the top level so the key set is closed, and keep the existing procinfo schema (extra="forbid").
load_pipeline_config keeps its role for now; R2 replaces it with confresolve. The existing {var} substitution stays as is.
Renames: session.py → compute_unit.py, ComputeSession → ComputeUnit, get_session/get_session_abs_name → one fps_name function, self.sessions → the spec dict (R1c turns it into a view property), and test helpers (e.g. StreamDelayComputeSession). Update the CLI help text.
Migrate tests/resources/*/conf.toml and ../milk-examples/pipelines/*/conf.toml. Substitution references change accordingly (e.g. {compute_units.delay0.fps.out_name}).
Docs: publish the glossary under ../docs/ and link it; replace "CU" with "CompU" in ../docs and ../AGENTS.md.
Tests: milk (existing conftest spoofing).
```

## /speckit-implement

Two phases: (1) the model, the TOML shapes and migrating the configs; (2) the vocabulary rename and docs.

## Done when

- The suite is green and ROADMAP R1b is done.
- The glossary is in `../docs/`, and nothing in the repo uses "CU" to mean a compute unit.
