# Feature Specification: One config model + compute-unit vocabulary

**Roadmap slice**: R1b ([ROADMAP.md](../pipelines/ROADMAP.md)) · **Tests**: milk

**Design**: [design-v7.md](../pipelines/design-v7.md) §0, §4.4, D-15, D-19; defects B2, X1, X2, X9 ·
[design-v7-appendix-a.md](../pipelines/design-v7-appendix-a.md) §A.1

**Created**: 2026-10-27 · **Status**: Draft

## Overview

Give pipeline configuration one model and one vocabulary.

Why: the config exists in three representations (X1), the same FPS-naming expression is repeated in
five places (X2), unknown TOML keys are silently ignored (B2), and "session" names a concept the
design calls a compute unit (X9).

## User Scenarios & Testing

### User Story 1 - One validated config, closed key set (Priority: P1)

A pipeline author writes a `conf.toml` in the Appendix A.1 shapes. A typo, or a key from the old
format, is reported instead of being silently ignored.

**Independent Test**: Load a valid migrated `conf.toml`, then the same file with one extra
top-level key.

**Acceptance Scenarios**:

1. **Given** a `conf.toml` with identity keys, `[compute_units.<cu>]` tables, `[[dataloads]]` and
   `[systemlinkage.<mode>]`, **When** it is loaded, **Then** one frozen `PipelineConfig` is
   returned, whose `compute_units` maps each CompU name to a `ComputeUnitSpec` (exec, FPS params,
   procinfo).
2. **Given** a `conf.toml` with any other top-level key (including the old `tasks`, `[sessions]`,
   and per-CompU top-level tables such as `[delay0]`), **When** it is loaded, **Then**
   `ConfigError` is raised and names the key.
3. **Given** a procinfo table under `.fps.procinfo` with an unknown field, **When** it is loaded,
   **Then** `ConfigError` is raised and names the field.
4. **Given** a CompU table without `exec`, **When** it is loaded, **Then** `ConfigError` names the
   CompU and the missing key.

### User Story 2 - FPS names from one place (Priority: P1)

An operator or test refers to the FPS of a CompU and always gets the same name, `<name>_<cu>`.

**Independent Test**: Deploy a migrated pipeline and list its FPS.

**Acceptance Scenarios**:

1. **Given** pipeline `name = "basic"` and CompU `delay0`, **When** its FPS name is requested,
   **Then** it is `basic_delay0`.
2. **Given** the package, **When** it is searched for the FPS-naming expression, **Then** it is
   defined in exactly one function and every deploy, start, stop and check path calls it.
3. **Given** a deployed pipeline, **When** the live FPS set is listed, **Then** it holds the names
   of this rule (no `delay0_000`-style names).

### User Story 3 - "Session" means tmux only (Priority: P2)

A maintainer reads code, CLI help, tests, example configs and docs, and meets the words "compute
unit" / CompU for the unit, and "session" only for tmux sessions.

**Independent Test**: Search `milk/`, `tests/`, the example configs and the docs for "session".

**Acceptance Scenarios**:

1. **Given** the repo, **When** searched for compute-unit-related "session" identifiers (classes,
   functions, attributes, modules, test helpers, test files), **Then** there are none.
2. **Given** `milk-pipes --help` and its subcommands' help, **When** read, **Then** they say
   "compute unit"/CompU wherever they said "session" for that concept.
3. **Given** `../docs/` and `../AGENTS.md`, **When** searched for the abbreviation "CU", **Then**
   it appears nowhere with the meaning compute unit; it reads "CompU".

### User Story 4 - Glossary in the docs (Priority: P2)

A user or agent looks up a pipeline term in the human documentation.

**Acceptance Scenarios**:

1. **Given** `../docs/`, **When** the pipelines page is opened, **Then** it links to a glossary
   page that carries the design-v7 §0 definitions.
2. **Given** that glossary, **When** compared with design-v7 §0, **Then** the terms and
   definitions are the same (the glossary is the published copy; design-v7 stays frozen).

### Edge Cases

- Old-format `conf.toml` (`[sessions]`, top-level CompU tables, string-form `dataloads`,
  `tasks`): refused with `ConfigError` naming the first offending key; no automatic conversion.
- A CompU table with no `.fps` sub-table: valid; empty FPS params and no procinfo.
- A CompU named like a former reserved key (e.g. `tasks`, `name`): valid, since it lives under
  `compute_units`.
- `{var}` references to a former top-level alias key (`_ln`): the alias key is now an unknown
  key; examples reference `{loop_number}` directly.
- Dataloads written as `"src -- target"` strings: refused (table-only).
- A `systemlinkage` link that is invalid keeps raising as today.

## Requirements

### Functional Requirements

- **FR-001**: There MUST be one validated, frozen configuration model, `PipelineConfig`, with
  `compute_units: dict[str, ComputeUnitSpec]`. A `ComputeUnitSpec` holds `exec`, the FPS params
  and the procinfo settings. The plain duplicate `PipelineConfig` class and `LOOP_INFO` MUST be
  removed, and no code copies fields between two config representations.
- **FR-002**: `conf.toml` MUST use the Appendix A.1 shapes for the keys implemented so far:
  - identity keys `long_name`, `name`, `loop_number`;
  - `[compute_units.<cu>]` with `exec` and an optional `.fps` sub-table, which is written
    verbatim as FPS params (dotted keys flattened as today);
  - procinfo under `.fps.procinfo`, validated by the existing procinfo schema (unknown fields
    refused);
  - table-only `[[dataloads]]` entries with `source` and `target`;
  - `[systemlinkage.<mode>]` tables, with today's validation.
- **FR-003**: The top-level key set MUST be closed to `long_name name loop_number compute_units
  dataloads systemlinkage`. Any other key, including the old `tasks` and `[sessions]`, MUST raise
  `ConfigError` naming the key. `kind`, `includes`, `hooks` and `plans` are not accepted until
  the slices that implement them.
- **FR-004**: Schema failures in `conf.toml` MUST surface as `ConfigError` (with the `path`
  context) chained from the underlying validation error. The key or field at fault MUST appear in
  the message.
- **FR-005**: The FPS name of a CompU MUST be `<name>_<cu>` and MUST be computed in exactly one
  place; every site that needs it calls that function.
- **FR-006**: The word "session" MUST NOT name the compute-unit concept anywhere: classes,
  functions, attributes, modules, CLI help, test helpers, test files, example configs. It remains
  only for tmux sessions.
- **FR-007**: Every `conf.toml` in the repo (test resources, `../milk-examples/pipelines/`) MUST
  be migrated to the new shapes. `{var}` references follow the new paths (e.g.
  `{compute_units.delay0.fps.out_name}`).
- **FR-008**: The design-v7 §0 glossary MUST be published in `../docs/` and linked from the
  pipelines page. In `../docs/` and `../AGENTS.md`, the abbreviation "CU" meaning compute unit
  MUST become "CompU".
- **FR-009**: Behaviour beyond FR-001…FR-008 MUST NOT change. The existing `{var}` substitution
  stays as is, and `load_pipeline_config` keeps its role.

### Key Entities

- **`PipelineConfig`**: the one frozen config model: identity keys, `compute_units`, `dataloads`,
  `systemlinkage`.
- **`ComputeUnitSpec`**: one CompU's `exec`, FPS params and procinfo.
- **FPS name function**: the single definition of `<name>_<cu>`.

## Out of Scope

Includes, kinds, hooks, plans, overrides and substitution changes (R2); `Pipeline.compute_units`
as a view property (R1c); rundir claim and uniqueness enforcement (D-19, later slice); fixing
bugs other than B2.

## Success Criteria

- **SC-001**: The existing pytest suite is green after the migration (milk-tagged tests run in a
  milk environment).
- **SC-002**: A test asserts that a `conf.toml` with an unknown top-level key (and one with the
  old `tasks`/`[sessions]`) raises `ConfigError` naming it.
- **SC-003**: Tests assert that FPS names of deployed test pipelines follow `<name>_<cu>`, and
  that the naming expression is defined once.
- **SC-004**: A search of `milk/` and `tests/` finds no compute-unit-related "session"
  identifiers; remaining hits concern tmux.
- **SC-005**: `PipelineConfigModel`, the plain `PipelineConfig` and `LOOP_INFO` no longer exist.
- **SC-006**: The glossary page exists under `../docs/`, is linked from the pipelines page, and a
  search of `../docs/` and `../AGENTS.md` finds no "CU" meaning compute unit.

## Assumptions

- Python ≥3.11, pydantic 2, tomli; no new dependencies.
- The FPS name changes from today's `<cu>_<loop_number:03d>` (e.g. `delay0_000`) to
  `<name>_<cu>` (e.g. `basic_delay0`), per D-19. Tests that hard-code the old names are updated.
- Compute-unit tables are closed too (`exec`, `fps` only), so an old-shape key left directly under
  a CompU is refused rather than ignored, which is the same silent-ignore defect as B2.
- Migrated configs: `tests/resources/pipelinebasic-conf`, `tests/resources/simulator-rootdir`,
  `../milk-examples/pipelines/aolinearsimulator-conf`. The cacao example
  `../plugins/cacao-src/examples/dmcomb-conf/conf.toml` belongs to another repository and is
  outside this slice; it will be refused until migrated there.
- Substitution remains unchanged. Values inside table arrays, including `[[dataloads]]`, pass
  through untouched, consistent with design-v7 D-26. Migrated dataload entries use literal values.
- The glossary's file name and nav entry are a plan decision.
- Completion also updates ROADMAP R1b Status/Sub-spec, per the constitution.
