---
description: "Task list for R1b: One config model + compute-unit vocabulary"
---

# Tasks: One config model + compute-unit vocabulary

**Input**: [spec.md](./spec.md), [plan.md](./plan.md) · **Roadmap slice**: R1b · **Tests**: milk

**Tests**: Required (SC-001 to SC-006). Run `pytest` from `python/` with `MILK_INSTALLDIR` set; the
milk-tagged tests need it.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: different files, no dependency on an unfinished task
- **[Story]**: US1 (one validated config), US2 (FPS names from one place), US3 ("session" means tmux
  only), US4 (glossary)
- Paths are relative to `python/` unless they start with `../`
- Files are moved with plain `mv`, no git commands (constitution IX). No formatting-only tasks.
- At the end of each implement phase, run from `python/`, in order: `git add .`,
  `pre-commit run`, `git restore --staged .` (constitution IX). A failure reporting only
  hook-modified files requires no action; report any other failure to Vincent without looping.
  T033 is the closing phase's invocation.

---

## Phase 1: Setup

- [ ] T001 Run `pytest` from `python/` with `MILK_INSTALLDIR` set and record the baseline (passed,
  failed, skipped), so that later phases can be compared against it

---

## Phase 2: Foundational (blocks US1 and US2)

**Purpose**: the single model, its loader, and the migrated configs. US1 and US2 share these files
(`pipeline_models.py`, `deploy_tasks.py`, `pipeline.py`), so they ship together here and are
checked by separate tests in Phases 3 and 4.

- [ ] T002 `milk/infra/pipeline_models.py`: delete `LOOP_INFO`, `PipelineConfigModel`, the plain
  `PipelineConfig` and `DataLoadEntryModel.from_str` with its `before` validator. Add
  `ComputeUnitSpec` (`extra="forbid"`, frozen; `exec: str`, `fps: dict[str, Any] = {}`; a field
  validator runs `ProcessInfoData.model_validate(fps["procinfo"])` when the key is present,
  refusing unknown fields and non-table values with `pydantic.ValidationError`; the `procinfo`
  property uses the same model-validation API when present, otherwise returning `None`, without
  storing a duplicate) and the new frozen `PipelineConfig` (`extra="forbid"`;
  `long_name`, `name`, `loop_number`, `compute_units`, `dataloads`, `systemlinkage`; move
  `_check_systemlinkage` onto it unchanged). `DataLoadEntryModel` becomes `extra="forbid"`,
  `source` and `target` only. Keep `ProcessInfoData` unchanged.
- [ ] T003 `milk/infra/pipeline_models.py` (after T002): add `PipelineConfig.fps_name(cu)` returning
  `f"{self.name}_{cu}"`, with a docstring saying why (D-19). It is the only place the expression
  exists.
- [ ] T004 `milk/infra/toml_manipulation.py` (after T002): `load_pipeline_config` keeps its
  signature and the unchanged `substitute_toml_variables_in_nested`; then
  `PipelineConfig.model_validate(data)`, re-raising `pydantic.ValidationError` as
  `ConfigError(str(err), path=toml_path) from err`. Delete the field-copying block and the
  `session_configs` construction.
- [ ] T005 [P] `milk/pipeline.py` `Pipeline.__init__`: `self.compute_units =
  self.config.compute_units`; remove `self.sessions` and `self.session_configs`; keep `dataloads`,
  `systemlinkage`, `short_name`, `loop_number`. Delete `get_session_abs_name` and `get_session`.
  Update `is_running` and `nuke` to build `ComputeSession(spec.exec, self.config.fps_name(cu))`
  from `self.compute_units.items()` (the class is renamed in Phase 5).
- [ ] T006 [P] `milk/infra/deploy_tasks.py`: every `ComputeSession(exec, s + f"_{p.loop_number:03d}")`
  becomes `ComputeSession(spec.exec, p.config.fps_name(cu))`, iterating
  `p.compute_units.items()`; `ConfigureFPSs` denests `spec.fps` instead of
  `p.session_configs[…]`; the "FPS … not found/invalid" messages use `fps_name`. Remove the
  `TODO … Pipeline.sessions property` comment (R1c's).
- [ ] T007 [P] Migrate `tests/resources/pipelinebasic-conf/conf.toml`: drop `tasks` and
  `[sessions]`; `[delay0]`/`[delay1]` → `[compute_units.<cu>]` with `exec`, `.fps` and
  `.fps.procinfo` (same values); refs → `{compute_units.delay0.fps.in_name}` / `.out_name`,
  including `systemlinkage.shared.middle`.
- [ ] T008 [P] Migrate `tests/resources/simulator-rootdir/conf.toml`: drop `tasks`, `[sessions]` and
  the empty tables; three `[compute_units.<cu>]` tables with `exec` only.
- [ ] T009 [P] Migrate `../milk-examples/pipelines/aolinearsimulator-conf/conf.toml`: drop `tasks`,
  `_ln`, `[sessions]`; nest the CompU tables as in T007 (`option.MODENORM` stays a dotted key under
  `.fps`); `immodes = 'aol{loop_number}_simu_modes'`; refs `{mvm.outcoeff}` →
  `{compute_units.mvm.fps.outcoeff}`; dataloads → `[[dataloads]]` with
  `source = "sim_matrix_flat1.fits"`, `target = "aol1_simu_modes"` (literal, plan Flag 1).
- [ ] T010 Grep `milk/` and `tests/` for `PipelineConfigModel`, `LOOP_INFO`, `session_configs`,
  `.sessions`, `get_session`, `from_str`, `loop_number:03d`; fix any leftover use outside the test
  files handled in Phases 3–4 (after T002–T006).

**Checkpoint**: `milk` imports; `load_pipeline_config` on the three migrated configs returns a
`PipelineConfig`.

---

## Phase 3: User Story 1 — One validated config, closed key set (P1) 🎯 MVP

**Goal**: one frozen `PipelineConfig`; unknown keys and fields are refused with `ConfigError`.

**Independent Test**: `tests/pipeline/toml_schemas_test.py` passes.

- [ ] T011 [US1] Rewrite `tests/pipeline/toml_schemas_test.py` on `PipelineConfig`, via
  `PipelineConfig.model_validate` (expecting `pydantic.ValidationError`): valid config
  (acceptance 1: `compute_units` maps to `ComputeUnitSpec`); systemlinkage good and bad; missing
  identity key; missing `exec` (error names the CompU and `exec`); procinfo typo under
  `compute_units.delay0.fps.procinfo` (names the field); string-form dataload refused; CompU
  without `.fps` valid (`fps == {}`, `procinfo is None`); CompU named `tasks` valid; the model is
  frozen.
- [ ] T012 [US1] Add to `tests/pipeline/toml_schemas_test.py` (SC-002): through
  `load_pipeline_config` on a `tmp_path` copy of `pipelinebasic-conf`, parametrize an extra
  top-level key (`foo`, `tasks`, `sessions`, `delay0`, `kind`) and assert `ConfigError` whose
  message contains the key and whose `path` is the TOML file; also a key left directly under a
  CompU table (`exec` alongside an old-shape key) is refused. Add loader-level cases for scalar
  and array procinfo values; assert `ConfigError`, a message identifying the offending field,
  the TOML file in `path`, and the underlying `pydantic.ValidationError` in `__cause__`.
- [ ] T013 [US1] Add to `tests/pipeline/toml_schemas_test.py` (SC-005): `PipelineConfigModel` and
  `LOOP_INFO` are absent from `milk.infra.pipeline_models`, and `PipelineConfig` is the pydantic
  model (`issubclass(…, BaseModel)`).
- [ ] T014 [US1] Run `pytest tests/pipeline/toml_schemas_test.py`; also load all three migrated
  configs via `load_pipeline_config` as a smoke check (FR-007, the example one by path).

**Checkpoint**: US1 is testable on its own.

---

## Phase 4: User Story 2 — FPS names from one place (P1)

**Goal**: `<name>_<cu>` defined once; every deploy, start, stop and check path calls it.

**Independent Test**: `tests/trivial/fps_name_test.py` and the updated
`tests/pipeline/deploy_pipeline_test.py` pass.

- [ ] T015 [P] [US2] Create `tests/trivial/fps_name_test.py` (py-safe, SC-003): `ast`-walk
  `milk/**/*.py`; exactly one `FunctionDef` named `fps_name`; no source contains
  `loop_number:03d`; `PipelineConfig(...).fps_name("delay0") == "basic_delay0"` for `name="basic"`.
  Supplement these tests with an explicit audit of all package sites constructing or requesting
  FPS names: deploy, start, stop and check paths use `PipelineConfig.fps_name`, and no site
  independently constructs `<name>_<cu>`. Record the audit result alongside the naming tests.
- [ ] T016 [US2] Update `tests/pipeline/deploy_pipeline_test.py` (milk): substitution assertions
  read `pp.compute_units["delay0"].fps[...]` instead of `pp.session_configs`; FPS loops and the
  tmux-kill loop use `pp.config.fps_name(cu)` instead of `get_session_abs_name`/`get_session`; after
  deploy, assert the live FPS names are exactly `{"basic_delay0", "basic_delay1"}` (US2
  acceptance 3). Keep `ComputeSession` use until Phase 5.
- [ ] T017 [US2] Grep `tests/` for hard-coded old FPS names (`_000`, `delay0_`, `delay1_`) and
  update them to `<name>_<cu>`.
- [ ] T018 [US2] Run `pytest tests/trivial/fps_name_test.py tests/pipeline` with `MILK_INSTALLDIR`;
  everything but the Phase 5 renames must be green.

**Checkpoint**: US1 and US2 pass. Vocabulary is still the old one.

---

## Phase 5: User Story 3 — "Session" means tmux only (P2)

**Goal**: no compute-unit-related "session" identifier, module, test helper, test file or help
text remains.

**Independent Test**: the "session" check in `tests/trivial/fps_name_test.py` and a grep of
`milk/` and `tests/` (SC-004).

- [ ] T019 [US3] `mv milk/session.py milk/compute_unit.py`; rename `ComputeSession` →
  `ComputeUnit`; `__str__`/`__repr__` print `ComputeUnit …` instead of `Session …`; rename the
  `session`-named locals and docstring words to `compu`/CompU. Update every import in `milk/`
  (`pipeline.py`, `infra/deploy_tasks.py`) and the local variables there (`session`, `sessions`,
  `session_name`, `s_name` → `compu`, `compus`, `cu`).
- [ ] T020 [P] [US3] `milk/infra/mains_pipeline_mgmt.py`: `milk-pipes nuke` help "on all sessions"
  → "on all compute units (CompUs)" (both occurrences); check every other subcommand help for the
  same concept. `milk/pipeline.py`: the "trylink sessions" comment likewise.
- [ ] T021 [P] [US3] `mv tests/session tests/compute_unit` (with `__init__.py`; delete stale
  `__pycache__/` under the old path); in `deployment_test.py`: `StreamDelayComputeSession` →
  `StreamDelayComputeUnit`, `func_session_cleanup` → `func_compute_unit_cleanup`, `session`
  locals → `compu`, imports from `milk.compute_unit`. Same import and local-name updates in
  `paramsync_test.py`, `stability_test.py`.
- [ ] T022 [P] [US3] `mv tests/infra/session_fixtures.py tests/infra/compute_unit_fixtures.py`;
  `session_cls` → `compu_cls`, `func_session_cleanup` → `func_compute_unit_cleanup`,
  `ComputeSession` → `ComputeUnit`.
- [ ] T023 [US3] (after T022) `mv tests/infra/session_deployment_test.py
  tests/infra/compute_unit_deployment_test.py` and `mv tests/infra/session_fps_runtime_sync_test.py
  tests/infra/compute_unit_fps_runtime_sync_test.py`; update their imports (`.compute_unit_fixtures`,
  `milk.compute_unit`), base classes (`StreamDelayCU`, `FPSTestCU` derive from `ComputeUnit`) and
  `session` locals/comments.
- [ ] T024 [P] [US3] `tests/trivial/errors_location_test.py`: `from milk import compute_unit`,
  `compute_unit.ComputeUnit("sh", "myfps")`. `tests/pipeline/deploy_pipeline_test.py`:
  `ComputeSession` → `ComputeUnit` import and use, `sname`/`sesh` → `cu`/`compu`.
  `tests/mains/experiment_deploy.py` already uses "session" for the tmux session named after the
  FPS: leave it.
- [ ] T025 [US3] Extend `tests/trivial/fps_name_test.py` (SC-004): `ast`-walk `milk/**/*.py`; no
  module file name, class, function, argument or attribute name contains `session`
  (case-insensitive), with an explicit tmux allowlist (empty in `milk/` today).
- [ ] T026 [US3] Grep `milk/` and `tests/` (excluding `__pycache__`) for "session" and "Session";
  every remaining hit concerns tmux, pytest scope, nox or `start_new_session`. Grep for
  `milk.session`, `ComputeSession`, `get_session`. `milk.egg-info/` is generated; do not edit.
- [ ] T027 [US3] Run `pytest` from `python/` with `MILK_INSTALLDIR` set (SC-001); compare with the
  T001 baseline.

**Checkpoint**: US1–US3 pass.

---

## Phase 6: User Story 4 — Glossary in the docs (P2)

**Goal**: design-v7 §0 published in `../docs/`, linked from the pipelines page; "CU" → "CompU".

**Independent Test**: file exists, link resolves, grep for "CU" finds no compute-unit meaning
(SC-006).

- [ ] T028 [P] [US4] Create `../docs/pipelines_glossary.md`: one-line header stating it is the
  published copy of design-v7 §0, then the §0 table with terms and definitions verbatim. Do not add
  it to the `mkdocs.yml` nav (consistent with `pipelines.md`). Do not edit `design-v7.md`.
- [ ] T029 [US4] `../docs/pipelines.md` (after T028): link the glossary near the top. Mermaid:
  `Compute Unit (CU)` → `Compute Unit (CompU)`, node ids `CUG`/`CU` → `CompUG`/`CompU`, label
  "CompU Orchestration". Text "CUs" → "CompUs". "CU sections (main and subs)" →
  "`[compute_units.<cu>]` tables (`exec`, `.fps`, `.fps.procinfo`)". Deployment paragraph: "all
  sessions" → "all CompUs", and state that FPS names are `<name>_<cu>`. Check the page's config
  examples against Appendix A.1 shapes; link, do not duplicate, design-v7.
- [ ] T030 [US4] Grep `../docs/` and `../AGENTS.md` for `\bCU\b`/`CUs`; none may mean compute
  unit (AGENTS.md is expected to have none). Compare the glossary against design-v7 §0 term by
  term.

**Checkpoint**: all four stories complete.

---

## Phase 7: Polish and closing (constitution VIII, IX)

- [ ] T032 Run the full `pytest` from `python/` with `MILK_INSTALLDIR`; SC-001 to SC-006 must hold.
- [ ] T033 From `python/`, in order: `git add .`, `pre-commit run`, `git restore --staged .`
  (constitution IX). A "files were modified by this hook" result needs no action; report any other
  hook failure to Vincent instead of looping.
- [ ] T031 (after T030, T032 and T033 succeed) `specs/pipelines/ROADMAP.md`, row R1b: set Status
  to `done` and Sub-spec to `specs/002-r1b-model-vocabulary/` (edit only those two cells).

---

## Dependencies & Execution Order

- Phase 1 → Phase 2 → Phase 3 and Phase 4 → Phase 5 → Phase 6 → Phase 7. Phases 3 and 4 touch
  `toml_schemas_test.py` and `deploy_pipeline_test.py` respectively, so they can run in parallel
  once Phase 2 is done; Phase 5 renames the files they edit, so it waits for both.
- Phase 6 depends on nothing in Phases 3–5 except the final vocabulary; it can start after Phase 2.
- File chains: `pipeline_models.py` T002 → T003; `toml_manipulation.py` T004 after T002;
  `deploy_tasks.py` T006 → T019; `pipeline.py` T005 → T019 → T020;
  `deploy_pipeline_test.py` T016 → T024; `fps_name_test.py` T015 → T025;
  `compute_unit_fixtures.py` T022 → T023.
- T010 waits for T002–T006; T026 for T019–T025.
- Closing order: documentation checks T030 → final pytest validation T032 → closing pre-commit
  gate T033 → completion metadata T031.

### Parallel opportunities

- T005 ∥ T006 ∥ T007 ∥ T008 ∥ T009 (different files, after T002/T003 for the code ones).
- T015 ∥ T011 (different files).
- T020 ∥ T021 ∥ T022 ∥ T024.

## Implementation Strategy

MVP = Phases 1–3 (one validated config, closed key set). Phase 4 adds the single FPS-name function
and its tests, Phase 5 the vocabulary rename, Phase 6 the glossary. The slice ships in one pass.
Nothing in the out-of-scope list is implemented (includes, kinds, hooks, plans, overrides,
substitution changes, the `Pipeline.compute_units` view property, rundir claim, bugs other than B2).
The cacao example `../plugins/cacao-src/examples/dmcomb-conf/conf.toml` is not migrated (other
repository).
