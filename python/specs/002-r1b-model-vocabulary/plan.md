# Implementation Plan: One config model + compute-unit vocabulary

**Spec**: [spec.md](./spec.md) · **Roadmap slice**: R1b · **Tests**: milk

**Design**: [design-v7.md](../pipelines/design-v7.md) §0, §4.4, D-15, D-19; defects B2, X1, X2, X9 ·
[design-v7-appendix-a.md](../pipelines/design-v7-appendix-a.md) §A.1

## Technical Context

- Python ≥3.11, pydantic 2, tomli. No new dependencies. Package root `milk/`, tests in `tests/`
  (`pytest` from `python/`; milk-tagged tests need `MILK_INSTALLDIR` and the existing conftest
  spoofing).
- Today the config exists three times (X1): `PipelineConfigModel` (pydantic, `extra="allow"`), a
  plain `PipelineConfig` filled field by field in `load_pipeline_config`, and the unused
  `LOOP_INFO` dataclass. Per-CompU tables are top-level keys found through `[sessions]`, which is
  why unknown keys pass silently (B2).
- The FPS name `f"{cu}_{loop_number:03d}"` is written in `Pipeline.get_session_abs_name` and 7 more
  times in `infra/deploy_tasks.py` (X2).
- Per-CompU params, including the `procinfo` sub-table, are denested and written verbatim as FPS
  params by `ConfigureFPSs` (`procinfo.enabled`, …). This behaviour is kept.
- `{var}` substitution (`substitute_toml_variables_in_nested`) flattens nested tables only. List
  values are substituted per string item; tables inside lists are left untouched. So
  `[[dataloads]]` values are already not substituted, which matches D-26 (see Flags).
- `docs/pipelines.md` is not in the `mkdocs.yml` nav; it is reached by links only.
- `AGENTS.md` contains no "CU" abbreviation today; only `docs/pipelines.md` does.

## Flags

1. **Dataloads substitution stays unchanged.** Values inside table arrays pass through untouched,
   consistent with D-26 and FR-009. The migrated example writes the dataload target literally
   (`aol1_simu_modes`).
2. **Ad hoc `ComputeUnit` construction stays** (X3). Call sites build
   `ComputeUnit(spec.exec, config.fps_name(cu))` directly until R1c adds the
   `Pipeline.compute_units` view property. No interim helper is added.
3. **Out-of-repo config**: `../plugins/cacao-src/examples/dmcomb-conf/conf.toml` (cacao repo) keeps
   the old shape and will be refused with `ConfigError` until migrated there.

## Model: `milk/infra/pipeline_models.py`

Delete `LOOP_INFO`, `PipelineConfigModel`, the plain `PipelineConfig`, and
`DataLoadEntryModel.from_str` with its `before` validator. Keep `ProcessInfoData` unchanged.

```text
ProcessInfoData      extra="forbid"            (unchanged)
DataLoadEntryModel   extra="forbid"            source: str, target: str   (table-only)
ComputeUnitSpec      extra="forbid", frozen    exec: str
                                               fps: dict[str, Any] = {}
                                               procinfo (property) -> ProcessInfoData | None
PipelineConfig       extra="forbid", frozen    long_name: str, name: str, loop_number: int
                                               compute_units: dict[str, ComputeUnitSpec] = {}
                                               dataloads: list[DataLoadEntryModel] = []
                                               systemlinkage: dict[str, dict[str, str]] = {}
                                               fps_name(cu: str) -> str
```

- `ComputeUnitSpec.fps` is the raw `.fps` table, including its `procinfo` sub-table, so deploy
  keeps writing it verbatim. A field validator runs
  `ProcessInfoData.model_validate(fps["procinfo"])` when the key is present (unknown fields and
  non-table values refused with `pydantic.ValidationError`). The `procinfo` property uses the same
  model-validation API when the key is present, otherwise returning `None`. One data source, no
  stored copy.
- `PipelineConfig.fps_name(cu)` returns `f"{self.name}_{cu}"` (D-19). It is the only place the
  expression exists. `Pipeline` and the tasks call it through `config`.
- `_check_systemlinkage` moves onto the new `PipelineConfig` unchanged.
- `extra="forbid"` on `PipelineConfig` closes the top-level set to the six implemented keys
  (FR-003); `kind`, `includes`, `hooks`, `plans` are refused like any other key.

## Loader: `milk/infra/toml_manipulation.py`

`load_pipeline_config` keeps its signature and role (R2 replaces it with confresolve):

1. Missing file → `ConfigError(path=…)` (unchanged).
2. `tomli.load`, then the unchanged `substitute_toml_variables_in_nested`.
3. `PipelineConfig.model_validate(data)`; `pydantic.ValidationError` is re-raised as
   `ConfigError(str(err), path=toml_path) from err`. Pydantic's message carries the `loc`
   (for example `tasks` or `compute_units.delay0.exec`), which names the key, CompU or field
   (FR-004). Loader-level tests cover scalar and array procinfo values and assert the offending
   field, TOML path context and underlying `pydantic.ValidationError` cause.

The field-copying block and the `session_configs` construction are deleted.

## Call sites (phase 1)

- **`pipeline.py`** `__init__`: `self.compute_units = self.config.compute_units` (spec dict; R1c
  turns it into the view property). Remove `self.sessions` and `self.session_configs`. Keep
  `dataloads`, `systemlinkage`, `short_name`, `loop_number` as they are.
  `get_session_abs_name` and `get_session` are deleted. `is_running` and `nuke` build
  `ComputeUnit(spec.exec, self.config.fps_name(cu))`.
- **`infra/deploy_tasks.py`**: every `ComputeSession(exec, s + f"_{p.loop_number:03d}")` becomes
  `ComputeUnit(spec.exec, p.config.fps_name(cu))`, iterating `p.compute_units.items()`.
  `ConfigureFPSs` denests `spec.fps` instead of `p.session_configs[…]`. Error messages use
  `fps_name`.

## Config migration (phase 1)

| File | Changes |
|------|---------|
| `tests/resources/pipelinebasic-conf/conf.toml` | Drop `tasks` and `[sessions]`. `[delay0]`/`[delay1]` → `[compute_units.<cu>]` with `exec` + `.fps` + `.fps.procinfo` tables (same values). Refs → `{compute_units.delay0.fps.in_name}` / `.out_name`, including `systemlinkage.shared.middle`. |
| `tests/resources/simulator-rootdir/conf.toml` | Drop `tasks`, `[sessions]` and the empty tables; three `[compute_units.<cu>]` tables with `exec` only. |
| `../milk-examples/pipelines/aolinearsimulator-conf/conf.toml` | Drop `tasks`, `_ln`, `[sessions]`. CompU tables nested as above (`option.MODENORM` stays a dotted key under `.fps`). `immodes = 'aol{loop_number}_simu_modes'`. Refs `{mvm.outcoeff}` → `{compute_units.mvm.fps.outcoeff}`. Dataloads → `[[dataloads]] source = "sim_matrix_flat1.fits"`, `target = "aol1_simu_modes"` (literal, Flag 1). |

## Vocabulary rename (phase 2)

Files are moved with plain `mv` (no git commands, per constitution IX). Stale `__pycache__/`
directories left under the old test paths are deleted.

| Old | New |
|-----|-----|
| `milk/session.py` | `milk/compute_unit.py` |
| `ComputeSession` | `ComputeUnit`; `__str__`/`__repr__` print `ComputeUnit …` instead of `Session …` |
| `Pipeline.get_session_abs_name`, `Pipeline.get_session` | `PipelineConfig.fps_name` (phase 1) |
| `Pipeline.sessions`, `Pipeline.session_configs` | `Pipeline.compute_units` (phase 1) |
| `tests/session/` (`deployment_test.py`, `paramsync_test.py`, `stability_test.py`, `__init__.py`) | `tests/compute_unit/` |
| `tests/infra/session_fixtures.py` | `tests/infra/compute_unit_fixtures.py` |
| `tests/infra/session_deployment_test.py` | `tests/infra/compute_unit_deployment_test.py` |
| `tests/infra/session_fps_runtime_sync_test.py` | `tests/infra/compute_unit_fps_runtime_sync_test.py` |
| `StreamDelayComputeSession` | `StreamDelayComputeUnit` |
| `func_session_cleanup`, `session_cls`, local `session`/`sesh`/`sname` vars and comments | `func_compute_unit_cleanup`, `compu_cls`, `compu`/`cu` |

- `milk-pipes nuke` help (`mains_pipeline_mgmt.py`): "on all sessions" → "on all compute units
  (CompUs)". `pipeline.py` comment "trylink sessions" likewise.
- `tests/trivial/errors_location_test.py`: `from milk import compute_unit`, `ComputeUnit(...)`.
- Kept, because they mean tmux or pytest/nox sessions: pytest `scope="session"`, nox,
  `start_new_session`, libtmux `session`/`session_name`, the tmux-kill loop in
  `deploy_pipeline_test.py` (now keyed by `fps_name`).
- `milk.egg-info/SOURCES.txt` is generated and not edited.

## Tests (milk)

- **`tests/pipeline/toml_schemas_test.py`** (rewritten on `PipelineConfig`):
  - Direct-model checks keep expecting `pydantic.ValidationError`: valid config, systemlinkage
    good/bad, missing identity key, missing `exec`, procinfo typo under
    `compute_units.delay0.fps.procinfo`, string-form dataload refused, CompU without `.fps` valid
    (`fps == {}`, `procinfo is None`), CompU named `tasks` valid.
  - Through `load_pipeline_config` on a `tmp_path` copy of `pipelinebasic-conf` (SC-002):
    parametrized extra top-level key (`foo`, `tasks`, `sessions`, `delay0`, `kind`) raises
    `ConfigError` whose message contains the key and whose `path` is the TOML file.
  - `fps_name("delay0") == "basic_delay0"`.
- **New `tests/trivial/fps_name_test.py`** (py-safe, SC-003, SC-004): `ast`-walk `milk/**/*.py`;
  exactly one `FunctionDef` named `fps_name`; no source contains `loop_number:03d`; no module,
  class, function or attribute name contains `session` (case-insensitive) except a tmux allowlist
  (empty in `milk/` today).
- Supplement the naming tests with an explicit audit of all package sites constructing or
  requesting FPS names: deploy, start, stop and check paths use `PipelineConfig.fps_name`, and no
  site independently constructs `<name>_<cu>`. Record the result alongside the naming tests.
- **`tests/pipeline/deploy_pipeline_test.py`**: substitution assertions read
  `pp.compute_units["delay0"].fps[...]`; FPS loops use `pp.config.fps_name(cu)`; after deploy,
  assert the FPS names are exactly `{"basic_delay0", "basic_delay1"}` (US2.3).
- Renamed test files only change names as in the table above.
- **Run**: `pytest` from `python/` with `MILK_INSTALLDIR` set; the whole suite must be green.

## Docs & roadmap (closing step)

- **New `../docs/pipelines_glossary.md`**: the design-v7 §0 table, terms and definitions verbatim,
  under a one-line header stating it is the published copy of design-v7 §0. Not added to the
  `mkdocs.yml` nav, consistent with `pipelines.md`.
- **`../docs/pipelines.md`**: link the glossary near the top. Mermaid: `Compute Unit (CU)` →
  `Compute Unit (CompU)`, node ids `CUG`/`CU` → `CompUG`/`CompU`, label "CompU Orchestration". Text
  "CUs" → "CompUs". "CU sections (main and subs)" → "`[compute_units.<cu>]` tables (`exec`, `.fps`,
  `.fps.procinfo`)". Deployment paragraph: "all sessions" → "all CompUs"; mention FPS names
  `<name>_<cu>`.
- **`../AGENTS.md`**: no "CU" hit; no change.
- **`specs/pipelines/ROADMAP.md`** R1b: Status `done`, Sub-spec `specs/002-r1b-model-vocabulary/`.
  Update only after documentation checks, final pytest validation and the closing pre-commit gate
  succeed.
- End of each implement phase: `git add .`, `pre-commit run`, `git restore --staged .`
  (constitution IX).
