# R3b brief: init, drift and resync

Roadmap: R3b · Design: design-v7 §0 (init, rootdir, drift, resync, provisioning), §4.0 (table,
rules 1, 2, 5, 6, provisioning/resync diagram), D-7, D-16, D-17, D-21, D-22; Appendix A.3, A.5 ·
Tests: py · Workflow: lean; prompts in `specs/pipelines/lean-prompts.md`. This brief and its
tests replace spec/plan/tasks, which overrides constitution §VIII for this slice.
**Important slice.** The decisions were settled by Vincent on 2026-10-27 (see "Decided").

## Goal
After `init`, the rootdir is the authoritative, self-contained home of a pipeline instance.
Operators can see what differs from the confdir (`drift`) and choose what to bring over
(`resync`). The operate layer (`Pipeline`, tasks, `milk-pipes`) reads the rootdir and never the
confdir.

## Rootdir layout (after init)
```
<name>-rootdir/
  conf/            conf_files(confdir), copied: *.toml outside data/, plus scripts/**
  initdata/        <product> and <product>.tag for each hook, copied from confdir/data/
  provenance.toml  A.3
  work/            FPS workdir (D-17), empty at init
  logs/            empty at init (D-22)
```
Artifact keys, used in provenance, drift and resync:
- conf files: their relpath, for example `conf.toml` or `scripts/a.sh`;
- hook products: `data/<product>`. The key maps to `confdir/data/<product>` on the source side
  and to `rootdir/initdata/<product>` on the rootdir side.

## Interface (the spec)

```python
# milk/config/layout.py (new): what a folder looks like, without resolving it
is_confdir(path) -> bool         # has conf.toml and no provenance.toml
is_rootdir(path) -> bool         # has provenance.toml and conf/conf.toml
list_pipeline_dirs(folder)       # moved from config/resolve.py; selects by sniffing, not by name suffix

# milk/config/resolve.py: confresolve gains rootdirs (D1)
confresolve(path, overrides=None)
#   If is_rootdir(path): resolve path/conf/ with {**recorded_overrides, **overrides};
#   config reads ONLY the [overrides] table of provenance.toml. Otherwise: as today.

# milk/provision/hooks.py: user-triggered refetch (D5)
consolidate(confdir, refetch: Collection[str] | Literal["all"] | None = None) -> list[str]
#   The listed hook ids (or all of them) run even when resolved. An unknown id raises ProvisionError.

# milk/provision/provenance.py: the only writer of provenance.toml
class ArtifactRecord(BaseModel): content_hash: str; size: int; hook_id: str | None; decl_digest: str | None
class Provenance(BaseModel): milk_version: str; source_confdir: str; initialized: datetime
                             last_resync: datetime | None; overrides: dict[str, object]
                             artifacts: dict[str, ArtifactRecord]
read_provenance(rootdir) -> Provenance
write_provenance(rootdir, prov)   # TOML, exclude_none

# milk/provision/rootdir.py
init(confdir, parent=None, overrides=None) -> Path      # parent defaults to confdir.parent
drift(rootdir, confdir=None) -> DriftReport             # confdir defaults to provenance.source_confdir
resync(rootdir, *, run_active: Callable[[], bool], artifacts=None, overrides=None,
       overwrite_local=False, confdir=None) -> ResyncReport

@dataclass(frozen=True) class ArtifactDrift:  key: str; source: str; local: str
@dataclass(frozen=True) class DriftReport:    artifacts: list[ArtifactDrift]
                                              resolved_diff: dict[str, tuple[object, object]]
                                              structural: list[str]
                                              milk_version: tuple[str, str] | None   # (recorded, installed) if they differ
@dataclass class ResyncReport:                applied: list[str]; refused: dict[str, str]
                                              resolved_diff: dict; structural: list[str]
```

### init
Checks happen before any write; the steps after them are not atomic (D-21).
1. Run `confresolve(confdir, overrides)`. This refuses identity overrides (ConfigError).
2. The target is `<parent>/<name>-rootdir`. If it exists, raise `ProvisionError` and point the
   user to resync.
3. Run `consolidate(confdir)`.
4. Copy the conf files to `conf/`, and the products and tags to `initdata/`.
5. Create `work/` and `logs/`.
6. Write provenance: `milk_version = importlib.metadata.version("milk")`, `source_confdir`
   resolved and absolute, `initialized`, `overrides`, and one record per artifact. A hook
   record is copied from its tag.
7. Run `confresolve(rootdir)` as the final check.

### drift: one row per key
The keys are the union of: the recorded keys, the confdir's conf files, the rootdir's
`conf/` files, and the confdir's currently declared hooks.

| Field | Compares | Statuses |
|-------|----------|----------|
| `source` | rootdir **recorded** vs **confdir current**; for hooks also the recorded `decl_digest` vs the digest of the confdir's **current declaration** (resolved, not its tag) | (a) moved on: `modified/added/removed`; `stale` = the hook **declaration** changed (cheap, already in R3a). Upstream content changes are **not** detected: the user runs `consolidate --refetch` (D4, D5) |
| `local` | rootdir **recorded** vs rootdir **current** (`conf/…` or `initdata/…`) | (b) local edit: `modified/added/removed` |

- **`resolved_diff`:** dotted keys (denested, `model_dump(mode="json", exclude_none=True)`).
  Each value is `(rootdir_value, confdir_value)`, with `None` when the key is absent. The
  confdir side is resolved **with the recorded overrides**, so overrides never show up as
  drift. A cosmetic TOML edit gives `source = modified` and an empty `resolved_diff`.
- **`structural`:** one human line per compute unit added or removed, or whose `exec` changed.
  FPS names derive from identity keys, which resync refuses anyway.
- **`milk_version`:** reported separately; it is never artifact drift.

### resync
Checks happen in this order:
1. `run_active()` → `ProvisionError` ("run process active"). Nothing is written.
2. Identity keys differ between `confresolve(rootdir)` and
   `confresolve(confdir, {**recorded, **overrides})` → `ProvisionError` ("init a new rootdir
   instead"). Nothing is written. Identity overrides raise ConfigError earlier.
3. Run `consolidate(confdir)` (confdir-local, safe).
4. For each row with `source != unchanged`, selected by `artifacts` (default: all), in sorted
   key order:
   - if `local != unchanged` and not `overwrite_local`: `refused[key]` = the reason, then
     continue (one refusal does not abort the others);
   - if `source == removed`: delete it on the rootdir side (with its tag, for a hook) and drop
     the record;
   - otherwise: copy it from the confdir (with its tag, for a hook) and record it (for a hook,
     from the tag).
5. `provenance.overrides = {**recorded, **overrides}`, set `last_resync`, and write.
6. `resolved_diff` and `structural` compare **rootdir before vs after**, in the same
   orientation.

resync never triggers teardown or deploy. The run check is injected: the CLI passes
`lambda: Pipeline(rootdir).any_run_active()`. That is the only import from provision into the
operate layer, and it sits in `provision/cli.py`.

### milk-pipeinit
- **`consolidate`** gains `--refetch ID` (repeatable) and `--refetch-all`.
- **New commands:** `init CONFDIR [--parent DIR] [--set K=V …]`; `drift ROOTDIR` (one line per
  drifted row, the resolved diff, then `STRUCTURAL:` lines); `resync ROOTDIR [--artifact KEY …]
  [--set K=V …] [--overwrite-local]`.
- **`resync` always prints the banner** "teardown → deploy may be needed", then the structural
  lines, even when nothing was applied.
- **`resolve`** accepts a rootdir; that works for free through confresolve.
- **`--set` values** are parsed as TOML values (`tomli.loads(f"v = {raw}")`); if that fails, the
  raw string is used.
- Errors propagate (no top-level handler).

### Operate layer on the new layout (minimal, R4 rewrites it)
- **Constructors (D2).** The main constructor is `Pipeline(target)`. If `target` is an existing
  path that `is_rootdir`, it is used as the rootdir. Otherwise `target` is a short name and the
  rootdir is `<cwd>/<target>-rootdir`. If neither is a rootdir, raise `ConfigError` naming both
  paths it tried.
- **Auxiliary constructors** (classmethods): `Pipeline.from_folder(rootdir)` and
  `Pipeline.from_parent(parent, short_name)`. They raise `ConfigError` when the folder is not a
  rootdir.
- **What `Pipeline` drops:** `parent_folder`, `conf_folder` and `log_folder`. `root_folder` is
  the rootdir, `config` is `confresolve(rootdir)`, and `work_folder` is `rootdir/work`.
- **`LoadDataFiles`** reads `rootdir/initdata/<source>`.
- **Delete `InitialFolderSetup` and `milk-pipes init`.** `init` replaces them, and drops the
  `fpsCTRL.log` link, `fpstmuxenv`, `aorun-*` and the sibling logdir (D-22).
- **`milk-pipes spool|sys|teardown|nuke TARGET [--parent DIR]`** replace `NAME` + `--folder`.
  `TARGET` follows `Pipeline(target)`. With `--parent`, the CLI uses
  `Pipeline.from_parent(DIR, TARGET)`. This is how instances outside cwd are reached.
- **teardown vs nuke (D3):**
  - `Pipeline.teardown()` is today's nuke minus the rm: runstop, deapply linkages, confstop,
    destroy FPSs, all best-effort with one aggregate error. It **keeps the rootdir and `work/`**,
    which will hold reusable data. R4's `teardown` plan supersedes it under the same name (and
    adds the rundir release), so the names do not conflict.
  - `Pipeline.nuke()` is `teardown()`, then `rm -rf` of the rootdir, even if teardown failed.
    The library never prompts.
  - `milk-pipes nuke` prompts the operator to **type the short name**; a mismatch aborts with
    nothing deleted. `--yes` skips the prompt for scripts.

### Changes to R3a code
- `status(None, None) == "unchanged"`: absent on both sides is not drift.
- `consolidate` therefore treats a **missing product** as pending: `current is None or status
  != unchanged`.

## Decided (Vincent, 2026-10-27)
- **D1.** Rootdir resolve lives in `confresolve`, detected by `is_rootdir`; the config layer
  reads only `[overrides]`. Operate resolves a rootdir without importing provision.
- **D2.** The main constructor is `Pipeline(target)`, taking a rootdir path or a short name in
  cwd, with `from_folder` and `from_parent` as auxiliary constructors. A sniff test
  (`config/layout.py`) tells confdirs and rootdirs apart without resolving them, and drives
  listing. **Deferred:** a global scan of rundirs to find configured pipelines (LOOSE_ENDS, R4+).
- **D3.** `teardown` deconfigures and keeps the rootdir and `work/`. `nuke` is teardown plus
  deleting the rootdir, behind a type-the-short-name prompt (`--yes` to skip).
- **D4.** Two statuses per row. Only declaration-level `stale` is detected; upstream content
  changes are not (for example, a hook that downloads from the network).
- **D5.** `consolidate --refetch ID` (repeatable) and `--refetch-all` are the user's tool for
  upstream changes.
- **D6.** Everything is validated before the first write; after that, resync is not atomic
  (D-21). A partial state is visible through `drift`.

## Tests
- **New:**
  - `tests/provision/rootdir_test.py`: 40 tests, pure Python, needs `rsync` and `sh`. It is
    parked with `pytest.importorskip("milk.provision.rootdir")`. **P1 deletes that line first.**
  - `tests/conftestaux/provision.py`: `make_confdir`, `fetch_hook`, and the `src_file`
    fixture. It is already written and registered in `tests/conftest.py`.
- **Allowed migrations of existing tests:**

| File | Change |
|------|--------|
| `tests/provision/provision_test.py` | delete its local `_confdir`, `_fetch_hook` and `src_file`; use `tests.conftestaux.provision`. In `test_cli_list`, `two-rootdir` must be a real rootdir (listing now sniffs) |
| `tests/conftestaux/pipeline.py` | `make_tmp_pipeline` copies the conf, runs `init`, returns `Pipeline(rootdir)`. `cloned_pipeline` runs `clone` then `init` into `AOloop/` and returns `Pipeline(rootdir)`; its teardown removes the confdir and the rootdir |
| `tests/pipeline/deploy_pipeline_test.py` | the `pipeline` fixture inits pipelinebasic into `tmp_path`. Remove the `parent_folder` assertions and the `InitialFolderSetup` steps. `TestMilkPipes.test_no_pipelines` goes (it tested `milk-pipes init`) |
| `tests/pipeline/system_linkage_test.py` | the fixture inits into `tmp_path` (no `root_folder` patching); `test_cli_sys` calls `sys TARGET LINKAGE` |
| `tests/pipeline/compu_views_test.py` | B4 test deleted (its task is gone; init refuses an existing rootdir, which is tested in R3b). B5 writes to `initdata/`. B7 is unchanged: `nuke()` still attempts every step and removes the rootdir |
| `tests/pipeline/confresolve_test.py` | `test_pipeline_config_comes_from_confresolve` inits first |

## Not doing
Rundir, plans, the teardown **plan** and `state()` (R4). Moving `pipeline.py` and `infra/` into
`operate/` (R4). Atomicity and locks (D-20, D-21). A fingerprint cache. Detecting upstream
content changes. A global rundir scan. Git in the rootdir (Configuration round).

## Done when
- The suite is green: the 40 R3b tests, the 32 R3a tests, and the migrated milk tests.
- Budget: net +250 LOC in `milk/`. A throwaway reference came to about +145 net
  (+385/−242, including the milk-pipes CLI rewrite).
- ROADMAP R3b is `done`. `docs/` gains:
  - the rootdir layout;
  - the drift table above;
  - the resync checks;
  - the new `milk-pipeinit` commands and `--refetch`;
  - the `milk-pipes TARGET [--parent]` convention, and teardown vs nuke;
  - one paragraph: "the rootdir is authoritative; resync is the only way confdir changes
    reach it".
