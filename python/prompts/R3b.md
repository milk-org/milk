# R3b: Init, drift, resync

Feature directory: `specs/006-r3b-init-drift-resync` · Tests: py (run check injected)

## /speckit-specify

```text
Feature directory: specs/006-r3b-init-drift-resync
Parent roadmap: specs/pipelines/ROADMAP.md → entry R3b. Design: specs/pipelines/design-v7.md (§0 init/rootdir/drift/resync/provisioning, §4.0 table + rules 1, 2, 5, 6 + the provisioning/resync diagram, D-7, D-16, D-22) and design-v7-appendix-a.md §A.3, §A.5.

Build an authoritative rootdir from a confdir, report drift between them, and apply selected changes on request.

Why: after init, the rootdir is the long-lived, authoritative home of a pipeline. Operators need to see what changed in the confdir or locally in the rootdir, and to choose what to bring over.

Requirements:
- init creates <name>-rootdir/ from a confdir. It consolidates if any hook is pending, copies the conf file artifacts into conf/ and the hook products and tagfiles into initdata/, and writes provenance.toml (A.3: source confdir, overrides used, milk package version, per-artifact fingerprints). The rootdir's config must then resolve successfully. init refuses if the rootdir exists. It is not atomic: on failure it reports and leaves what it wrote.
- Overrides given to init or resync (--set KEY=VAL) are recorded in provenance and replayed on every rootdir resolve, so a rootdir resolves identically from any shell.
- Hand edits in rootdir/conf/ take effect immediately and are reported as drift.
- drift reports, per artifact: (a) the confdir moved on, (b) a local edit in the rootdir, (c) a stale fetch. It also gives a key-level diff of the resolved config (confdir vs rootdir) and flags structural changes (the compute-unit set, an exec or an FPS name changed).
- resync applies selected drifted artifacts from the confdir to the rootdir and updates provenance. It refuses entirely if any compute unit's run process is active, or if an identity key would change. It refuses each locally modified artifact unless --overwrite-local is given, and such a refusal does not abort the other artifacts. It returns a ResyncReport (applied, refused, resolved diff, structural yes/no). It never triggers teardown or deploy.
- milk-pipeinit gains init, drift and resync. resync always prints a strong banner that a teardown → deploy may be needed, listing the structural changes. The old `milk-pipes init` is removed and its tests are migrated.
- The existing operating tasks (spool) read the new rootdir layout (conf/, initdata/), so deploying still works until R4 replaces them.

Out of scope: the rundir, plans and teardown (R4); atomicity; locks.

Success criteria: the suite is green; pure-Python tests cover init (layout, provenance content, refusal), override replay, the three drift comparisons and stale status, the structural flag, and every resync refusal path, with the run check injected.
```

## /speckit-plan

```text
Python ≥3.11, click, tomli/tomli_w; no new dependencies. Extend milk/provision/ (init.py, drift.py, resync.py) and its cli.py.
Copying is generic over ConfArtifact. provenance.toml is read and written in one place.
For the run check, resync calls Pipeline.any_run_active() (R1c) through an injectable parameter so py tests can stub it.
confresolve (R2) gains rootdir support: it reads rootdir/conf/ and replays the provenance overrides.
Minimal adaptation of infra/deploy_tasks.py to the new layout (data loads from initdata/). Don't redesign the tasks; R4 does.
Tests: py only.
```

## /speckit-implement

Three phases: (1) init and provenance; (2) drift; (3) resync and the CLI.

## Done when

- The suite is green, ROADMAP R3b is done, and `../docs/` describes init, drift and resync, linking to the design's provisioning/resync diagram.
