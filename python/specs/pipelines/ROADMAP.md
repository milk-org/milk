# Roadmap: `python/milk` pipeline layer (round 1: deploy lifecycle)

This roadmap breaks the frozen design `specs/pipelines/design-v7.md` into slices that Spec Kit
specifies one at a time. Each slice is one feature directory. Its `spec.md` Input line cites the
slice ID.

Rules:
- Agents may edit only the **Status** and **Sub-spec** columns.
- IDs are immutable.
- Scope changes go through a design amendment (v8), agreed by Vincent.

**Status legend**: planned · in-progress · done · deferred
**Tests**: **py** = pure Python (tmp dirs, no milk binaries) · **milk** = needs milk binaries, tmux and SHM through the existing conftest spoofing (`tests/conftestaux/milk.py`)

| ID | Sub-feature | Intent | Scope boundary (in / deferred) | Depends on | Tests | Status | Sub-spec |
|----|-------------|--------|--------------------------------|------------|-------|--------|----------|
| R1a | Import safety + errors | The package imports without a milk environment; one exception contract | In: `milk.env`, lazy `cliwrap` (keeps `HAVE_CLI`), `milk/errors.py` replaces every scattered class (B3, X8). Deferred: CLI pretty-printing; renames (R1b); `nuke` (R1c) | — | py | done | `specs/001-r1a-import-safety/` |
| R1b | One model + vocabulary | One config model; "compute unit" replaces "session"; Appendix A.1 shapes | In: single `PipelineConfig`/`ComputeUnitSpec` (X1); nested `[compute_units]`, table-only `[[dataloads]]`, `systemlinkage`; closed top-level key set (implemented keys only; this closes B2); FPS name `<short>_<cu>` in one place (X2); rename everywhere incl. tests, CLI help, examples (X9); glossary moved into `docs/`, CU → CompU in docs/AGENTS.md. Deferred: includes/kind/overrides (R2) | R1a | milk | done | `specs/002-r1b-model-vocabulary/` |
| R1c | CompU views + fixes | CompUs obtained one way; known bugs fixed | In: `Pipeline.compute_units` property (X3); `any_run_active()`; fixes B1, B4, B5, B6, B7 (B2 and B3 are closed by R1b and R1a). Deferred: state model (R4) | R1b | milk | done | `specs/003-r1c-compu-views-fixes/brief.md` |
| R2 | confresolve | The only way to obtain a config | In: kind-defaults layer (`generic` only), confdir-relative `includes`, API overrides (identity keys refused), any-key substitution except inside table arrays (D-26), validation, works on a confdir or a rootdir; no cache. Deferred: override replay from provenance (R3b); real kinds (R5) | R1c | py | done | `specs/004-r2-confresolve/brief.md` |
| R3a | Conf artifacts & consolidate | Track configuration files and fetched products | In: `ConfArtifact` (conf file + hook artifacts, fingerprints, `stale`), tagfile (A.2), `ConfHook` + the shared registry, `fetch`/`command` hook types, clone, consolidate; `milk-pipeinit list/clone/consolidate/resolve`; old `milk-pipes clone` removed. Deferred: init/drift/resync (R3b) | R2 | py | done | `specs/007-r3a-artifacts-consolidate/brief.md` |
| R3b | Init, drift & resync | Build and maintain an authoritative rootdir | In: init (`conf/`, `initdata/`, `provenance.toml` A.3, recorded overrides replayed on rootdir resolve), drift (3 comparisons, statuses, structural flag, resolved diff), resync (refusals, `ResyncReport`, CLI banner); `milk-pipeinit init/drift/resync`; old `milk-pipes init` removed; existing operating tasks read the new rootdir layout so `spool` keeps working. Deferred: rundir/plans (R4) | R3a | py (run check injected) | planned | — |
| R4 | Rootdir operations | Operate a pipeline from its rootdir | In: rundir (A.4) with claim + FPS list + uniqueness; cold/inconsistent model; per-plan consistency check; `state()`; `Task` protocol with CompU subsets; plans as data with `on_failure`; plan runner; plans deploy/start/stop/teardown; `milk-pipes` with plan subcommands + `state`/`link`/`unlink`; FPS workdir `rootdir/work/`; linkage pairs in the rundir; `plans` TOML key; old `spool/sys/nuke` removed. Deferred: kinds (R5) | R3b | milk | planned | — |
| R5 | Kind infrastructure | Extend pipelines without touching the core | In: `PipelineKind` protocol (§4.3), registration, `generic` kind, kind defaults feeding confresolve, kind plans as subcommands, one stub kind defined outside the core package. Deferred: standard-kind content (Vincent, manually) | R4 | milk | planned | — |
| R6 | pyMilk boundary | One narrow, tested seam to pyMilk | In (repo-merge round): narrow pyMilk seam, pure-Python SHM-name validator, ISIO/FPS struct version checks, engine-lib and bindings-build consolidation. Deferred now: requires repo architecture and C changes | R1a | milk | deferred | `specs/006-r6-pymilk-boundary/brief.md` (draft) |
| R7 | Waitable test states | Tests wait on observable states, not guessed sleeps | In: replace guessed sleeps in CompU tests with predicate polls; dedupe CompU fixtures into conftestaux. Deferred: library wait primitive (R4); cli_hardening pacing | R1c | milk | done | `specs/005-r7-waitable-tests/brief.md` |

Order: R1a → R1b → R1c → R2 → R3a → R3b → R4 → R5. R6 is deferred to the repo-merge round. R7 can run any time after R1c.

**Not to be specified in this round** (do not implement or invent): the CLI top-level error
handler; concurrent long-running AO tasks; atomic init/resync; the confresolve cache; the
fingerprint stat cache; plan extensions; an undo protocol; standard-kind content;
connectivity check / graph-gen; the Configuration round; the Operations/aorun port; machine
configuration. Full list: design-v7 §5.
