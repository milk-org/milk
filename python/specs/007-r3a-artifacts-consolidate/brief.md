# R3a brief: conf artifacts, conf hooks, clone, consolidate

Roadmap: R3a · Design: design-v7 §0 (confdir, clone, conf hook, conf artifact, fingerprint,
tagfile, consolidate), §4.0, §4.9, D-6, D-8; Appendix A.1 `[[hooks]]`, A.2, A.5 · Tests: py
Workflow: lean; prompts in `specs/pipelines/lean-prompts.md`. This brief and its tests replace
spec/plan/tasks, which overrides constitution §VIII for this slice. Run it after R7 lands.

## Goal
A confdir can be cloned cleanly and can fetch or generate its data products. Every conf file
and product has a fingerprint, so that R3b can do init, drift and resync on top.

## Package layout (start here)
`milk/infra/` is a grab bag. R3a starts splitting the package by responsibility, with a
one-way dependency rule:

```
milk/
  errors.py  env.py  cliwrap.py      shared; cliwrap is the milk-cli test harness, unrelated
  config/                            what a pipeline IS: pure data, no I/O beyond reading TOML
    model.py        ← infra/pipeline_models.py   (+ hook declarations, below)
    substitute.py   ← infra/toml_manipulation.py
    resolve.py      ← confresolve.py
  provision/                         milk-pipeinit: file operations, never the live system
    artifacts.py    fingerprints, conf files, status
    hooks.py        tagfiles, hook runners, consolidate
    clone.py
    cli.py          milk-pipeinit
  operate/  (R4)                     milk-pipes: SHM, FPS, tmux, processes
    pipeline.py, compute_unit.py, tasks.py, cli.py   ← the rest of infra/; infra/ disappears
```

Dependencies: `config` ← `provision` and `config` ← `operate`. `provision` and `operate`
never import each other. In R3b, resync's run check reaches `operate` only through a callable
that its CLI injects. This is enforced in review, not by tests.

**Step 0 of P1 is a pure move.** Use `git mv` for the three `config/` files, then fix the imports
everywhere, tests included. No shims and no re-export modules left at the old paths. The suite
must be green before any new code is written. Report the move's diff separately; it does not
count against the budget. `milk/config/__init__.py` exports `confresolve` and `PipelineConfig`.

## Interface (the spec)

```python
# milk/errors.py
class ProvisionError(MilkError): ...

# milk/config/model.py: hook declarations are config data, validated at load (closed key set gains `hooks`)
class FetchHookDecl(BaseModel):   id: str; type: Literal["fetch"];   product: str; source: str
class CommandHookDecl(BaseModel): id: str; type: Literal["command"]; product: str; command: list[str]
# Both are frozen with extra="forbid". The union is discriminated on `type`.
# product: a plain name under data/ (no "/", no "..", must not end in ".tag").
# PipelineConfig.hooks: list[...]. Ids and products must be unique.

# milk/provision/__init__.py exports exactly:
Fingerprint            # frozen dataclass(content_hash: "sha256:<hex>", size: int)
fingerprint(path)      # file or directory; ProvisionError on any symlink
conf_files(confdir)    # sorted relpaths: **/*.toml outside data/, plus every file under scripts/
decl_digest(decl)      # "sha256:" + sha256(json.dumps(decl, sort_keys=True, default=str))
status(recorded, current, recorded_decl=None, current_decl=None)
                       # -> "unchanged" | "modified" | "stale" | "added" | "removed"
consolidate(confdir)   # runs the pending hooks; returns their ids, in declaration order
read_tag(product)      # the A.2 dict, or None
clone(src, dest_parent)  # -> dest_parent / src.name

# milk/provision/cli.py: `milk-pipeinit` console script (pyproject)
#   list [DIR] | clone SRC [DEST_PARENT] | consolidate CONFDIR | resolve CONFDIR
```

Rules:
- **Directory fingerprint:** sha256 over the sorted lines `relpath\0hash\n`, with each empty
  directory as `relpath/\0\n`; size is the total bytes. Modes and mtime are ignored.
- **`status` precedence:** no current → `removed`; no recorded → `added`; the bytes differ →
  `modified`; the declaration differs → `stale`; otherwise `unchanged`. It is a pure function;
  R3b reuses it with provenance records.
- **A hook is pending** when its status is not `unchanged`. To consolidate a pending hook:
  remove the old product, run it, check that the product exists (`ProvisionError` naming the
  hook id otherwise), fingerprint it (which refuses symlinks), then write `<product>.tag` (A.2,
  TOML). Hook declarations come from `confresolve(confdir).hooks`.
- **Hook runners:** a plain dict `{"fetch": fn, "command": fn}`.
  - `fetch` runs `rsync -a --delete <source>[/] data/<product>`. A relative source is relative
    to the confdir; an absolute path or an rsync URL is used as given.
  - `command` runs the argv list with cwd `data/`, and must create `<product>` there.
- **clone** copies `conf_files(src)` only, writes `.gitignore` = `data/`, runs no hooks and no
  git, and refuses an existing target before writing anything.
- **resolve** prints `tomli_w.dumps(config.model_dump(mode="json", exclude_none=True))`.
- **Errors propagate.** The CLI does not catch `MilkError`; the top-level handler is out of scope.

## Removed
- From `Pipeline`: `clone_to`, `update_local_git`, `check_clone_target`, `_git`,
  `_has_local_git`, `_CLONE_LOG`, and the git-recording behaviour of clone (D-8 and A.5: clone
  makes no git commit).
- `milk-pipes clone`, `milk_pipes_clone_function`, and the `-m/-c` example lookup. The examples
  are reached with `milk-pipeinit list $MILK_ROOT/milk-examples/pipelines`.
- `browse_and_print_conf_folders`: `milk-pipes init` and `spool` with no name now call the same
  listing as `milk-pipeinit list`. One listing function.

## Decisions for Vincent
1. **The `config/` move happens in R3a, as step 0;** `operate/` waits for R4, which rewrites
   those files anyway. Alternatively, defer the whole layout to R4 and put the new code at
   `milk/provision/` only.
2. **No `ConfHook` / `ConfArtifact` classes or protocols yet.** Declarations are pydantic
   models, runners are two functions in a dict, and artifacts are fingerprint functions plus
   `status`. That is the two-caller rule; design §4.9's protocols wait until R3b's
   drift/copy loop (artifacts) or R5's kinds (hooks) give them a second real caller.
3. **No generic `milk/registry.py` yet,** for the same reason. The hook-runner dict is the only
   registry. R4 (tasks) creates the shared mechanism and migrates the dict.
4. **One product per hook** (A.1 shows a single `product`).
5. **Conf files are all `*.toml` outside `data/`, plus `scripts/**`,** rather than conf.toml
   plus its declared includes. It's simpler, and clone and drift use the same set.

## Tests
- New: `tests/provision/__init__.py` (empty) and `tests/provision/provision_test.py` (32 tests,
  pure Python; needs `rsync` and `sh`).
- Allowed edits to existing tests:
  - Import paths (step 0).
  - `tests/conftestaux/pipeline.py` `cloned_pipeline`: replace `.clone_to("./AOloop")` with
    `clone(RESOURCES / "pipelinebasic-conf", Path("AOloop"))` plus
    `Pipeline("AOloop", "pipelinebasic")`.
- `tests/mains/` (ignored by pytest) still calls `clone_to`. Leave it and add a line to LOOSE_ENDS.

## Not doing
init, drift, resync and provenance (R3b). Any git operation. Caching fingerprints. rsync URL
tests (only local paths are tested). Rootdir support in `resolve` (R3b).

## Done when
- The suite is green, including the 32 R3a tests.
- Budget: net +100 LOC in `milk/`, step 0 excluded. A throwaway reference came to about +15
  net: +200 new, −190 for the removed git/clone code.
- ROADMAP R3a is `done`. `docs/` gains the confdir layout (`conf.toml`, includes, `scripts/`,
  `data/` + tags), the four `milk-pipeinit` commands, and the package layout above with its
  dependency rule.
