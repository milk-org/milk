# Pipeline configuration files

How a confdir is laid out, what `conf.toml` can say, how the files are resolved into one
configuration, and how changes reach a rootdir. The entry point is
[Pipelines](pipelines.md); the terms are in the [glossary](pipelines_glossary.md).

## The confdir

```
<long_name>-conf/
  conf.toml          the pipeline description (required)
  *.toml             included files, listed in conf.toml `includes = [...]`
  scripts/           any files, copied as-is; typically what `command` hooks run
  data/              hook products and their <product>.tag files; git-ignored, never cloned
  .gitignore         `data/`, written by clone
```

The **conf files** of a confdir are every `*.toml` outside `data/` and every file under
`scripts/`. They are what `clone` copies, what `init` copies into `rootdir/conf/`, and what
drift is computed on, file by file. Everything else in the folder is ignored.

`data/` is filled by **conf hooks** ([below](#hooks-fetching-and-generating-files)) when the
confdir is consolidated, and copied to `rootdir/initdata/` at init.

Custom plans and tasks are Python objects registered from code
([API concepts](pipelines_api.md#plans-and-tasks)); a confdir cannot carry Python files yet.

## `conf.toml`

The top-level keys are a closed set. Anything else is refused, except keys starting with `_`,
which are kept and serve as variables for [substitution](#substitution).

| Key | Required | Meaning |
|---|---|---|
| `long_name` | yes | Identity key. The confdir is `<long_name>-conf/`. |
| `name` | yes | Identity key, the short name. The rootdir is `<name>-rootdir/`, the rundir `pipe_rundirs/<name>/`, and FPS names are `<name>_<cu>`. |
| `loop_number` | yes | Identity key. Unique among the deployed pipelines of a machine. |
| `kind` | no | Pipeline kind, default `"generic"`, the only kind for now. |
| `includes` | no | Confdir-relative TOML files merged below `conf.toml`, in order. |
| `plans` | no | Restricts the plans that `milk-pipes <plan>` and `run_plan` may run on this pipeline. |
| `teardown_kill_tmux` | no | `true`: `teardown` kills the CompUs' tmux sessions. Default `false`. |
| `[compute_units.<cu>]` | no | One table per compute unit. |
| `[[dataloads]]` | no | Files to load into streams at deploy. |
| `[[hooks]]` | no | Conf hooks, run by consolidate. |
| `[systemlinkage.<mode>]` | no | Named sets of stream symlinks, applied by `link`. |

The identity keys cannot be overridden (`--set`) and `resync` refuses to change them: a change
of identity is a new pipeline.

### Compute units

```toml
[compute_units.delay0]
exec = "milk-fpsexec-mem-streamdelay"      # the fpsexec executable

[compute_units.delay0.fps]                 # FPS parameters, written at deploy
naive_mode = true
delaysec   = 0.01
in_name    = "input"
out_name   = "middle"

[compute_units.delay0.fps.procinfo]        # validated: only known procinfo fields
enabled      = true
loopcntMax   = -1
triggermode  = 3
triggersname = "input"
```

`exec` is the executable name, and the CompU's FPS is `<name>_<cu>`. Everything under `fps` is
flattened to dotted FPS parameter names (`procinfo.enabled`, `option.saveseq`) and set on the FPS
right after `fpsinit`, by the `InitFPS` task. The values are not checked against the
executable; a wrong name shows up at deploy. `fps.procinfo` is the exception: its fields are
validated (`RTprio`, `cset`, `taskset`, `NBthread`, `enabled`, `loopcntMax`, `triggermode`,
`triggersname`, `MeasureTiming`, `semindexrequested`, `triggerdelay`, `triggertimeout`).

Dotted keys are equivalent to nested tables: `procinfo.enabled = true` inside
`[compute_units.delay0.fps]` is the same as the `[compute_units.delay0.fps.procinfo]` table.

### Data loads

```toml
[[dataloads]]
source = "dm.fits"            # a file in rootdir/initdata/, so: a hook product
target = "mlatsim_dm"         # the stream to create
```

The `LoadData` step of `deploy` creates each `target` stream from the FITS file `source`,
with the file's shape and dtype. Every stream a CompU triggers on should exist before its conf
process starts, so this is the way to create the input streams of a pipeline.

### Hooks: fetching and generating files

A hook is a declared, idempotent action that produces one file or folder, its `product`, in
`data/`. Hooks are run by `milk-pipeinit consolidate`, and by `init`. They only ever touch the
confdir.

```toml
[[hooks]]
id      = "dm_flat"
type    = "fetch"                        # rsync -a --delete <source> data/<product>
source  = "calib/dm_flat.fits"           # confdir-relative, absolute, or an rsync URL (host:path)
product = "dm_flat.fits"

[[hooks]]
id      = "dm"
type    = "command"                      # argv, run with cwd = data/; must create <product>
command = ["python3", "../scripts/mkdm.py", "dm.fits"]
product = "dm.fits"
```

`id` and `product` are unique within a pipeline. A product is a plain name (no `/`, not ending
in `.tag`). Next to each product, `data/<product>.tag` records the hook's declaration digest and
the product's content hash. A hook is pending, and re-run, when its product is missing, its
bytes changed, or its declaration changed. Upstream changes at an rsync source are *not*
detected: use `consolidate --refetch ID` or `--refetch-all`.

### System linkages

```toml
[systemlinkage.loopback]       # local stream = remote stream
input  = "source_point"
output = "source_point"
```

A linkage is a named set of symlinks in `$MILK_SHM_DIR`: each local stream file
`<local>.im.shm` is pointed at `<remote>.im.shm`. `milk-pipes link <mode>` applies one,
`unlink <mode>` or `unlink all` removes them, and `teardown` removes them all. `all` is thus
a reserved mode name. Linking needs a deployed pipeline that is not running.

## Resolving: includes, overrides, substitution

`milk-pipeinit resolve DIR` prints what the pipeline will actually use, as TOML. Resolving reads
the files every time, in this order, from the lowest layer to the highest:

1. the **kind defaults** (empty for `generic`);
2. each file of `includes`, in order;
3. `conf.toml` itself;
4. the **overrides**, `--set KEY=VALUE` on `init` and `resync`, or an API argument.

Tables merge recursively, so a higher layer can change one key of a CompU. Lists and
table arrays (`[[dataloads]]`, `[[hooks]]`) are replaced whole. Then the
[substitutions](#substitution) are expanded on the merged result and the whole is validated.
Errors come out as one `ConfigError`, with one `dotted.key: <why>` line per invalid key.

### Includes

```toml
includes = ["common/streams.toml", "site.toml"]
```

Paths are relative to the confdir and must stay inside it. Included files are merged in order,
below `conf.toml`, and may not have `includes` of their own (one level only). Includes are a
way to share a stream naming scheme or a procinfo setup between pipelines of the same family.

### Overrides

```console
$ milk-pipeinit init pipelinebasic-conf --set compute_units.delay0.fps.delaysec=0.02
```

`--set` takes a dotted key and a value parsed as TOML (`0.02`, `true`, `[1, 2]`; anything that
does not parse is a string). Overrides are recorded in the rootdir's `provenance.toml` and
replayed every time the rootdir is resolved, so they survive a resync. `resync --set` adds to
them.

### Substitution

Any string value, in any table outside the table arrays, may contain `{dotted.key}` references
to other keys of the resolved configuration:

```toml
_cam   = "wfscam"                    # _-prefixed extra keys: variables
_delay = 0.003

[compute_units.delay1.fps]
in_name  = "{compute_units.delay0.fps.out_name}"      # a stream named once
procinfo.triggersname = "{_cam}_raw"                  # "wfscam_raw"
delaysec = "{_delay}"                # a value that is only a reference keeps its type: 0.003
label    = "{_delay:.1e} s"          # a format spec, as in str.format: "3.0e-03 s"
```

A value that is exactly one reference takes the referenced value, type included, so a number
or a boolean can be defined once in a `_` variable and used in several CompUs. The same works
the other way round, to give a short alias to a long key: `_ln = "{loop_number}"`, then
`triggersname = "aol{_ln}_wfsim"` (see `milk-examples/pipelines/aoloop-conf`). Inside a list,
each string element is substituted on its own. References resolve in dependency order; a
missing key or a cycle is an error. Values inside `[[dataloads]]` and `[[hooks]]` are never
substituted and cannot be referenced.

Substitution runs after the overrides are applied, so a `--set` value may be a reference too.

## Keeping a rootdir in sync: drift and resync

After `init`, the rootdir is authoritative. It has its own copies of the conf files
(`rootdir/conf/`) and of the hook products (`rootdir/initdata/`), and `provenance.toml` records
the fingerprint of each of these **artifacts** as copied, the source confdir and the overrides.

Edits can then happen in two places:

- **in the confdir**: the normal workflow. The rootdir does not change until a `resync`.
- **in `rootdir/conf/`**: a hand edit of a deployed pipeline. It takes effect at the next deploy
  and is reported as a *local* drift; nothing in the confdir is affected.

### Drift

`milk-pipes state` says whether anything drifted; `milk-pipeinit drift ROOTDIR` says what:

```console
$ milk-pipeinit drift basic-rootdir
conf.toml: source modified, local unchanged
data/dm.fits: source stale, local unchanged
  compute_units.delay0.fps.delaysec: 0.01 -> 0.02
STRUCTURAL: compute unit delay2: added
```

Each artifact gets two statuses against its record: `source` (the confdir now) and `local`
(the rootdir now), each `unchanged`, `modified`, `added` or `removed`; a hook product is
`stale` when only its declaration changed. Then come the key-level differences between the
rootdir's resolved config and the confdir's (resolved with the recorded overrides, so overrides
never show as drift), and the `STRUCTURAL` lines: CompUs added or removed, or whose `exec`
changed. A structural change means the deployed set of FPSs no longer matches, and the pipeline
has to be torn down and deployed again. The milk version is also reported when it differs from
the one recorded at init.

### Resync

```console
$ milk-pipeinit resync basic-rootdir [--artifact KEY ...] [--set K=V ...] [--overwrite-local]
applied conf.toml
refused scripts/mkdm.py: local edit (modified); use overwrite_local
Resynced: teardown → deploy may be needed.
  compute_units.delay0.fps.delaysec: 0.01 -> 0.02
```

Resync copies the drifted artifacts from the confdir into the rootdir and updates the records.
In order:

1. It is refused while any run process of the pipeline is active, and when the identity keys
   differ between the rootdir and the confdir (init a new rootdir instead).
2. The confdir is consolidated, so the hook products are current.
3. Every artifact whose `source` drifted is copied, or deleted when it was removed, and recorded.
   `--artifact KEY` restricts this to the given keys (`conf.toml`, `scripts/a.py`,
   `data/<product>`). An artifact that also has a `local` edit is refused, so a hand edit is
   never silently lost, unless `--overwrite-local`. One refusal does not stop the others.
4. The `--set` overrides are merged into the recorded ones.

Resync never tears down or deploys anything, and the banner is always printed: it is up to you
to run `milk-pipes teardown` and `deploy` when the resolved diff or a structural line calls for
it. After a partial resync (a refused artifact), `drift` shows the remaining state.

### Where the files are

| | confdir | rootdir |
|---|---|---|
| conf files | `conf.toml`, `*.toml`, `scripts/` | `conf/` (same relative paths) |
| hook products | `data/<product>`, `data/<product>.tag` | `initdata/<product>`, `initdata/<product>.tag` |
| records | — | `provenance.toml` |
| FPS work folder, logs | — | `work/`, `logs/` |

A folder is a confdir when it is named `*-conf` and holds `conf.toml` but no `provenance.toml`;
a rootdir when it is named `*-rootdir` and holds `provenance.toml` and `conf/conf.toml`.
`milk-pipeinit list [DIR]` lists both.
