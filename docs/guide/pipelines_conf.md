# Pipeline configuration files

How a confdir is laid out, what `conf.toml` can say, and how the files are resolved into one
configuration.

**Audience:** Integrator · See also: [Pipelines](pipelines.md) ·
[Drift and resync](pipelines_provision.md) · [Plans and tasks](pipelines_plans.md) ·
[Glossary](pipelines_glossary.md)

## 1. The confdir

```text
<long_name>-conf/
  conf.toml          the pipeline description (required)
  *.toml             included files, listed in conf.toml `includes = [...]`
  scripts/           any files, copied as-is; typically what `command` hooks run
  data/              hook products and their <product>.tag files; git-ignored, never cloned
  .gitignore         `data/`, written by clone
```

The **conf files** of a confdir are every `*.toml` outside `data/` and every file under
`scripts/`. They are what `clone` copies, what `init` copies into `rootdir/conf/`, and what drift
is computed on, file by file. Everything else in the folder is ignored.

`data/` is filled by **conf hooks** ([below](#33-hooks-fetching-and-generating-files)) when the
confdir is consolidated, and copied to `rootdir/initdata/` at init.

Custom plans and tasks are described in [Plans and tasks](pipelines_plans.md#4-custom-plans).

## 2. A minimal `conf.toml`

This runs a delayed ping-pong between two compute units.

```toml
long_name   = "pipelinebasic"
name        = "basic"
loop_number = 0
teardown_kill_tmux = true

[compute_units.delay0]
exec = "milk-fpsexec-mem-streamdelay"

[compute_units.delay0.fps]              # (1)!
naive_mode = true
delaysec   = 0.01
in_name    = "input"
out_name   = "middle"
procinfo.enabled      = true
procinfo.triggermode  = 3
procinfo.triggersname = "{compute_units.delay0.fps.in_name}"   # (2)!

[compute_units.delay1]
exec = "milk-fpsexec-mem-streamdelay"

[compute_units.delay1.fps]
naive_mode = true
delaysec   = 0.01
in_name    = "{compute_units.delay0.fps.out_name}"
out_name   = "output"
procinfo.enabled      = true
procinfo.triggermode  = 3
procinfo.triggersname = "{compute_units.delay0.fps.out_name}"

[systemlinkage.loopback]                # (3)!
input  = "source_point"
output = "source_point"
```

1. Written to the FPS at deploy.
2. A [substitution](#43-substitution): the value of another key.
3. Local stream → remote stream, applied by `milk-pipes link`.

## 3. `conf.toml` keys

The top-level keys are a closed set. Anything else is refused, except keys starting with `_`,
which are kept and serve as variables for [substitution](#43-substitution).

| Key | Required | Meaning |
|---|---|---|
| `long_name` | yes | Identity key. The confdir is `<long_name>-conf/`. |
| `name` | yes | Identity key, the short name. The rootdir is `<name>-rootdir/`, the rundir `pipe_rundirs/<name>/`, and FPS names are `<name>_<cu>`. |
| `loop_number` | yes | Identity key. Unique among the deployed pipelines of a machine. |
| `kind` | no | Pipeline kind, default `"generic"`. Other kinds: not yet implemented. |
| `includes` | no | Confdir-relative TOML files merged below `conf.toml`, in order. |
| `plans` | no | Restricts the plans that `milk-pipes <plan>` and `run_plan` may run on this pipeline. |
| `teardown_kill_tmux` | no | `true`: `teardown` kills the CompUs' tmux sessions. Default `false`. |
| `[compute_units.<cu>]` | no | One table per compute unit. |
| `[[dataloads]]` | no | Files to load into streams at deploy. |
| `[[hooks]]` | no | Conf hooks, run by consolidate. |
| `[systemlinkage.<mode>]` | no | Named sets of stream symlinks, applied by `link`. |

The identity keys cannot be overridden (`--set`) and `resync` refuses to change them: a change of
identity is a new pipeline.

### 3.1. Compute units

```toml
[compute_units.delay0]
exec = "milk-fpsexec-mem-streamdelay"      # the fpsexec executable

[compute_units.delay0.fps]                 # FPS parameters, written at deploy
naive_mode = true
delaysec   = 0.01
in_name    = "input"
out_name   = "middle"

[compute_units.delay0.fps.procinfo]        # (1)!
enabled      = true
loopcntMax   = -1
triggermode  = 3
triggersname = "input"
```

1. Validated: only known procinfo fields.

`exec` is the executable name, and the CompU's FPS is `<name>_<cu>`. Everything under `fps` is
flattened to dotted FPS parameter names (`procinfo.enabled`, `option.saveseq`) and set on the FPS
right after `fpsinit`, by the `InitFPS` task. The values are not checked against the executable;
a wrong name shows up at deploy. `fps.procinfo` is the exception: its fields are validated
(`RTprio`, `cset`, `taskset`, `NBthread`, `enabled`, `loopcntMax`, `triggermode`,
`triggersname`, `MeasureTiming`, `semindexrequested`, `triggerdelay`, `triggertimeout`).

Dotted keys are equivalent to nested tables: `procinfo.enabled = true` inside
`[compute_units.delay0.fps]` is the same as the `[compute_units.delay0.fps.procinfo]` table.

### 3.2. Data loads

```toml
[[dataloads]]
source = "dm.fits"            # (1)!
target = "mlatsim_dm"         # the stream to create
```

1. A file in `rootdir/initdata/`, so: a hook product.

The `LoadData` step of `deploy` creates each `target` stream from the FITS file `source`, with
the file's shape and dtype. Every stream a CompU triggers on should exist before its conf process
starts, so this is the way to create the input streams of a pipeline.

### 3.3. Hooks: fetching and generating files

A hook is a declared, idempotent action that produces one file or folder, its `product`, in
`data/`. Hooks are run by `milk-pipeinit consolidate`, and by `init`. They only ever touch the
confdir.

```toml
[[hooks]]
id      = "dm_flat"
type    = "fetch"                        # (1)!
source  = "calib/dm_flat.fits"           # (2)!
product = "dm_flat.fits"

[[hooks]]
id      = "dm"
type    = "command"                      # (3)!
command = ["python3", "../scripts/mkdm.py", "dm.fits"]
product = "dm.fits"
```

1. `rsync -a --delete <source> data/<product>`.
2. Confdir-relative, absolute, or an rsync URL (`host:path`).
3. `command` is an argv, run with cwd = `data/`; it must create `<product>`.

`id` and `product` are unique within a pipeline. A product is a plain name (no `/`, not ending in
`.tag`). Next to each product, `data/<product>.tag` records the hook's declaration digest and the
product's content hash. A hook is pending, and re-run, when its product is missing, its bytes
changed, or its declaration changed. Upstream changes at an rsync source are *not* detected: use
`consolidate --refetch ID` or `--refetch-all`.

### 3.4. System linkages

```toml
[systemlinkage.loopback]       # local stream = remote stream
input  = "source_point"
output = "source_point"
```

A linkage is a named set of symlinks in `$MILK_SHM_DIR`: each local stream file
`<local>.im.shm` is pointed at `<remote>.im.shm`. `milk-pipes link <mode>` applies one,
`unlink <mode>` or `unlink all` removes them, and `teardown` removes them all. `all` is thus a
reserved mode name. When linking is allowed is in [Operating: commands](pipelines_operate.md#2-commands).

## 4. Resolving: includes, overrides, substitution

Resolving reads the files every time and turns them into the configuration the pipeline will
actually use.

=== "CLI"

    ```bash
    milk-pipeinit resolve pipelinebasic-conf  # (1)!
    ```

    1. A confdir or a rootdir; prints the result as TOML.

=== "Python"

    ```python
    from milk.config.resolve import confresolve

    config = confresolve("pipelinebasic-conf")    # (1)!
    config = confresolve("basic-rootdir")         # (2)!
    config = confresolve("pipelinebasic-conf", {"compute_units.delay0.fps.delaysec": 0.02})
    config.name, config.loop_number, config.fps_name("delay0")  # 'basic', 0, 'basic_delay0'
    config.compute_units["delay0"].exec           # 'milk-fpsexec-mem-streamdelay'
    config.compute_units["delay0"].fps["delaysec"]  # 0.02
    ```

    1. A confdir.
    2. A rootdir: `conf/` plus the recorded overrides.

The layers, from the lowest to the highest:

1. the **kind defaults** (empty for `generic`; other kinds not yet implemented);
2. each file of `includes`, in order;
3. `conf.toml` itself;
4. the **overrides**, `--set KEY=VALUE` on `init` and `resync`, or an API argument.

Tables merge recursively, so a higher layer can change one key of a CompU. Lists and table arrays
(`[[dataloads]]`, `[[hooks]]`) are replaced whole. Then the [substitutions](#43-substitution) are
expanded on the merged result and the whole is validated. Errors come out as one `ConfigError`,
with one `dotted.key: <why>` line per invalid key. The result object is described in
[Internals: configuration](pipelines_internals.md#2-configuration-pipelineconfig).

### 4.1. Includes

```toml
includes = ["common/streams.toml", "site.toml"]
```

Paths are relative to the confdir and must stay inside it. Included files are merged in order,
below `conf.toml`, and may not have `includes` of their own (one level only). Includes are a way
to share a stream naming scheme or a procinfo setup between pipelines of the same family.

### 4.2. Overrides

```bash
milk-pipeinit init pipelinebasic-conf --set compute_units.delay0.fps.delaysec=0.02
```

`--set` takes a dotted key and a value parsed as TOML (`0.02`, `true`, `[1, 2]`; anything that
does not parse is a string). Overrides are recorded in the rootdir's `provenance.toml` and
replayed every time the rootdir is resolved, so they survive a resync. `resync --set` adds to
them.

### 4.3. Substitution

Any string value, in any table outside the table arrays, may contain `{dotted.key}` references to
other keys of the resolved configuration:

```toml
_cam   = "wfscam"                    # (1)!
_delay = 0.003

[compute_units.delay1.fps]
in_name  = "{compute_units.delay0.fps.out_name}"      # a stream named once
procinfo.triggersname = "{_cam}_raw"                  # "wfscam_raw"
delaysec = "{_delay}"                # (2)!
label    = "{_delay:.1e} s"          # (3)!
```

1. `_`-prefixed extra keys: variables.
2. A value that is only a reference keeps its type: `0.003`.
3. A format spec, as in `str.format`: `"3.0e-03 s"`.

A value that is exactly one reference takes the referenced value, type included, so a number or a
boolean can be defined once in a `_` variable and used in several CompUs. The same works the other
way round, to give a short alias to a long key: `_ln = "{loop_number}"`, then
`triggersname = "aol{_ln}_wfsim"` (see `milk-examples/pipelines/aoloop-conf`). Inside a list,
each string element is substituted on its own. References resolve in dependency order; a missing
key or a cycle is an error. Values inside `[[dataloads]]` and `[[hooks]]` are never substituted
and cannot be referenced.

Substitution runs after the overrides are applied, so a `--set` value may be a reference too.
