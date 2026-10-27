# Design v7, Appendix A: TOML shapes, file formats, CLI map (frozen)

Companion to `design-v7.md` (frozen 2026-10-27). Terms follow its §0. Normative; amend only with v8.

## A.1 `conf.toml` (migrated `pipelinebasic`)

```toml
long_name   = "pipelinebasic"          # identity key
name        = "basic"                  # identity key (short name)
loop_number = 0                        # identity key
kind        = "generic"                # optional; default "generic"
includes    = ["common/streams.toml"]  # optional; confdir-relative; merged in order, before this file
plans       = ["deploy", "start", "stop", "teardown"]  # optional; must be allowed by the kind

[compute_units.delay0]
exec = "milk-fpsexec-mem-streamdelay"

[compute_units.delay0.fps]             # written verbatim as FPS params (dotted keys flattened)
naive_mode = true
delaysec   = 0.01
in_name    = "input"
out_name   = "middle"

[compute_units.delay0.fps.procinfo]    # validated by the procinfo schema (extra = forbid)
enabled      = true
loopcntMax   = -1
triggersname = "{compute_units.delay0.fps.in_name}"
triggermode  = 3

[compute_units.delay1]
exec = "milk-fpsexec-mem-streamdelay"

[compute_units.delay1.fps]
naive_mode = true
delaysec   = 0.01
in_name    = "{compute_units.delay0.fps.out_name}"
out_name   = "output"

[compute_units.delay1.fps.procinfo]
enabled      = true
loopcntMax   = -1
triggersname = "{compute_units.delay0.fps.out_name}"
triggermode  = 3

[[dataloads]]                          # loaded from rootdir/initdata/ into SHM by `deploy`
source = "dm_flat.fits"
target = "dm00flat"

[[hooks]]                              # conf hooks, run by consolidate
id      = "dm_flat"
type    = "fetch"                      # registered ConfHook type
source  = "rsync://server/calib/dm_flat.fits"
product = "dm_flat.fits"               # under confdir/data/

[systemlinkage.loopback]               # local stream → remote stream
input  = "source_point"
output = "source_point"
```

`{var}` substitution applies to all keys **except inside table arrays** (`[[dataloads]]`,
`[[hooks]]`). Their values pass through untouched, and nothing can reference into them (D-26).

Top-level keys are a closed set: `long_name name loop_number kind includes plans compute_units
dataloads hooks systemlinkage`. Anything else is a `ConfigError`, which fixes B2. CompU names live under
`compute_units`, so they cannot collide with reserved keys. FPS names are `<name>_<cu>`
(`basic_delay0`).

Decided: the nested CompU layout (Q-A1), table-only `dataloads` (Q-A2), and the name
`systemlinkage` is kept (Q-A3).

## A.2 Tagfile: `confdir/data/<product>.tag` (TOML)

```toml
hook_id      = "dm_flat"
decl_digest  = "sha256:…"     # hash of the [[hooks]] entry
content_hash = "sha256:…"
size         = 4194304
source       = "rsync://server/calib/dm_flat.fits"   # informational
time         = 2026-10-27T07:00:00Z                  # informational
```

## A.3 `rootdir/provenance.toml`

```toml
milk_version   = "1.0"
source_confdir = "/home/vdeo/pipes/pipelinebasic-conf"
initialized    = 2026-10-27T07:00:00Z
last_resync    = 2026-10-28T09:30:00Z     # absent until the first resync

[overrides]                               # API/CLI overrides used at init/resync; replayed on resolve
"compute_units.delay0.fps.delaysec" = 0.02

[artifacts."conf.toml"]                   # conf file artifacts (TOML + scripts/)
content_hash = "sha256:…"
size         = 812

[artifacts."scripts/align.sh"]
content_hash = "sha256:…"
size         = 230

[artifacts."data/dm_flat.fits"]           # hook artifacts: copy of the tag at init/resync
hook_id      = "dm_flat"
decl_digest  = "sha256:…"
content_hash = "sha256:…"
size         = 4194304
```

## A.4 Rundir: `$MILK_SHM_DIR/pipe_rundirs/<short>/`

| File | Content | Written by |
|------|---------|-----------|
| `boot_id` | the value of `/proc/sys/kernel/random/boot_id` at deploy | deploy |
| `claim.toml` | `name`, `loop_number`, `rootdir`, `deployed` (time), `user`, `pid`, `fps = [...]` (the FPS names deploy creates; used by teardown and the consistency check) | deploy |
| `linkage.toml` | active linkage `name`, `time`, `pairs = [{local, remote}, …]` (exactly what was applied; unlink/teardown remove these) | link / unlink |

Uniqueness: deploy refuses if `pipe_rundirs/<short>/` already exists, or if any other
`pipe_rundirs/*/claim.toml` has the same `loop_number`.

## A.5 CLI map

`milk-pipeinit` (provisioning; never needs a rundir):

| Command | Does | Replaces |
|---------|------|----------|
| `list [DIR]` | list the confdirs and rootdirs in DIR (default cwd) | `milk-pipes init` / `spool` with no name |
| `clone SRC_CONFDIR [DEST_PARENT]` | plain copy + `.gitignore` | `milk-pipes clone` |
| `consolidate CONFDIR` | run pending conf hooks | — |
| `resolve CONFDIR\|ROOTDIR` | print the resolved config | — |
| `init CONFDIR [--parent DIR] [--set KEY=VAL …]` | create `<short>-rootdir` | `milk-pipes init` |
| `drift ROOTDIR` | per-artifact drift + resolved diff | — |
| `resync ROOTDIR [--artifact ID …] [--set KEY=VAL …] [--overwrite-local]` | apply the selected drift; always prints the teardown banner | — |

`milk-pipes` (rootdir only):

| Command | Does | Replaces |
|---------|------|----------|
| `state ROOTDIR` | derived state, drift summary, package-version change | — |
| `<plan> ROOTDIR` | run a registered plan: `deploy`, `start`, `stop`, `teardown`, plus any kind plans | `spool` → `deploy`; `nuke` → `teardown` |
| `link ROOTDIR LINKAGE` / `unlink ROOTDIR LINKAGE` | apply/deapply a linkage | `sys` |

Decided (Q-A4): plans are `milk-pipes` subcommands generated from the plan registry, so kind
plans appear automatically. Single tasks are API-only.
- Existing CLI tests (`TestMilkPipes`, `test_cli_sys`) are rewritten against the new commands in
  R3b/R4. Their behavioural assertions carry over.
