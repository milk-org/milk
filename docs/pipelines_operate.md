# Operating a pipeline: `milk-pipes`

`milk-pipes` operates an initialized pipeline from its rootdir: it deploys, links and tears
down. To build or update a rootdir, use [`milk-pipeinit`](pipelines.md#milk-pipeinit). Terms
are defined in the [pipelines glossary](pipelines_glossary.md). The states and the plans that
move between them are drawn in the design's
[operating lifecycle](../python/specs/pipelines/design-v7.md#41-pipeline-identity-and-state).

## Commands

```
milk-pipes <plan> TARGET [--parent DIR]          run plan <plan>: deploy, teardown, ...
milk-pipes state  TARGET [--parent DIR]          state, reasons, drift summary, milk version change
milk-pipes link   TARGET LINKAGE [--parent DIR]  apply [systemlinkage.<LINKAGE>]
milk-pipes unlink TARGET LINKAGE [--parent DIR]  remove its symlinks
milk-pipes nuke   TARGET [--parent DIR] [--yes]  teardown, then rm -rf the rootdir
```

- **TARGET** is a rootdir path, or a short name meaning `<cwd>/<TARGET>-rootdir`
  (`<DIR>/<TARGET>-rootdir` with `--parent DIR`). In Python: `Pipeline(target)` and
  `Pipeline.from_parent(parent, short_name)`.
- **Plan commands** are one per registered plan (`milk-pipes --help` lists them). `deploy` runs
  ClaimRundir → InitFPS → LoadData → ConfStart and is idempotent. `teardown` runs RunStop →
  DeapplyLinkages → ConfStop → KillTmux → DestroyFPS → ReleaseRundir, best effort, and keeps the
  rootdir and `work/`.
- **`state`** never fails because of the pipeline's state. It prints the state with its reasons,
  then `drift: none`, `drift: <n> artifact(s); run ...` or `drift: unavailable (<why>)` (for
  example when the source confdir is gone), then the milk version if it changed since `init`.
- **`link`** needs a deployed pipeline that is not running.
- **`nuke`** asks you to type the short name; `--yes` skips that. It deletes the rootdir even if
  teardown fails.

Errors print `Error: <message>` and exit 1. A best-effort plan prints one
`warning: <step>: <error>` line per failed step, after running all of them, and exits 1.

## Example session

```console
$ milk-pipeinit clone $MILK_ROOT/milk-examples/pipelines/pipelinebasic-conf ~/pipes
$ milk-pipeinit init ~/pipes/pipelinebasic-conf            # → ~/pipes/basic-rootdir
$ cd ~/pipes
$ milk-pipes state basic
basic: cold
drift: none
$ milk-pipes deploy basic                                  # ClaimRundir → InitFPS → LoadData → ConfStart
$ milk-pipes state basic
basic: configured
drift: none
$ milk-pipes link basic loopback                           # needs a deployed, non-running pipeline
$ milk-pipes deploy basic                                  # idempotent: every step already succeeded
$ vi pipelinebasic-conf/conf.toml                          # edit the source confdir
$ milk-pipes state basic
basic: configured
drift: 1 artifact(s); run `milk-pipeinit drift /home/vdeo/pipes/basic-rootdir`
$ milk-pipes teardown basic                                # best effort; keeps the rootdir and work/
$ milk-pipeinit resync basic-rootdir                       # prints the teardown → deploy banner
$ milk-pipes deploy basic
$ milk-pipes state basic --parent /elsewhere               # a pipeline outside cwd
$ milk-pipes nuke basic                                    # asks you to type 'basic'
Type 'basic' to delete /home/vdeo/pipes/basic-rootdir: basic
```

For the drift details and resync, see [Drift](pipelines.md#drift) and
[Resync](pipelines.md#resync).

## Failures

```console
$ milk-pipes link basic loopback                           # cold pipeline
Error: Pipeline basic is cold; run deploy.
$ milk-pipes deploy basic                                  # an FPS was destroyed by hand
Error: Pipeline basic is inconsistent (FPS missing: basic_delay1); run teardown.
$ milk-pipes teardown basic                                # one step failed, the others ran
warning: ConfStop: <error>
$ echo $?
1
```
