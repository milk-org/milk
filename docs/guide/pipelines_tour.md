# Pipelines: tour

From an example confdir to a running pipeline and back, with the commands and their Python
equivalents.

**Audience:** Operator · See also: [Pipelines](pipelines.md) · [Operating](pipelines_operate.md) ·
[Configuration](pipelines_conf.md) · [Drift and resync](pipelines_provision.md)

The tour uses `pipelinebasic-conf`, two `milk-fpsexec-mem-streamdelay` CompUs in a row. Example
confdirs live in `milk-examples/pipelines/` (`aoloop-conf`, `pipelinebasic-conf`); the test
confdirs in `python/tests/resources/` (`pipelinebasic-conf`, `mlatsim-conf`).

## 1. Get a confdir

=== "CLI"

    ```bash
    milk-pipeinit clone $MILK_ROOT/milk-examples/pipelines/pipelinebasic-conf ~/pipes  # (1)!
    cd ~/pipes
    ```

    1. Copies the conf files only: no `data/`, no hooks run, no git.

=== "Python"

    ```python
    from milk.provision.clone import clone

    confdir = clone(src_confdir, dest_parent)
    ```

## 2. Check and initialize

=== "CLI"

    ```bash
    milk-pipeinit resolve pipelinebasic-conf  # (1)!
    milk-pipeinit init pipelinebasic-conf     # (2)!
    ```

    1. Prints the resolved configuration; catches errors early.
    2. Runs the hooks and builds `~/pipes/basic-rootdir`.

=== "Python"

    ```python
    from milk.config.resolve import confresolve
    from milk.provision.rootdir import init

    config = confresolve("pipelinebasic-conf")
    rootdir = init("pipelinebasic-conf")
    ```

## 3. Deploy and link

=== "CLI"

    ```bash
    milk-pipes state -p basic           # (1)!
    milk-pipes deploy -p basic          # (2)!
    milk-pipes link loopback -p basic   # (3)!
    ```

    1. `cold`: nothing deployed yet. `-p basic` means `<cwd>/basic-rootdir`.
    2. Runs the [deploy plan](pipelines_plans.md#1-built-in-plans). The state becomes `configured`.
    3. Applies `[systemlinkage.loopback]`.

=== "Python"

    ```python
    from milk.operate.pipeline import Pipeline
    from milk.operate.plans import run_plan
    from milk.operate.state import state

    pp = Pipeline("basic")      # (1)!
    state(pp)
    run_plan(pp, "deploy")
    pp.link("loopback")
    ```

    1. `<cwd>/basic-rootdir`, or a rootdir path.

## 4. Start and stop the run processes

Python only: no CLI equivalent yet.

```python
from milk.operate.pipeline import Pipeline
from milk.operate.tasks import RunStart, RunStop

pp = Pipeline("basic")
pp.task_do(RunStart)                    # (1)!
pp.task_do(RunStop(pp, ["delay1"]))     # (2)!
```

1. All CompUs.
2. A subset.

## 5. Tear down

=== "CLI"

    ```bash
    milk-pipes teardown -p basic  # (1)!
    milk-pipes nuke -p basic      # (2)!
    ```

    1. Back to `cold`; keeps the rootdir and `work/`.
    2. Teardown, then delete the rootdir. Asks you to type `basic`.

=== "Python"

    ```python
    run_plan(pp, "teardown")
    ```

To change the configuration of a deployed pipeline, see
[Drift and resync](pipelines_provision.md#1-changing-a-deployed-pipeline).
