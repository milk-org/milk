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
    mokainit clone --milk pipelinebasic ~/pipes  # (1)!
    cd ~/pipes
    ```

    1. Copies the conf files only: no `data/`, no hooks run, no git.

=== "Python"

    ```python
    from moka.provision.clone import clone

    confdir = clone(src_confdir, dest_parent)
    ```

## 2. Check and initialize

=== "CLI"

    ```bash
    mokainit resolve pipelinebasic-conf  # (1)!
    mokainit init pipelinebasic-conf     # (2)!
    ```

    1. Prints the resolved configuration; catches errors early.
    2. Runs the hooks and builds `~/pipes/basic-rootdir`.

=== "Python"

    ```python
    from moka.config.resolve import confresolve
    from moka.provision.rootdir import init

    config = confresolve("pipelinebasic-conf")
    rootdir = init("pipelinebasic-conf")
    ```

## 3. Deploy and link

=== "CLI"

    ```bash
    moka state -p basic           # (1)!
    moka deploy -p basic          # (2)!
    moka link loopback -p basic   # (3)!
    ```

    1. `cold`: nothing deployed yet. `-p basic` means `<cwd>/basic-rootdir`.
    2. Runs the [deploy plan](pipelines_plans.md#1-built-in-plans). The state becomes `configured`.
    3. Applies `[systemlinkage.loopback]`.

=== "Python"

    ```python
    from moka.operate.pipeline import Pipeline
    from moka.operate.plans import run_plan
    from moka.operate.state import state

    pp = Pipeline("basic")      # (1)!
    state(pp)
    run_plan(pp, "deploy")
    pp.link("loopback")
    ```

    1. `<cwd>/basic-rootdir`, or a rootdir path.

## 4. Start and stop the run processes

=== "CLI"

    ```bash
    moka runstart -p basic          # (1)!
    moka runstop -p basic delay1    # (2)!
    ```

    1. The CompUs of [`runstart`](pipelines_conf.md#3-conftoml-keys) (default: all, at once), one at a
       time when listed. The state becomes `running`.
    2. A subset.

=== "Python"

    ```python
    run_plan(pp, "runstart")
    run_plan(pp, "runstop", ["delay1"])
    ```

## 5. Tear down

=== "CLI"

    ```bash
    moka teardown -p basic  # (1)!
    moka nuke -p basic      # (2)!
    ```

    1. Back to `cold`; keeps the rootdir and `work/`.
    2. Teardown, then delete the rootdir. Asks you to type `basic`.

=== "Python"

    ```python
    run_plan(pp, "teardown")
    ```

To change the configuration of a deployed pipeline, see
[Drift and resync](pipelines_provision.md#1-changing-a-deployed-pipeline).
