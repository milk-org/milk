# moka: tour

From an example confdir to a running pipeline and back.

**Audience:** Operator · See also: [moka](moka.md) · [Operating](operate.md) ·
[Configuration](conf.md) · [Drift and resync](provision.md)

The tour uses `pipelinebasic-conf`: two `milk-fpsexec-mem-streamdelay` CompUs in a row. Example
confdirs: `milk-examples/pipelines/`; test confdirs: `python/tests/resources/`.

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

    1. Checks the configuration.
    2. Builds `~/pipes/basic-rootdir`.

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

    1. `-p basic` means `<cwd>/basic-rootdir`.
    2. The [deploy plan](plans.md#1-built-in-plans).
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

    1. The CompUs of [`runstart`](conf.md#3-conftoml-keys).
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

    1. Keeps the rootdir and `work/`.
    2. Teardown, then delete the rootdir.

=== "Python"

    ```python
    run_plan(pp, "teardown")
    ```

To change the configuration of a deployed pipeline, see
[Drift and resync](provision.md#1-changing-a-deployed-pipeline).
