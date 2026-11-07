# Pipelines glossary

Short definitions of the terms used in the pipelines pages. Each links to its normative
definition.

**Audience:** All · See also: [Pipelines](pipelines.md) · [Configuration](pipelines_conf.md) ·
[Operating](pipelines_operate.md) · [Internals: normative glossary](pipelines_internals.md#7-normative-glossary)

## 1. Files and folders

confdir
:   `<long_name>-conf/`, the editable source of a pipeline: TOML files, `scripts/`, hook
    declarations and `data/`. Kept in git. [Normative](pipelines_internals.md#term-confdir)

rootdir
:   `<name>-rootdir/`, the long-lived home of one pipeline instance, built by `init`. After init,
    it is authoritative. [Normative](pipelines_internals.md#term-rootdir)

rundir
:   `$MILK_SHM_DIR/pipe_rundirs/<name>/`, the claim a deploy writes for the current boot.
    Ephemeral. [Normative](pipelines_internals.md#term-rundir)

conf artifact
:   A file whose fingerprint is tracked: each conf file (TOML, `scripts/`) and each hook product.
    Drift is computed per artifact. [Normative](pipelines_internals.md#term-conf-artifact)

fingerprint
:   The record of an artifact's content (a hash) and, for a hook product, of its declaration.
    [Normative](pipelines_internals.md#term-fingerprint)

tagfile
:   `<product>.tag`, the fingerprint of a hook product, stored next to it.
    [Normative](pipelines_internals.md#term-tagfile)

identity keys
:   `long_name`, `name` and `loop_number`. Every path and FPS name derives from them; they cannot
    be overridden. [Normative](pipelines_internals.md#term-identity-keys)

## 2. Provisioning

provisioning
:   The file operations on confdirs and rootdirs: clone, consolidate, confresolve, init, drift,
    resync. CLI `milk-pipeinit`. [Normative](pipelines_internals.md#term-provisioning)

clone
:   A one-time copy of a confdir's conf files to a new location, without `data/`.
    [Normative](pipelines_internals.md#term-clone)

conf hook
:   A declared action that fetches or generates a file into `confdir/data/`.
    [Normative](pipelines_internals.md#term-conf-hook)

consolidate
:   Run the pending conf hooks of a confdir. [Normative](pipelines_internals.md#term-consolidate)

confresolve
:   Turn a confdir or rootdir, plus overrides, into one validated configuration.
    [Normative](pipelines_internals.md#term-confresolve)

init
:   Build a rootdir from a confdir. [Normative](pipelines_internals.md#term-init)

drift
:   A change of a conf artifact since it was recorded, in the confdir or in the rootdir. Only
    reported. [Normative](pipelines_internals.md#term-drift)

resync
:   Copy drifted artifacts from the confdir into an existing rootdir.
    [Normative](pipelines_internals.md#term-resync)

## 3. Live system

compute unit (CompU)
:   One milk `fpsexec` executable in a pipeline, with its FPS, its conf and run processes and its
    tmux session. [Normative](pipelines_internals.md#term-compute-unit)

kind
:   The protocol a pipeline declares (`kind = "…"`). Kinds other than `generic`: not yet
    implemented.
    [Normative](pipelines_internals.md#term-kind)

state
:   `cold`, `deployed`, `configured`, `running`, `mixed` or `inconsistent`. Computed, never
    stored. [Normative](pipelines_internals.md#term-state)

cold
:   Nothing deployed: no rundir and no FPS. [Normative](pipelines_internals.md#term-cold)

inconsistent
:   The rundir and the live FPSs disagree; only `teardown` is accepted.
    [Normative](pipelines_internals.md#term-inconsistent)

task
:   One idempotent action on the live pipeline, on all CompUs or a subset.
    [Normative](pipelines_internals.md#term-task)

plan
:   A named, ordered list of tasks with a failure policy, run by `milk-pipes <plan>`.
    [Normative](pipelines_internals.md#term-plan)
