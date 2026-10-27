# R3a: Conf artifacts, conf hooks, clone, consolidate

Feature directory: `specs/005-r3a-artifacts-consolidate` · Tests: py

## /speckit-specify

```text
Feature directory: specs/005-r3a-artifacts-consolidate
Parent roadmap: specs/pipelines/ROADMAP.md → entry R3a. Design: specs/pipelines/design-v7.md (§0 confdir/clone/conf hook/conf artifact/fingerprint/tagfile/consolidate, §4.0, §4.9, D-6, D-8) and design-v7-appendix-a.md §A.1 [[hooks]], §A.2, §A.5 milk-pipeinit.

Track configuration files and fetched products, and provide the first provisioning commands.

Why: drift detection and resync (R3b) need every configuration file and fetched product to be identifiable by fingerprint, and confdirs need a clean way to be copied and to fetch their data.

Requirements:
- Conf artifacts: every TOML file of the conf set and every file under scripts/ is a conf file artifact; every conf-hook product is a hook artifact. Each artifact reports its current fingerprint, its recorded fingerprint and its drift status: unchanged, modified, stale, added or removed. stale means the bytes are unchanged but the hook declaration changed.
- Fingerprints follow design-v7 §0: sha256 of the file bytes; for a directory, a hash over the sorted (relpath, hash) list with empty directories included; size; for hook artifacts also decl_digest. Modes and mtime are ignored. Symlinks inside products are refused. The .tag suffix is reserved.
- Conf hooks are declared as [[hooks]] entries (id, type, plus type-specific fields; see A.1). Two types: "fetch", which copies a file or directory from a local path or rsync URL, and "command", which runs a local generator command that writes the product. A hook is resolved when all its artifacts match their declaration. Hooks never write outside confdir/data/.
- Each hook product gets a tagfile, <product>.tag, in the A.2 format.
- Conf-hook types are registered by name in one registry mechanism, which later slices reuse for tasks and kinds.
- clone copies a confdir's TOML files, scripts/ and hook declarations to a new location. It excludes data/, writes a .gitignore containing data/, makes no git commit, and refuses if the target exists.
- consolidate runs a confdir's pending hooks; already-resolved hooks are skipped.
- A new CLI, milk-pipeinit, provides list, clone, consolidate and resolve (resolve prints the confresolve result). The old `milk-pipes clone` is removed and its tests are migrated.

Out of scope: init, drift reports and resync (R3b); any git operation.

Success criteria: the suite is green; pure-Python tests cover the fingerprint rules (bytes only, directory hashing, empty directories, symlink refusal), stale detection, idempotent consolidate, clone's exclusions and refusal, and the milk-pipeinit commands through click's test runner.
```

## /speckit-plan

```text
Python ≥3.11, click, tomli/tomli_w; no new dependencies. rsync is invoked as a subprocess for "fetch" (it is already used by today's clone).
Modules: milk/provision/ (artifacts.py, hooks.py, clone.py, consolidate.py, cli.py) and milk/registry.py (one name → class registry with explicit registration).
ConfArtifact and ConfHook are typing.Protocols (design-v7 §4.9). Implementations: ConfFileArtifact, HookArtifact, FetchHook, CommandHook.
Register a milk-pipeinit console script in pyproject.toml.
Retire Pipeline.clone_to/update_local_git and check_clone_target (git recording is out of scope; see design-v7 D-8 and A.5). Keep the defensive symlink checks where they still apply.
Tests: py only.
```

## /speckit-implement

Two phases: (1) the artifacts, fingerprints and registry; (2) hooks, clone, consolidate and the CLI.

## Done when

- The suite is green, ROADMAP R3a is done, and `../docs/` describes the confdir layout and the milk-pipeinit commands.
