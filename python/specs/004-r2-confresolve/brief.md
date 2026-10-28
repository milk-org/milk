# R2 brief: confresolve

Roadmap: R2 · Design: design-v7 §0 (confresolve, identity keys), §4.0 rule 2, §4.4, D-5, D-7, D-26;
Appendix A.1 · Tests: py · Workflow: lean (brief + tests), as for R1c.

## Goal
`confresolve` is the only way to get a `PipelineConfig`. It builds the config from layers,
reproducibly, with no side effects and no cache.

## Touch
New `milk/confresolve.py`. `milk/infra/toml_manipulation.py` (delete `load_pipeline_config`),
`milk/infra/pipeline_models.py` (add `kind`), `milk/pipeline.py` (call `confresolve`),
`tests/pipeline/toml_schemas_test.py` (switch to `confresolve`; `kind` is no longer refused),
`docs/`.

## Interface (the spec)

```python
# milk/confresolve.py
def confresolve(confset: Path, overrides: Mapping[str, object] | None = None) -> PipelineConfig:
    """confset = the folder holding conf.toml (a confdir, or later rootdir/conf/)."""

def kind_defaults(kind: str) -> dict:
    """The kind-defaults layer: {} for "generic"; ConfigError for anything else. R5 replaces the body."""
```

Layers, lowest to highest:

1. `kind_defaults(kind)`. `kind` is read from the merged includes + conf.toml; the default is
   `"generic"`.
2. `includes`, in list order. They are read from conf.toml only.
3. conf.toml.
4. `overrides`.
5. `{var}` substitution, using the existing engine.
6. Validation (`PipelineConfig`).

Rules:
- **Merge:** recursive over tables, and the later layer wins on leaves. Lists, including table
  arrays, are replaced whole.
- **`includes` paths** are relative to `confset`; subfolders are fine. These are refused with
  `ConfigError`: absolute paths, paths that escape `confset`, missing files, and an `includes`
  key inside an included file. `includes` is consumed by the merge; it is not a model field.
- **`kind`** becomes a `PipelineConfig` field: `kind: str = "generic"`.
- **Overrides** are dotted paths through tables, applied before substitution, so they reach
  dependent values. Overriding an identity key (`long_name`, `name`, `loop_number`) is a
  `ConfigError`, and so is a path that runs through a non-table.
- **Table arrays** (`[[dataloads]]`) are neither substituted nor referenceable. The current
  engine already behaves this way; keep it.
- **Errors at this boundary:** substitution `ValueError` and pydantic `ValidationError` are
  re-raised as `ConfigError(path=<conf.toml>)` with `raise … from`.

Decisions for Vincent:
- confresolve gets its own module, `milk/confresolve.py`. R3a's `milk.provision` can re-export
  it.
- The rootdir case is just `confset = rootdir/conf/`. Rootdir detection and override replay
  from provenance are R3b's job.
- includes are one level only (not recursive).
- `kind_defaults` is a plain function, not a registry (two-caller rule; R5 adds the protocol).

## Not doing
Override replay from provenance and CLI `--set` parsing (R3b). Real kinds and kind
registration (R5). The `plans` and `hooks` keys stay refused until their slices (R4 and R3a).
No cache. No changes to the substitution engine beyond the error wrapping.

## Tests
`tests/pipeline/confresolve_test.py`, pure Python. Edit the existing tests only as listed
under Touch.

## Done when
- The R2 tests pass and the suite is green.
- Budget: net +100 LOC in `milk/`. Stop and ask if you will exceed it.
- ROADMAP R2 is `done`. `docs/` gains a 5-line layering section that links to design-v7 D-5.
