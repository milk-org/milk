# Step 0: constitution

Run once, in Copilot chat, from the `milk/python` workspace:

```text
/speckit-constitution Project: milk python pipeline layer (package `milk` in this folder). Derive the constitution from the files cited below and link to them; do not restate them, and add no rules of your own.

I. Design contract. specs/pipelines/design-v7.md and design-v7-appendix-a.md are normative and frozen. Terms are used exactly as defined in design-v7 §0, with no synonyms (notably "compute unit"/CompU, never "session" except for tmux). Decisions D-1…D-26 (design-v7 §3) are binding. Only Vincent can amend the design (it becomes v8); agents never edit it.

II. Repository rules. The Critical Rules in ../CLAUDE.md apply to every agent: no git operations that move branches/HEAD/tags, touch GitHub, or overwrite working-tree files; no scope creep; ask when ambiguous; flag duplicate infrastructure, features or docs. ../docs/ is the human source of truth: link to it, never duplicate it.

III. Scope discipline. One feature = one slice of specs/pipelines/ROADMAP.md, and its spec cites the slice ID. Items listed as "not to be specified" or "anticipated" (ROADMAP, design-v7 §5) must not be implemented. A conflict between any artifact and design-v7 stops work until Vincent decides.

IV. Lean and readable (design-v7 §4.8.1–2). Low-volume docs with clear intent; docstrings say why. Flat control flow, early returns and guard clauses, small intent-named functions. Indirection only at the declared protocols of design-v7 §4.9 (Task, ConfHook, ConfArtifact, PipelineKind).

V. No API duplication (design-v7 §4.8.3). Reuse or extend before adding. One construction point for compute units (Pipeline.compute_units), one way to get a config (confresolve), one exception module, one registry. CLIs only parse arguments and call the API.

VI. Exception contract (design-v7 §4.7 rules 1–4). All exception classes live in milk/errors.py under MilkError. Library code never prints, exits or swallows errors; best-effort operations raise one aggregate error. Exceptions carry structured context. pyMilk exceptions propagate unwrapped.

VII. Tests. The existing pytest suite stays green at every slice. New behaviour ships with tests in the environment the ROADMAP names (py or milk).

VIII. Delivery. Lean workflow: spec.md, plan.md and tasks.md per feature. Flow-forward persistence: finished feature folders are frozen records. Each slice ends by updating ../docs/ and the slice's Status/Sub-spec cells in the ROADMAP.

Governance: Vincent ratifies and amends this constitution. Version 1.0.0, ratified today.
```

Check the result before going on. It should point to these files, not paraphrase them at length,
and it should contain nothing that isn't in the input.
