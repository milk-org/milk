---
name: feature-planner
description: Structured feature planning and
  architectural decomposition for new capabilities
---

# Feature Planner

This skill guides the agent through structured
planning before any code is written. It ensures
new features are decomposed into concrete tasks,
checked against the architecture, and sequenced
for safe incremental delivery.

## When to Use

- User describes a **new feature** or capability
- User asks to "plan", "design", or "architect"
  a change
- A task involves **multiple modules**, build
  tiers, or cross-cutting concerns
- You are unsure how a feature fits into the
  existing architecture

**Skip this skill** for:
- Single-file bug fixes
- Documentation-only changes
- Straightforward template-based tasks (use the
  relevant workflow instead)

## Phase 1 — Classify the Change

Determine the scope by answering these questions:

| Question | Options |
|----------|---------|
| **Build tier** | Engine / Core / Full? |
| **New or extension?** | New module, new file in existing module, or modification? |
| **Scope** | Single module / cross-module / framework-level? |
| **Template** | V2 compute unit / stream processor / plain function / CLI builtin / none? |

If the feature spans multiple tiers or modules,
flag this early — it likely needs phased delivery.

## Phase 2 — Map the Architecture

### 2.1 Dependency Analysis

1. Read `docs/dependency_graph.md` — identify
   which libraries and modules are in scope.
2. Check whether new cross-module dependencies
   are needed. If so, verify they respect the
   layered build order:
   ```
   Engine → Core → Full
   (no reverse dependencies allowed)
   ```
3. List any new `target_link_libraries` entries
   that will be needed in CMake.

### 2.2 Shared Memory Contracts

For each new shared-memory object, document:

| Object | Type | Name Pattern | Details |
|--------|------|-------------|---------|
| Stream | `IMAGE` | e.g., `wfs0_raw` | dtype, dimensions, shared flag |
| FPS | `FUNCTION_PARAMETER_STRUCT` | e.g., `fps.myproc` | parameter list with types |
| Processinfo | `PROCESSINFO` | auto | loop type (finite/infinite) |

### 2.3 CLI Surface

- New CLI commands? (keyword, args, help text)
- New standalone executables? (`milk-fpsexec-*`)
- Changes to existing command behavior?

## Phase 3 — Enumerate Touchpoints

### 3.1 Code Touchpoints

List every file that will be created or modified.
Group by component:

```
### [Component Name]
- [NEW] filename.c — purpose
- [MODIFY] filename.c — what changes
- [MODIFY] CMakeLists.txt — new targets/links
```

### 3.2 Documentation Touchpoints

Check which documentation rules will fire:

| Rule | Applies? | Action needed |
|------|----------|---------------|
| `readme-update` | Module files added/removed? | Update module README |
| `help-consistency` | CLI commands changed? | Cross-check help sources |
| `whatsnew-update` | Significant feature? | Add entry to whatsnew.md |
| `maintain-programmers-guide` | Architecture changed? | Update programmers_guide.md |
| `script-docs` | Scripts changed? | Update docs/scripts.md |
| `documentation-site` | New doc page needed? | Add to mkdocs.yml nav |

### 3.3 Test Touchpoints

- Existing tests that may break?
- New tests needed? (unit, CLI robustness, ctest)
- Performance benchmarks affected?

## Phase 4 — Sequence the Work

Break the implementation into **phases**, where
each phase is independently compilable and
testable:

```markdown
### Phase 1: [Foundation]
- What to implement
- How to verify (compile, test)
- Estimated complexity (low/med/high)

### Phase 2: [Core Logic]
- What to implement
- Dependencies on Phase 1
- How to verify

### Phase 3: [Integration + Polish]
- Wire up CLI / docs / tests
- How to verify end-to-end
```

**Rules for phasing:**
- Each phase must compile and pass tests
- Dependencies flow forward (Phase 2 depends on
  Phase 1, never backward)
- Put infrastructure/API changes in early phases
- Put CLI integration and documentation in later
  phases
- Keep phases small enough to review in one PR
  (prefer 1 PR per phase)

## Phase 5 — Risk Assessment

Flag any of these if they apply:

| Risk | Mitigation |
|------|------------|
| Cross-module dependency added | Verify with `dependency_graph.md`; consider `_compute` variant |
| Performance-sensitive path | Consult `performance-practices.md`; plan benchmarking |
| Breaking API change | Document migration path; consider deprecation period |
| Standalone linkage | Verify with `add_milk_standalone()`; check `_compute` variants |
| Large refactor | Use `refactor-c-source` skill; split into multiple PRs |
| Concurrency / shared memory | Consult `concurrency-practices.md` |

## Output Format

Present the plan using the template in
`templates/feature-plan-template.md`. Write it
as the implementation plan artifact and request
user review before proceeding to execution.

## Integration with Existing Rules

This skill does **not** replace any existing
rules — it orchestrates them. During planning,
explicitly check which rules will fire during
execution and account for them in the plan.

Key rules to consult during planning:
- `architecture-principles.md` — dependency
  direction
- `cmake-conventions.md` — build target setup
- `fpsexec-conventions.md` — V2 template layout
- `module-deps-declaration.md` — MODULE_DEPS
- `performance-practices.md` — hot path design
