# Feature Plan: [Title]

## Overview

**Goal**: [One-sentence description of the feature]
**Scope**: [Engine / Core / Full] · [New module /
Extension / Modification]
**Template**: [V2 compute unit / stream processor /
plain function / CLI builtin / none]

## Architecture Impact

### Build Tier

- **Affected tier(s)**: [Engine / Core / Full]
- **New dependencies**: [list or "none"]
- **Dependency direction valid**: [yes/no — check
  `dependency_graph.md`]

### Shared Memory Objects

| Object | Type                    | Name | Details             |
| ------ | ----------------------- | ---- | ------------------- |
|        | Stream / FPS / Procinfo |      | dtype, dims, params |

### CLI Surface

- **New commands**: [list or "none"]
- **New executables**: [list or "none"]
- **Modified commands**: [list or "none"]

## File Changes

### [Component 1]

| Action | File             | Purpose      |
| ------ | ---------------- | ------------ |
| NEW    | `file.c`         | description  |
| MODIFY | `file.c`         | what changes |
| MODIFY | `CMakeLists.txt` | new targets  |

### [Component 2]

| Action | File | Purpose |
| ------ | ---- | ------- |
| ...    | ...  | ...     |

## Documentation Updates

- [ ] Module README
- [ ] Help text cross-check (`help-consistency`)
- [ ] What's New entry
- [ ] Programmer's guide (if architectural)
- [ ] MkDocs page (if new user-facing feature)

## Implementation Phases

### Phase 1: [Foundation]

**Changes**: [list]
**Verify**: [compile-test, specific tests]
**Complexity**: [low / medium / high]

### Phase 2: [Core Logic]

**Depends on**: Phase 1
**Changes**: [list]
**Verify**: [compile-test, specific tests]
**Complexity**: [low / medium / high]

### Phase 3: [Integration]

**Depends on**: Phase 2
**Changes**: [CLI wiring, docs, tests]
**Verify**: [end-to-end test, CLI robustness]
**Complexity**: [low / medium / high]

## Risks

| Risk | Severity     | Mitigation |
| ---- | ------------ | ---------- |
|      | low/med/high |            |

## Open Questions

1. [Any decisions that need user input]
