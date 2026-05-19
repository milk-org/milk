---
trigger: always_on
---

# Architecture Principles

- **Minimize Cross-Module Dependencies**: Avoid introducing "spaghetti code" dependencies (e.g., plugins depending on deep internal headers of `CLIcore` or vice versa). Refactor towards a layered architecture where modules/plugins interact only through well-defined public APIs.
- **Consult Dependency Graph**: Always review `dependency_graph.md` before introducing new dependencies between modules. Maintain and intentionally reduce cross-module dependencies to avoid fragile compilation structures.
