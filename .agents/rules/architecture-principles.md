---
trigger: always_on
---

# Architecture Principles

- **Minimize Cross-Module Dependencies**: modules and plugins interact only through public APIs, never
  through another module's internal headers. The layers, and the CTest guards that enforce them, are in
  [architecture.md §4](../../docs/arch/architecture.md#4-layered-architecture).
- **Consult Dependency Graph**: Always review `docs/arch/dependency_graph.md` before introducing new dependencies between modules. Maintain and intentionally reduce cross-module dependencies to avoid fragile compilation structures.
