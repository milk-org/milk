# Project: milk

Read `AGENTS.md` for full onboarding context and
follow its reading order (section 2).

## Critical Rules

- **NEVER** push to the `dev` branch. Use
  `framework-dev` or feature branches derived
  from it.
- Follow templates in `src/milk_module_example/`
  for all new code.
- Always compile-test after C/CMake edits (see
  `.agents/workflows/compile-test.md`).
- Max 80 character lines, Linux kernel C style,
  Kernel-Doc comments above functions.
- Check `docs/dependency_graph.md` before adding
  cross-module dependencies.
- Use `restrict` and performance macros on hot
  paths (see `.agents/rules/performance-practices.md`).

## Key Directories

- `.agents/rules/` — always-on guardrails
- `.agents/skills/` — deep-dive instruction sets
- `.agents/workflows/` — on-demand task templates
- `src/milk_module_example/` — code templates
- `docs/` — user and developer documentation

## Quick Reference

| Task | Start Here |
|------|------------|
| New FPS compute unit | `.agents/workflows/create-fpsexec.md` |
| New plugin module | `.agents/workflows/add-new-module.md` |
| Stream processor | `.agents/workflows/add-stream-processor.md` |
| Build & test | `.agents/workflows/compile-test.md` |
| Prepare a PR | `.agents/skills/pr-preparation/SKILL.md` |
| Diagnose build error | `.agents/skills/diagnose-build-failure/SKILL.md` |
