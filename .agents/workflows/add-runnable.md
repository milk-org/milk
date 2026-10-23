---
description: Add a CLI command and/or fpsexec (MILK_RUNNABLE) to a module
---

# Add a Runnable

Concepts and API: [`docs/arch/programmers_guide.md`](../../docs/arch/programmers_guide.md) §3.
Template: `src/milk_module_example/examplefunc_fps_cli_poc.c` (FPS unit),
`examplefunc4_streamprocess.c` (stream processor), `examplefunc1.c` (CLI-only unit).

**Rules**: `runnable-conventions`. **Skills**: `api-quick-reference`, `cmake-patterns`;
`imagestream-internals` for stream processors.

## 1. Gather information

Target module directory, C filename, CLI keyword, FPS name, one-line description, parameters
(types: `docs/guide/fps.md` §4), whether it is an fpsexec (`EXPORT_FPSEXEC`, milk or cacao) and
whether it is a stream processor.

## 2. Create the unit

Copy the template, then set `.app` (`fps_name`, `cmdkey`, `description`), the parameter table, the
compute function with the `MILK_PROCINFO_*(self)` loop, and the `EXPORT_FPSEXEC` line if needed.
Declare the runnable `extern` in the module header.

## 3. Register

List `&runnable_<module>_<stem>` in the module registry array (`MILK_MODULE_RUNNABLES`), add the
source to the module's `CMakeLists.txt` and a row to the module `README.md`.

## 4. Verify

Run [`/compile-test`](compile-test.md), then:

```bash
milk-fpsexec-<id> -h
milk-fpsexec-<id> -h1
milk-fpsexec-list | grep <id>
```
