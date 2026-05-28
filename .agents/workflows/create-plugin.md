---
name: create-plugin
description: Scaffold a new plugin module with full boilerplate.
---

# Create a New Plugin Module

Use this workflow to scaffold a new plugin module in the milk ecosystem.

**Skills to consult**:

- `plugin-creator` — directory structure and
  module registration boilerplate
- `cmake-patterns` — `_compute` variant setup
  and link conventions

**Rules to review**: `common-agent-mistakes`,
`architecture-principles`

## 1. Gather Information

Use the `plugin-creator` skill to determine how to structure the plugin. You will need to ask the user:

- **Plugin Name:** (e.g., `myplugin`)
- **Description:** A short description for the README.
- **Plugin Directory**: By default, new plugins reside directly under `plugins/`
  (e.g., `plugins/<pluginname>/`). Optionally, they can be nested under a custom group folder
  (e.g., `plugins/<group>/<pluginname>/`).
  **IMPORTANT**: Do NOT place new plugins under the `milk-extra-src` folder (which is reserved
  for core extra plugins).
- **Compute Variant:** Does it need a `_compute` variant for standalone linking?

## 2. Scaffold the Plugin

Following the guidelines in the `plugin-creator` skill, create the following files inside the new plugin folder:

1. `<pluginname>.c`
2. `<pluginname>.h`
3. `CMakeLists.txt`
4. `README.md`
5. The `#ifdef MILK_NO_CLI` conditional include
   guard in the main `.c` file.
6. If adding new cross-module dependencies,
   update `docs/dependency_graph.md`.

Note: The parent directory does NOT need to be edited to add `add_subdirectory()`. The root `CMakeLists.txt` dynamically discovers all plugins at depth 1 and 2 under `plugins/`.

## 3. Verify

Run `/compile-test` to ensure the new plugin builds successfully and installs correctly into `_build/_install/`.

## 4. Git Tracking Policy

**CRITICAL RULE**: New plugins must NEVER be committed to the `milk` repository.

- All folders under `plugins/` (except `plugins/milk-extra-src/`) are ignored by the root `.gitignore`.
- It is the user's responsibility to manage the new plugin directory (e.g., as a separate Git repository).
- Do NOT stage or commit files inside the new plugin folder to the main `milk` repository.
