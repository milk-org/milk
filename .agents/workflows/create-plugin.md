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

- **Plugin Name:** (e.g., `milk-extra-myplugin`)
- **Description:** A short description for the README.
- **Group Folder:** Plugins are stored in `plugins/<group>/`.
- **Compute Variant:** Does it need a `_compute` variant for standalone linking?

## 2. Scaffold the Plugin

Following the guidelines in the `plugin-creator` skill, create the following files:

1. `plugins/<group>/<pluginname>/<pluginname>.c`
2. `plugins/<group>/<pluginname>/<pluginname>.h`
3. `plugins/<group>/<pluginname>/CMakeLists.txt`
4. `plugins/<group>/<pluginname>/README.md`
5. The `#ifdef MILK_NO_CLI` conditional include
   guard in the main `.c` file.
6. If adding new cross-module dependencies,
   update `docs/dependency_graph.md`.

Note: The parent directory does NOT need to be edited to add `add_subdirectory()`. The root `CMakeLists.txt` dynamically discovers all plugins at depth 2 under `plugins/`.

## 3. Verify

Run `/compile-test` to ensure the new plugin builds successfully and installs correctly into `_build/_install/`.
