---
tags:
  - cli
  - getting-started
---

# CLI Overview

`milk` compiles into multiple executables:

- **`milk-cli`**: The interactive command line interface (CLI). Running it starts
  a prompt from which all module functions can be launched. Type `m?` to list
  loaded modules and their commands.
- **`milk-fpsexec-*`**: Standalone executables for individual compute units.
  Each runs a single FPS-enabled function in isolation, managed via `tmux` and
  `milk-fpsCTRL`.

The source code is organized in modules. A module includes C code and
documentation. It may also include data files, scripts, and extended
documentation.

For more details see the [Programmer's Guide](../programmers_guide.md).

---
← [Documentation Index](../index.md)
