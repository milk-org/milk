---
description: milk TUIs cannot be tested using browser testing tools.
---

# TUI Browser Testing

`milkCTRL` and all other `milk` TUIs (Terminal User Interfaces) operate inside the terminal and are not available through a web browser. 

You must never attempt to use web browser testing tools or subagents to launch, test, or interact with milk TUIs. They do not run a local webserver.

When testing TUIs, use text-based command output, inspect shared memory directly, or write unit tests.
