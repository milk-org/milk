---
description: Ensure the Programmer's Guide stays up-to-date with architectural changes.
---

When interacting with the `milk` repository, the `docs/programmers_guide.md` file must be treated as a living document of the system's core architecture.

## Trigger Conditions
You MUST proactively review and propose updates to `docs/programmers_guide.md` if any of the following occur during a task:
1. **Core Architecture Changes:** You modify the core implementation of `fps` (Function Processing System), `ImageStreamIO`, or `processinfo`.
2. **Module Layouts:** A new paradigm or layout (e.g., changes to the standard 8-section layout for an `fpsexec` unit) is introduced.
3. **Important Global Workflows:** A new global workflow (e.g., a major shift in how tmux deployment works like `cacao-fps-deploy-v2`) becomes standard practice.

## Action Taken
1. Check the current contents of `docs/programmers_guide.md` utilizing the `view_file` tool.
2. If the new changes contradict, deprecate, or significantly expand on what is currently written in the guide, you must update the markdown file utilizing the `multi_replace_file_content` or `replace_file_content` tool.
3. Summarize your documentation changes in your task updates.
