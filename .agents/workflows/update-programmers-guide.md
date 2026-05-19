---
description: Refresh and update the Programmer's Guide
---

# Refresh Programmer's Guide

This workflow is used when the user explicitly requests an update to the `milk` Programmer's Guide or uses the `/update-programmers-guide` command. It is designed to comb the repository for recent undocumented architectural changes and sync the guide.

1. **Review Existing Documentation:**
   Read the current content of `docs/programmers_guide.md`. Use the `view_file` tool.

2. **Scan for Recent Changes:**
   Execute a search for recent changes utilizing `run_command` in git:
   ```bash
   git log --oneline -n 20
   ```
   Or explicitly request the user if there are any specific architectural shifts they want covered.

3. **Check Core Layout Conventions:**
   Review `src/milk_module_example/examplefunc_fps_cli_poc.c` or similar templates to ensure the "8-section layout" or standalone macros (like `FPS_MAIN_STANDALONE_V2`) have not been fundamentally redesigned without documentation.

4. **Update the Document:**
   Using `replace_file_content` or `multi_replace_file_content`, modify `docs/programmers_guide.md` with accurate, concise, and structured enhancements. Focus purely on the architecture, avoiding granular code details unless explicitly relevant across the entire project.

5. **Notify:**
   Once completed, ask the user to review the updated `docs/programmers_guide.md` file.
