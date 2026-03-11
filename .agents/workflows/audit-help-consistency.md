---
description: Audit all help/documentation sources for cross-source consistency
---

# Audit Help Consistency

Run this workflow when you want to verify that all
help sources agree with each other. Invoke it with
`/audit-help-consistency` or when asked to check if
documentation is up to date.

## 1. Review the Source Map

Read the cross-reference groups defined in the
agent rule:
```
.agents/rules/help-consistency.md
```

## 2. Scan Recent Changes

Check which help-related files have changed recently:
```bash
git log --oneline --name-only -n 30 -- \
  'src/cli/CLIcore/milk-cli-help*.c' \
  'src/cli/CLIcore/milk-help.c' \
  'src/cli/CLIcore/CLIcore/CLIcore_help.c' \
  'src/cli/CLIcore/doc/help.txt' \
  'src/engine/libfps/milk-fps-help.c' \
  'src/engine/libfps/milk-fpsexec-help.c' \
  'src/engine/libprocessinfo/milk-procinfo-help.c' \
  'docs/*.md' \
  'src/*/README.md'
```

## 3. Cross-Check Each Group

For every cross-reference group (see the rule file),
read each sibling source and compare:

- **Command names**: identical spelling and
  module-qualified form (e.g. `mem.listim`
  vs `listim`).
- **Option flags**: same flags documented in all
  siblings.
- **Examples**: syntax and output still correct.
- **Descriptions**: no contradictions between the
  help executable and the markdown page.

Use `view_file` and `grep_search` to read each
source. Focus on the groups where recent changes
were detected in step 2.

## 4. Check fpsexec One-Liners

Verify that every installed fpsexec has a meaningful
`.description` in its `FPS_APP_INFO`. Run:
```bash
milk-fpsexec-list
```
and compare against the source code descriptions.
Flag any placeholder or empty descriptions.

## 5. Propose Updates

For each inconsistency found, update the out-of-date
source using `replace_file_content` or
`multi_replace_file_content`. Group related changes
and explain each update in the task summary.

## 6. Report

Notify the user with:
- Number of cross-reference groups checked.
- Number of inconsistencies found and fixed.
- Any items that could not be auto-fixed (e.g.,
  requiring a rebuild or manual verification).
