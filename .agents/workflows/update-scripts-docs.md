---
description: Sync docs/guide/scripts.md after adding or modifying shell scripts
---

# Update Scripts Documentation

Run this workflow after adding, renaming, or removing
any `milk-*` or `cacao-*` shell script.

## 1. Inventory Installed Scripts

List all scripts that will be installed:

```bash
find scripts/ src/ plugins/ -name 'milk-*' -o -name 'cacao-*' \
  | grep -v '\.c$' | grep -v '\.h$' | sort
```

Also check CMakeLists.txt `install(PROGRAMS ...)`
directives to see which scripts are actually
installed.

## 2. Read Current Documentation

Read `docs/guide/scripts.md`.

## 3. Compare and Update

For each script, update `docs/guide/scripts.md` as the
[`script-docs`](../rules/script-docs.md) rule requires (new, renamed and removed scripts,
accurate descriptions).

## 4. Verify Script Help

Ensure each script handles the help flags of the
[`help-message-standard`](../rules/help-message-standard.md) rule (`-h`, `-h1`, `-h2`, `-hm`):

```bash
<script> -h
<script> -h1
```

If a script lacks them, add a usage function
to the script.

## 5. Notify

Report the number of scripts checked and any
entries added, updated, or removed.
