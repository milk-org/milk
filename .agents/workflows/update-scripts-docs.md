---
description: Sync docs/scripts.md after adding or modifying shell scripts
---

# Update Scripts Documentation

Run this workflow after adding, renaming, or removing
any `milk-*` or `cacao-*` shell script.

## 1. Inventory Installed Scripts

List all scripts that will be installed:

```bash
find src/ plugins/ -name 'milk-*' -o -name 'cacao-*' \
  | grep -v '\.c$' | grep -v '\.h$' | sort
```

Also check CMakeLists.txt `install(PROGRAMS ...)`
directives to see which scripts are actually
installed.

## 2. Read Current Documentation

```bash
view_file docs/scripts.md
```

## 3. Compare and Update

For each script:

- Verify it appears in `docs/scripts.md`.
- Verify the description is accurate.
- If the script is new, add it to the appropriate
  section.
- If the script was renamed or removed, update or
  remove the entry.

## 4. Verify Script Help

Ensure each script supports `--help` or `-h`:

```bash
<script> --help
```

If a script lacks `--help`, add a usage function
to the script.

## 5. Notify

Report the number of scripts checked and any
entries added, updated, or removed.
