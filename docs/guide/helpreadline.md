---
tags:
  - cli
---

# Readline Keyboard Shortcuts

The milk CLI uses the [GNU Readline](https://tiswww.case.edu/php/chet/readline/rltop.html) library
for line editing. This page lists the most useful keybindings.

<!-- prettier-ignore -->
!!! tip
    For the complete reference, see the official
    [Readline User Manual](https://tiswww.case.edu/php/chet/readline/readline.html).

## Notation

| Prefix | Meaning                                                         |
| ------ | --------------------------------------------------------------- |
| `C-x`  | Hold **Ctrl** and press **x**                                   |
| `M-x`  | Hold **Alt/Meta** and press **x** (or press **Esc** then **x**) |

## Navigation

| Key   | Action                             |
| ----- | ---------------------------------- |
| `C-a` | Move to start of line              |
| `C-e` | Move to end of line                |
| `M-f` | Move forward one word              |
| `M-b` | Move backward one word             |
| `C-l` | Clear screen, reprint current line |

## Editing

| Key               | Action                  |
| ----------------- | ----------------------- |
| `C-_` / `C-x C-u` | Undo last edit          |
| `C-t`             | Transpose characters    |
| `M-t`             | Transpose words         |
| `M-u`             | Uppercase current word  |
| `M-l`             | Lowercase current word  |
| `M-c`             | Capitalize current word |

## Kill & Yank (Cut & Paste)

| Key     | Action                                    |
| ------- | ----------------------------------------- |
| `C-k`   | Kill from cursor to end of line           |
| `C-w`   | Kill previous word (whitespace-delimited) |
| `M-d`   | Kill from cursor to end of word           |
| `M-DEL` | Kill from cursor to start of word         |
| `C-y`   | Yank (paste) most recently killed text    |
| `M-y`   | Rotate kill-ring after `C-y`              |

## History

| Key         | Action                       |
| ----------- | ---------------------------- |
| `↑` / `C-p` | Previous history entry       |
| `↓` / `C-n` | Next history entry           |
| `C-r`       | Reverse incremental search   |
| `M-<`       | Jump to first history entry  |
| `M->`       | Jump to last (current) entry |

## Completion

| Key   | Action                             |
| ----- | ---------------------------------- |
| `TAB` | Autocomplete (commands, filenames) |
| `M-?` | List possible completions          |
| `M-*` | Insert all completions             |

---

← [CLI Overview](../quickstart/CLI_Overview.md) · [Documentation Index](../index.md)
