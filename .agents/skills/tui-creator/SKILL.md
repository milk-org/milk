---
name: tui-creator
description: Reference for implementing flicker-free terminal user interfaces (TUIs).
---

# TUI Creator Guide

Use this guide when creating a new terminal dashboard or control panel tool in the `milk`
framework. The design rules (delta rendering, colour levels, dependencies, input and mouse, help
flags, colours, tabs, panel focus, Unicode, modals) are in
[`tui-conventions`](../../rules/tui-conventions.md); `milk-CTRL` is the reference implementation.

---

## 1. Directory Structure

Engine-tier TUIs (no CLI dependency, like `milk-fpsCTRL` and `milk-procCTRL`) go in
`src/tui/<tool>/`, with their target in `src/tui/CMakeLists.txt`
([Directory Map](../../../docs/arch/programmers_guide.md#4-directory-map)). A TUI built only in
the CLI tier goes under `src/cli/`, like milk-CTRL (`src/cli/overview/`); neither links `CLIcore`.

```
src/tui/myTUI/
├── milk-myTUI.c           ← Standalone binary entry point
├── myTUI_ansi.h           ← Raw mode, keys, terminal size, colour level
├── myTUI_TUIcompat.h      ← Includes myTUI_ansi.h, then milkTUI_compat.h
├── myTUI_TUIcompat.c      ← Frame buffer global state definitions
├── myTUI_TUI.h            ← TUI definitions and constants
├── myTUI_TUI.c            ← Main event loop and key dispatcher
└── myTUI_render.c         ← Drawing functions
```

---

## 2. Files to Start From

- `src/tui/fpsCTRL/fpsCTRL_ansi.h`: raw mode, keys, terminal size and colour level, under the
  names `src/tui/milkTUI_compat.h` requires (`ANSI_KEY_NONE`, `ansi_get_key()`,
  `ansi_raw_mode_enter/exit()`, `ansi_get_terminal_size()`, `ansi_detect_color_level()`).
- `src/tui/fpsCTRL/fpsCTRL_TUIcompat.h`: includes `fpsCTRL_ansi.h` before `milkTUI_compat.h`;
  `fpsCTRL_TUIcompat.c` defines the frame-buffer globals (`sc_framebuf`, cursor, terminal size).
- `src/tui/fpsCTRL/fpsCTRL_TUI.c`: main loop (`poll()` on stdin at ~10 Hz, key dispatch, redraw).
- `src/tui/CMakeLists.txt`: the `milk-fpsCTRL` target links engine libraries only, never `CLIcore`.
- `src/tui/procCTRL/`: a second example (`procCTRL_TUI.c`, `procCTRL_TUI_render.c`,
  `procCTRL_TUI_input.c`).

---

## 3. Patterns in milk-CTRL

milk-CTRL's primitives are its own (CLI tier, not shared with `src/tui/`); read them for the
patterns the rule asks for:

- `src/cli/overview/ansi/`: shadow and front buffers with delta flush (`ov_buf_*`), SGR mouse
  decoding (`overview_ansi_key.c`).
- `src/cli/overview/render/overview_theme_draw.h`: colours, gradients, sparklines.
- `src/cli/overview/render/overview_theme_panel.h`: panel borders, focus, tabs on the top border.
- `src/cli/overview/render/overview_render_popup.c`: centred popup and y/N confirmation box.
