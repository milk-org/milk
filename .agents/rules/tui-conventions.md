---
description: Conventions for implementing and designing Terminal User Interfaces (TUIs) in milk.
---

# TUI Conventions

All `milk` TUIs must follow the design standards established by `milk-CTRL` to ensure consistent
user experience, low latency, and zero terminal flicker.

## 1. Delta-Rendering Shadow Buffers

Never write direct, unbuffered print loops (e.g., using raw `printf` statements) to render entire
TUI screens. Unbuffered terminal output causes severe screen flickering.

- **Double Buffering**: Maintain a shadow grid buffer (`ov__shadow` / `sc_shadow`) containing the
  intended screen state, and a front grid buffer (`ov__front` / `sc_front`) containing the
  currently rendered screen state.
- **Delta Flushing**: At each frame refresh, compare the shadow buffer with the front buffer and
  only emit the ANSI positioning and color/text escape sequences for cells that changed.
- **Synchronized Output**: Wrap frame updates with the synchronized output escape sequences
  `\033[?2026h` (start) and `\033[?2026l` (end) to prevent half-rendered screens in modern
  terminal emulators.

## 2. TrueColor and ANSI Color Fallbacks

To support different terminal environments (e.g., raw console, SSH session, local GUI terminal),
TUIs must detect and support three levels of color depth:

1. **Level 3 (TrueColor/24-bit RGB)**: Default mode when `COLORTERM=truecolor` is set.
2. **Level 2 (256-color)**: Fallback using standard `\033[38;5;Xm` and `\033[48;5;Xm` color codes.
3. **Level 1 (16-color)**: Basic terminal colors (ANSI codes 30-37 and 90-97).

Always use the color-level detection API (`ov_detect_color_level()`) and apply color definitions
conditionally using this hierarchy.

## 3. Dependency Controls

- **Zero CLIcore Linking**: Standalone TUI binaries (e.g., `milk-CTRL`, `milk-fpsCTRL`,
  `milk-streamCTRL`) must **never** link against the interactive CLI shell library (`CLIcore`).
  They must link against only compute libraries (`_compute` variants) or core engine libraries
  like `ImageStreamIO`, `libfps`, and `libprocessinfo`.
- **Standard ANSI Backends**: Rather than using full ncurses libraries which can carry heavy
  cross-platform dependency weights, implement a dedicated ANSI/VT100 serial sequence parser to
  manage terminal raw modes and input sequences.

## 4. Input and Mouse Handling

- **Non-Blocking Polls**: Run TUI key input handlers in non-blocking mode. Use `read` with
  `O_NONBLOCK` or `poll`/`select` with small timeouts (~100ms) to ensure the UI continues updating
  even when the user isn't pressing keys.
- **SGR 1006 Protocol**: For mouse tracking (clicks, drags, scrolls), enable the modern SGR 1006
  mouse tracking mode (`\033[?1002h\033[?1006h`). Parse escape sequences matching `\033[<...m` or
  `\033[<...M` to extract row, column, and button information.
- **Hover State**: Support mouse hover events where appropriate (using `\033[?1003h` sequence)
  to highlight UI items before they are clicked.

## 5. Help Options and Formatting

TUI executables must support the standard help interface:

- `-h` or `--help`: Prints full usage, keybinds, layout description, and control actions.
- `-h1` or `--help-oneline`: Prints a brief, one-line description and exits.
- `-hm` or `--help-mono`: Prints the full help text forced to monochrome (disabling colors).

## 6. Color Scheme Conventions

To deliver a premium visual aesthetic and high readability, TUIs must use a dark theme:

- **Backgrounds**: Deep charcoal or slate (e.g., RGB `20, 22, 28` for terminals; RGB `30, 32, 40`
  for panels). Avoid default pitch black or bright white backgrounds.
- **Component-Type Colors**: Always use semantically distinct, cohesive accents:
  - **Streams**: Cyan / Teal (e.g., RGB `80, 200, 220`).
  - **FPS Parameters**: Soft Blue (e.g., RGB `130, 170, 255`).
  - **Processes**: Light Purple (e.g., RGB `180, 140, 255`).
- **Status Indicators**:
  - **Active / Running**: Bright Green (e.g., RGB `80, 220, 80`).
  - **Idle / Waiting**: Slate Grey (e.g., RGB `130, 140, 160`).
  - **Stale / Zombie**: Amber / Yellow (e.g., RGB `255, 180, 0`).
  - **Crashed / Error**: Red (e.g., RGB `240, 60, 60`).

## 7. Interactive Buttons and Tabs

- **Tab Bars**: Place navigation tabs along the top borders of panels. Group tabs with single
  blank spaces between them.
- **Button/Tab States**:
  - **Active & Focused**: Fill with the component's accent color (foreground matches the panel
    background).
  - **Active & Unfocused**: Fill with a dimmed grey foreground to indicate it is selected but
    inactive.
  - **Inactive / Unselected**: Match the border color or use a muted grey text.
- **Mouse Context**: Append mouse operation hints (e.g., `(Click tab)` or `(Click header to sort)`)
  in muted colors next to buttons or headers.

## 8. Layout Splits and Panel Focus

- **Double Borders for Focus**: Draw double line box characters (`╔`, `═`, `║`) around the currently
  focused panel. Use thin single lines (`╭`, `─`, `│`) for unfocused panels.
- **Panel Resizing**: Support interactive vertical and horizontal panel split resizing (e.g.,
  dragging borders via mouse or using `{`, `}`, `(`, `)` hotkeys).
- **Drop Shadows**: Render panel borders with subtle drop shadows using shaded characters (`▒`) on
  the bottom and right edges to enhance depth.

## 9. Unicode Symbols and Emojis

- **ASCII-Only Core**: Core library files, headers, comments, and general string literals must be
  strictly ASCII-only.
- **TUI-Display Exceptions**: Unicode (non-ASCII) characters are **only** permitted inside TUI
  display files (e.g., `overview_theme.h`, `streamCTRL_TUI.c`) for box-drawing characters, status
  indicators, and progress bars.
- **Prohibited Emojis**: Do not use standard colorful emojis (e.g., 🔴, 🟢, ⚠️). They render with
  inconsistent widths in different terminals, leading to misaligned columns and layout corruption.
- **Approved Symbols**: Use standard monochrome block and arrow Unicode symbols:
  - Box borders: `╭`, `╮`, `╰`, `╯`, `─`, `│`, `╔`, `╗`, `╚`, `╝`, `═`, `║`.
  - Status/Chevrons: `▶`, `◀`, `▼`, `▲`, `●`, `◆`, `▌`, `▐`.
  - Sparklines / Progress: ` `, `▁`, `▂`, `▃`, `▄`, `▅`, `▆`, `▇`, `█`.
  - Drop shadows: `▒`.

## 10. Panels and Tabs in Complicated TUIs

When a TUI grows in complexity to display multiple subsystems, follow these layout strategies:

- **Multi-View Navigation (F2-F6)**: Organize primary functional modules into full-screen views.
  Provide a top-level tab header. Map views to Function keys (e.g. F2 for Dashboard, F3 for Streams,
  F4 for Processes, F5 for FPS, F6 for Connections) or Ctrl+Arrow keys.
- **Focused Panel Switching (TAB)**: For views containing multiple concurrent panels (e.g., a list
  panel next to an inspector pane), support the `TAB` key to cycle focus. Draw focused panels
  with double borders (`╔═║`) and unfocused panels with single borders (`╭─│`).
- **Sub-Panel Tabs**: If a panel has multiple internal detail tabs, render sub-tab lists inside
  the panel's top border. Update active highlights according to panel focus.
- **Floating Modal Overlays**: For transient overlays (e.g., help menus, parameter editors, or
  yes/no confirmations), render them as centered, floating boxes. Use a drop shadow (`▒`) on the
  bottom and right edges, and clear the background characters beneath the modal to visually
  distinguish it.
