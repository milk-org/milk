---
name: tui-creator
description: Reference for implementing flicker-free terminal user interfaces (TUIs).
---

# TUI Creator Guide

Use this guide when creating a new terminal dashboard or control panel tool in the `milk`
framework. This skill is modeled directly on the design patterns and features of `milk-CTRL`.

---

## 1. Directory Structure

Place all files for the new TUI under a dedicated subfolder in `src/cli/` (or in an appropriate
plugin folder):

```
src/cli/myTUI/
├── CMakeLists.txt
├── README.md
├── milk-myTUI.c           ← Standalone binary entry point
├── myTUI_ansi.h           ← VT100 / ANSI input and raw mode backend
├── myTUI_TUIcompat.h      ← Compatibility wrapper
├── myTUI_TUIcompat.c      ← Frame buffer global state definitions
├── myTUI_TUI.h            ← TUI definitions and constants
├── myTUI_TUI.c            ← Main event loop and key dispatcher
└── myTUI_render.c         ← Drawing functions
```

---

## 2. Terminal ANSI Backend (`myTUI_ansi.h`)

This file handles low-level terminal mode controls and byte parsing. It must provide raw mode
configuration and non-blocking input parsing.

```c
#ifndef MYTUI_ANSI_H
#define MYTUI_ANSI_H

#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#define MYTUI_KEY_NONE 0
#define MYTUI_KEY_UP 256
#define MYTUI_KEY_DOWN 257
#define MYTUI_KEY_MOUSE_CLICK 281

extern struct termios mytui__orig_termios;
extern int            mytui__raw_active;

static inline void mytui_raw_mode_enter(void)
{
    if (mytui__raw_active)
    {
        return;
    }
    tcgetattr(STDIN_FILENO, &mytui__orig_termios);
    struct raw = mytui__orig_termios;
    raw.c_iflag &= ~(unsigned int) (IXON | ICRNL | BRKINT | INPCK | ISTRIP);
    raw.c_oflag &= ~(unsigned int) (OPOST);
    raw.c_cflag |= (unsigned int) (CS8);
    raw.c_lflag &= ~(unsigned int) (ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN]  = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    // Set non-blocking stdin
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);

    // Save screen, hide cursor, enable mouse SGR 1006 tracking
    const char seq[] = "\033[?1049h\033[?25l\033[?1002h\033[?1006h";
    write(STDOUT_FILENO, seq, sizeof(seq) - 1);
    mytui__raw_active = 1;
}

static inline void mytui_raw_mode_exit(void)
{
    if (!mytui__raw_active)
    {
        return;
    }
    // Restore mouse, show cursor, restore screen
    const char seq[] = "\033[?1006l\033[?1002l\033[?25h\033[?1049l";
    write(STDOUT_FILENO, seq, sizeof(seq) - 1);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &mytui__orig_termios);
    mytui__raw_active = 0;
}
#endif
```

---

## 3. The Delta-Rendering Double Buffer

To achieve flicker-free renders, define a `CELL` matrix representing the screen and use a Delta
rendering approach to only redraw cells that differ from the previous frame.

### Compatibility Header (`myTUI_TUIcompat.h`)

Include the tool's ANSI definitions before the shared `milkTUI_compat.h`:

```c
#ifndef MYTUI_TUICOMPAT_H
#define MYTUI_TUICOMPAT_H

#include "myTUI_ansi.h"
#include "milkTUI_compat.h"

#endif
```

### Compatibility Source (`myTUI_TUIcompat.c`)

Define the required global variables for the frame buffer:

```c
#include "myTUI_TUIcompat.h"

char sc_framebuf[SC_FRAMEBUF_SIZE];
int  sc_framebuf_pos = 0;
int  sc_cursor_row   = 1;
int  sc_cursor_col   = 0;
int  sc_term_rows    = 24;
int  sc_term_cols    = 80;

struct termios mytui__orig_termios;
int            mytui__raw_active = 0;
```

---

## 4. Main Event & Render Loop

The main loop runs at a set frequency (usually ~10 Hz). In each tick, we compute the layout,
process key inputs, draw to the shadow buffer, and flush changes.

```c
#include "myTUI_TUIcompat.h"
#include <poll.h>

void myTUI_loop(void)
{
    mytui_raw_mode_enter();

    int last_rows = -1;
    int last_cols = -1;
    int need_render = 1;

    while (running)
    {
        int rows, cols;
        ansi_get_terminal_size(&rows, &cols);

        if (rows != last_rows || cols != last_cols)
        {
            // Terminal was resized; trigger full redraw
            write(STDOUT_FILENO, "\033[2J\033[H", 7);
            ov_buf_force_clear(); // Reset front buffer
            last_rows = rows;
            last_cols = cols;
            need_render = 1;
        }

        // Process all pending input
        int key;
        while ((key = ansi_get_key()) != MYTUI_KEY_NONE)
        {
            if (key == 'q')
            {
                running = 0;
            }
            // Handle other keys...
            need_render = 1;
        }

        if (need_render)
        {
            // Clear shadow buffer
            ov_buf_reset();

            // Render elements to shadow buffer
            ov_buf_pos(1, 1);
            ov_buf_bold();
            ov_buf_printf("My Tool Title");
            ov_buf_reset_attr();

            // Perform Delta Render Flush
            ov_buf_flush_delta(rows, cols);
            need_render = 0;
        }

        // Frame rate delay using poll on stdin
        struct pollfd pfd;
        pfd.fd = STDIN_FILENO;
        pfd.events = POLLIN;
        poll(&pfd, 1, 100); // Sleep for 100ms
    }

    mytui_raw_mode_exit();
}
```

---

## 5. Sleek Visual UI Guidelines

### Palette Definition (btop-inspired theme)

Define colors as `RGB` structs and apply them using the TrueColor/ANSI fallback wrappers:

```c
typedef struct { int r; int g; int b; } mytui_rgb_t;

#define TUI_BG_TERM     (mytui_rgb_t){ 20, 22, 28 }
#define TUI_BG_PANEL    (mytui_rgb_t){ 30, 32, 40 }
#define TUI_FG_STREAM   (mytui_rgb_t){ 80, 200, 220 }
#define TUI_FG_FPS      (mytui_rgb_t){ 130, 170, 255 }
#define TUI_FG_PROC     (mytui_rgb_t){ 180, 140, 255 }
#define TUI_FG_ACTIVE   (mytui_rgb_t){ 80, 220, 80 }
```

### Rendering Borders and Shadows

Render panel borders dynamically. Toggle thick/double borders for focused components, and thin
borders for unfocused components. Optionally apply a drop shadow:

```c
void mytui_draw_border(
    int row, int col, int height, int width,
    const char *title, mytui_rgb_t color,
    int is_focused, int drop_shadow)
{
    // Apply colors
    ov_theme_fg(is_focused ? color : TUI_FG_DIM);
    ov_theme_bg(TUI_BG_TERM);

    const char *tl = is_focused ? "╔" : "╭";
    const char *tr = is_focused ? "╗" : "╮";
    const char *bl = is_focused ? "╚" : "╰";
    const char *br = is_focused ? "╝" : "╯";
    const char *h  = is_focused ? "═" : "─";
    const char *v  = is_focused ? "║" : "│";

    // Top border
    ov_buf_pos(row, col);
    ov_buf_printf("%s", tl);
    ov_buf_hline_utf8(h, width - 2);
    ov_buf_printf("%s", tr);

    // Title overlay
    if (title)
    {
        ov_buf_pos(row, col + 2);
        ov_buf_bold();
        if (is_focused)
        {
            ov_theme_bg(color);
            ov_theme_fg(TUI_BG_TERM);
            ov_buf_printf(" %s ", title);
        }
        else
        {
            ov_theme_fg(TUI_FG_DIM);
            ov_theme_bg(TUI_BG_TERM);
            ov_buf_printf(" %s ", title);
        }
        ov_buf_reset_attr();
    }

    // Side borders
    for (int r = row + 1; r < row + height - 1; r++)
    {
        ov_buf_pos(r, col);
        ov_buf_printf("%s", v);
        ov_buf_pos(r, col + width - 1);
        ov_buf_printf("%s", v);
    }

    // Bottom border
    ov_buf_pos(row + height - 1, col);
    ov_buf_printf("%s", bl);
    ov_buf_hline_utf8(h, width - 2);
    ov_buf_printf("%s", br);

    // Drop shadow (bottom & right)
    if (drop_shadow)
    {
        ov_theme_fg(TUI_FG_DIM);
        ov_theme_bg(TUI_BG_TERM);
        ov_buf_pos(row + height, col + 1);
        ov_buf_hline_utf8("▒", width);
        for (int r = row + 1; r <= row + height; r++)
        {
            ov_buf_pos(r, col + width);
            ov_buf_printf("▒");
        }
    }
    ov_buf_reset_attr();
}
```

### Rendering Button and Tab States

Tabs and buttons are rendered on top borders or headers. Render them using inverted or highlighted
background colors when selected/active:

```c
void mytui_draw_tabs(
    int row, int col, int width,
    const char **tabs, int num_tabs,
    int active_tab, mytui_rgb_t color,
    int is_focused)
{
    int current_col = col + 2;
    for (int i = 0; i < num_tabs; i++)
    {
        ov_buf_pos(row, current_col);
        ov_buf_bold();
        if (i == active_tab)
        {
            if (is_focused)
            {
                ov_theme_bg(color);
                ov_theme_fg(TUI_BG_TERM);
            }
            else
            {
                ov_theme_bg(TUI_FG_DIM);
                ov_theme_fg(TUI_BG_TERM);
            }
        }
        else
        {
            ov_theme_fg(TUI_FG_MUTED);
            ov_theme_bg(TUI_BG_TERM);
        }

        char tab_text[64];
        snprintf(tab_text, sizeof(tab_text), " %s ", tabs[i]);
        ov_buf_printf("%s", tab_text);
        ov_buf_reset_attr();
        current_col += strlen(tab_text) + 1; // spacer
    }
}
```

### Interpolated Gradients and Sparklines

Draw progress bars or gauges by linear interpolation of color:

```c
mytui_rgb_t mytui_lerp(mytui_rgb_t a, mytui_rgb_t b, float t)
{
    t = (t < 0.0f) ? 0.0f : (t > 1.0f) ? 1.0f : t;
    return (mytui_rgb_t){
        a.r + (int)((b.r - a.r) * t),
        a.g + (int)((b.g - a.g) * t),
        a.b + (int)((b.b - a.b) * t)
    };
}
```

Draw sparklines using standard Unicode block characters:

```c
static const char *SPARK_CHARS[] = { " ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█" };

void mytui_draw_sparkline(
    int row, int col, const float *vals,
    int len, mytui_rgb_t color)
{
    ov_buf_pos(row, col);
    ov_theme_bg(TUI_BG_PANEL);
    ov_theme_fg(color);
    for (int i = 0; i < len; i++)
    {
        float v = (vals[i] < 0.0f) ? 0.0f : (vals[i] > 1.0f) ? 1.0f : vals[i];
        int idx = (int)(v * 8.0f + 0.5f);
        ov_buf_printf("%s", SPARK_CHARS[idx]);
    }
    ov_buf_reset_attr();
}
```

---

## 6. Unicode Symbols and Emojis Policy

- **ASCII Isolation**: Keep all core computation `.c` and `.h` files ASCII-only. Do not include
  non-ASCII characters in general comments or string literals.
- **TUI-Display Only**: Unicode symbols are strictly restricted to TUI rendering files.
- **No Colorful Emojis**: Colorful emojis (e.g. 🟢, 🔴, ⚠️) must never be used. Standard terminal
  fonts render them with variable/non-standard column widths, which will break column alignments and
  cause layout corruption.
- **Safe Unicode Glyphs**:
  - Use standard Unicode box drawing characters for panel layouts.
  - Use standard Unicode geometric blocks (e.g. `▶`, `▼`, `●`, `◆`) for monochrome chevrons and
    status glyphs.
  - Use block steps (e.g. ` `, `▂`, `▃`, `▄`, `▅`, `▆`, `▇`, `█`) for sparklines.

---

## 7. Panels, Tabs, and Modals in Complicated TUIs

When implementing complex user interfaces containing multiple panel sections, nested views, or
transient overlays:

### View Dispatching (F-Keys)

Use an `enum` to represent the top-level views and dispatch the render calls accordingly:

```c
typedef enum {
    VIEW_DASHBOARD,
    VIEW_STREAMS,
    VIEW_PROCESSES,
    VIEW_FPS,
    VIEW_CONNECTIONS
} view_mode_t;
```

### Focus Switching (TAB Key)

Track focus per-view. When the `TAB` key is pressed, cycle the active panel focus. Update the
borders dynamically to signal active focus:

```c
void mytui_render(view_mode_t mode, int focused_panel)
{
    if (mode == VIEW_DASHBOARD)
    {
        mytui_draw_border(1, 1, 10, 40, "Streams", TUI_FG_STREAM,
                          focused_panel == FOCUS_STREAMS, 0);
        mytui_draw_border(1, 41, 10, 40, "Processes", TUI_FG_PROC,
                          focused_panel == FOCUS_PROCS, 0);
    }
}
```

### Modal/Help Overlay Rendering

For floating modals (like confirmation prompts or help overlays), calculate coordinates to center
the modal. Clear the interior background cells within the shadow buffer before rendering, and
add a drop shadow:

```c
void mytui_draw_modal(
    int term_rows, int term_cols,
    int height, int width, const char *title)
{
    int row = (term_rows - height) / 2;
    int col = (term_cols - width) / 2;

    // Draw borders & drop shadow
    mytui_draw_border(row, col, height, width, title, TUI_FG_FPS, 1, 1);

    // Clear background cells in shadow buffer to prevent text leak
    for (int r = row + 1; r < row + height - 1; r++)
    {
        ov_buf_pos(r, col + 1);
        ov_theme_bg(TUI_BG_PANEL);
        ov_buf_hline(' ', width - 2);
    }
}
```
