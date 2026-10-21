// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file tv_ui.c
 * @brief The interface around the image: menubar and its dropdown menus, colorbar, status bar,
 *        range minibuffer and the key help box. Draws into the screen grid, registers hit rects.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "termview_internal.h"

/* Palette: milk-CTRL's default dark theme. Interface cells always set both colours. */
#define TV_C_TERM TV_RGB(20, 22, 28)
#define TV_C_BAR TV_RGB(30, 32, 40)
#define TV_C_FIELD TV_RGB(40, 44, 58)
#define TV_C_HOVER TV_RGB(45, 52, 72)
#define TV_C_SELECT TV_RGB(50, 60, 90)
#define TV_C_TEXT TV_RGB(200, 205, 215)
#define TV_C_DIM TV_RGB(100, 105, 120)
#define TV_C_MUTED TV_RGB(70, 75, 85)
#define TV_C_BRIGHT TV_RGB(240, 245, 255)
#define TV_C_ACCENT TV_RGB(130, 170, 255)
#define TV_C_STREAM TV_RGB(80, 200, 220)
#define TV_C_OK TV_RGB(80, 220, 80)
#define TV_C_WARN TV_RGB(255, 180, 0)
#define TV_C_ERR TV_RGB(240, 60, 60)
#define TV_C_SHADOW TV_RGB(12, 13, 16)

#define TV_HELP_BTN TV_M_COUNT
#define TV_QUIT_BTN (TV_M_COUNT + 1)
#define TV_SWATCH 10 /* colormap preview cells in the Cmap menu */
#define TV_ITEMS 32

const char *const tv_help_keys[][2] = {
    { "F2-F6", "Open the Cmap, Scale, Range, View, Stream menu" },
    { "Tab, F9", "Open the last menu used" },
    { "F1, ?", "This help" },
    { "Arrows", "Pan the view" },
    { "+ = / - _", "Zoom in / out" },
    { "0", "Reset zoom and pan" },
    { "c C, s S, r R", "Next / previous colormap, scaling, range mode" },
    { "l", "Lock or unlock the current range" },
    { "m / M", "Type the min / max (locks the range)" },
    { "h / v", "Flip horizontally / vertically" },
    { "Space", "Pause: freeze the image (zoom and pan still work)" },
    { "b", "Low-bandwidth mode: hold small changes on noisy images" },
    { "< , / > .", "Lower / raise the UI refresh rate" },
    { "Ctrl-L", "Repaint the screen" },
    { "Ctrl-Z", "Suspend" },
    { "q, Ctrl-C", "Quit" },
    { "Wheel", "Zoom in / out at the pointer" },
    { "Left drag", "Pan" },
    { "Right drag", "Zoom to the selected box" },
    { "Click", "Menus and items; Esc closes a menu, undoing its preview" },
};
const int tv_help_nkeys = (int) (sizeof(tv_help_keys) / sizeof(tv_help_keys[0]));

/* -----------------------------------------------------------------------
 * Drawing and hit rects
 * ----------------------------------------------------------------------- */

/**
 * tv_put - Write an interface cell with glyph and colors into the screen grid.
 * @s:  Termview state structure.
 * @r:  Zero-based screen row index.
 * @c:  Zero-based screen column index.
 * @g:  Glyph identifier or ASCII character byte.
 * @fg: Packed 24-bit foreground RGB color.
 * @bg: Packed 24-bit background RGB color.
 *
 * Stores a cell marked with `TV_ATTR_CHROME` at (@r, @c) if the coordinates
 * fall within the current terminal dimensions.
 *
 * Return: Nothing.
 */
static inline void tv_put(tv_state_t *s, int r, int c, uint8_t g, uint32_t fg, uint32_t bg)
{
    if (r >= 0 && r < s->rows && c >= 0 && c < s->cols)
    {
        s->screen[(size_t) r * s->cols + c] = tv_cell(bg, fg, g) | TV_ATTR_CHROME;
    }
}

/** tv_text() - ASCII @t from column @c, stopping before column @end; returns the next column. */
static int tv_text(tv_state_t *s, int r, int c, int end, const char *t, uint32_t fg, uint32_t bg)
{
    for (; *t != '\0' && c < end; t++, c++)
    {
        tv_put(s, r, c, (uint8_t) *t, fg, bg);
    }
    return c;
}

/**
 * tv_fill - Fill a horizontal span of cells on a single row with a background color.
 * @s:  Termview state structure.
 * @r:  Zero-based screen row index.
 * @c0: Starting column index (inclusive).
 * @c1: Ending column index (exclusive).
 * @bg: Packed 24-bit background RGB color.
 *
 * Writes space glyphs with `TV_C_TEXT` foreground and @bg background across
 * columns `@c0 .. @c1 - 1` on row @r.
 *
 * Return: Nothing.
 */
static void tv_fill(tv_state_t *s, int r, int c0, int c1, uint32_t bg)
{
    for (int c = c0; c < c1; c++)
    {
        tv_put(s, r, c, ' ', TV_C_TEXT, bg);
    }
}

/**
 * tv_hit - Register a mouse hit rectangle for an interactive UI element.
 * @s:    Termview state structure.
 * @x0:   Leftmost column index (0-based, inclusive).
 * @y0:   Topmost row index (0-based, inclusive).
 * @x1:   Rightmost column index (0-based, inclusive).
 * @y1:   Bottommost row index (0-based, inclusive).
 * @kind: Hit region category (`TV_HIT_*`).
 * @id:   Element identifier within @kind (e.g., button or menu item index).
 *
 * Clips (@x1, @y1) to the screen bounds and appends the rectangle to
 * `s->ui.hits[]` if capacity remains and the rectangle is non-empty.
 *
 * Return: Nothing.
 */
static void tv_hit(tv_state_t *s, int x0, int y0, int x1, int y1, int kind, int id)
{
    x1 = (x1 >= s->cols) ? s->cols - 1 : x1; /* clipped to the screen */
    y1 = (y1 >= s->rows) ? s->rows - 1 : y1;
    if (s->ui.nhits < TV_HITS && x1 >= x0 && y1 >= y0)
    {
        s->ui.hits[s->ui.nhits++] = (tv_hit_t) { .x0   = (int16_t) x0,
                                                 .y0   = (int16_t) y0,
                                                 .x1   = (int16_t) x1,
                                                 .y1   = (int16_t) y1,
                                                 .kind = (uint8_t) kind,
                                                 .id   = (int16_t) id };
    }
}

/** tv_hit_at() - Topmost hit rect at cell (@x, @y), 0-based; NULL if none, or if what is open
 * changed since they were drawn. */
static const tv_hit_t *tv_hit_at(const tv_state_t *s, int x, int y)
{
    if (s->ui.hits_gen != s->ui.gen)
    {
        return NULL;
    }
    for (int k = s->ui.nhits - 1; k >= 0; k--)
    {
        const tv_hit_t *h = &s->ui.hits[k];
        if (x >= h->x0 && x <= h->x1 && y >= h->y0 && y <= h->y1)
        {
            return h;
        }
    }
    return NULL;
}

/** tv_box() - Single-line rounded box (@x0, @y0)-(@x1, @y1), drop shadow right and below. */
static void tv_box(tv_state_t *s, int x0, int y0, int x1, int y1, uint32_t bg)
{
    for (int y = y0; y <= y1; y++)
    {
        for (int x = x0; x <= x1; x++)
        {
            uint8_t g = ' ';
            if (y == y0 || y == y1)
            {
                g = (x == x0)   ? ((y == y0) ? TV_G_TL : TV_G_BL)
                    : (x == x1) ? ((y == y0) ? TV_G_TR : TV_G_BR)
                                : TV_G_HLINE;
            }
            else if (x == x0 || x == x1)
            {
                g = TV_G_VLINE;
            }
            tv_put(s, y, x, g, TV_C_DIM, bg);
        }
    }
    for (int y = y0 + 1; y <= y1 + 1; y++)
    {
        tv_put(s, y, x1 + 1, TV_G_SHADE, TV_C_SHADOW, TV_C_TERM);
    }
    for (int x = x0 + 1; x <= x1 + 1; x++)
    {
        tv_put(s, y1 + 1, x, TV_G_SHADE, TV_C_SHADOW, TV_C_TERM);
    }
}

/* -----------------------------------------------------------------------
 * Menu contents
 * ----------------------------------------------------------------------- */

enum
{
    TV_I_CHOICE, /**< one of the menu's choice group: previewed when highlighted */
    TV_I_TOGGLE,
    TV_I_ACTION,
    TV_I_SEP
};

typedef struct
{
    char    label[32];
    uint8_t kind;
    uint8_t on;
    int16_t act;
    int16_t arg;
} tv_item_t;

static const int tv_fps_ladder[] = { 1, 2, 5, 10, 15, 20, 30, 60 };

/**
 * tv_add - Append an entry to a dropdown menu item array.
 * @it:    Menu item array of capacity `TV_ITEMS`.
 * @n:     Current number of items in @it.
 * @kind:  Item kind (`TV_I_CHOICE`, `TV_I_TOGGLE`, `TV_I_ACTION`, or `TV_I_SEP`).
 * @on:    Non-zero if the choice or toggle is currently active/checked.
 * @act:   Action code (`tv_action_t`) triggered when the item is activated.
 * @arg:   Integer argument passed to `tv_do()` with @act.
 * @label: Display string for the menu entry.
 *
 * Populates `it[n]` and increments @n if `@n < TV_ITEMS`.
 *
 * Return: Updated item count.
 */
static int tv_add(tv_item_t *it, int n, int kind, int on, int act, int arg, const char *label)
{
    if (n < TV_ITEMS)
    {
        snprintf(it[n].label, sizeof(it[n].label), "%s", label);
        it[n].kind = (uint8_t) kind;
        it[n].on   = (uint8_t) (on != 0);
        it[n].act  = (int16_t) act;
        it[n].arg  = (int16_t) arg;
        n++;
    }
    return n;
}

/** tv_menu_items() - The items of menu @m, as things are now; returns their number. */
static int tv_menu_items(const tv_state_t *s, int m, tv_item_t *it)
{
    const termview_options_t *o = &s->opt;
    int                       n = 0;
    char                      lab[32];
    switch (m)
    {
    case TV_M_CMAP:
        for (int k = 0; k < COLORMAP_NB; k++)
        {
            n = tv_add(it, n, TV_I_CHOICE, (int) o->colormap == k, TV_ACT_SET_CMAP, k,
                       tv_cmap_names[k]);
        }
        break;
    case TV_M_SCALE:
        for (int k = 0; k < SCALE_NB; k++)
        {
            n = tv_add(it, n, TV_I_CHOICE, (int) o->scale == k, TV_ACT_SET_SCALE, k,
                       tv_scale_names[k]);
        }
        break;
    case TV_M_RANGE:
        for (int k = 0; k < RANGE_NB; k++)
        {
            n = tv_add(it, n, TV_I_CHOICE, !o->range_locked && (int) o->range == k,
                       TV_ACT_SET_RANGE, k, tv_range_names[k]);
        }
        n = tv_add(it, n, TV_I_SEP, 0, 0, 0, "");
        /* the state the box asks for, so undoing a preview first cannot turn it around */
        n = tv_add(it, n, TV_I_TOGGLE, o->range_locked, TV_ACT_LOCK, !o->range_locked,
                   "Lock range");
        n = tv_add(it, n, TV_I_ACTION, 0, TV_ACT_PROMPT, 1, "Set min / max...");
        break;
    case TV_M_VIEW:
        n = tv_add(it, n, TV_I_ACTION, 0, TV_ACT_VIEW_RESET, 0, "Fit to window");
        n = tv_add(it, n, TV_I_ACTION, 0, TV_ACT_ZOOM, 1, "Zoom in");
        n = tv_add(it, n, TV_I_ACTION, 0, TV_ACT_ZOOM, -1, "Zoom out");
        n = tv_add(it, n, TV_I_ACTION, 0, TV_ACT_CENTRE, 0, "Centre");
        n = tv_add(it, n, TV_I_TOGGLE, o->flip_h, TV_ACT_FLIP_H, 0, "Flip horizontally");
        n = tv_add(it, n, TV_I_TOGGLE, o->flip_v, TV_ACT_FLIP_V, 0, "Flip vertically");
        n = tv_add(it, n, TV_I_SEP, 0, 0, 0, "");
        n = tv_add(it, n, TV_I_CHOICE, o->colors == 0, TV_ACT_SET_COLORS, 0, "Colors: auto");
        n = tv_add(it, n, TV_I_CHOICE, o->colors == 3, TV_ACT_SET_COLORS, 3, "Colors: 24-bit");
        n = tv_add(it, n, TV_I_CHOICE, o->colors == 2, TV_ACT_SET_COLORS, 2, "Colors: 256");
        n = tv_add(it, n, TV_I_CHOICE, o->colors == 1, TV_ACT_SET_COLORS, 1, "Colors: 16");
        break;
    case TV_M_STREAM:
        n = tv_add(it, n, TV_I_TOGGLE, s->paused, TV_ACT_PAUSE, 0, "Pause");
        n = tv_add(it, n, TV_I_SEP, 0, 0, 0, "");
        for (size_t k = 0; k < sizeof(tv_fps_ladder) / sizeof(tv_fps_ladder[0]); k++)
        {
            snprintf(lab, sizeof(lab), "UI %d fps", tv_fps_ladder[k]);
            n = tv_add(it, n, TV_I_CHOICE, s->target_fps == tv_fps_ladder[k], TV_ACT_SET_FPS,
                       tv_fps_ladder[k], lab);
        }
        n = tv_add(it, n, TV_I_SEP, 0, 0, 0, "");
        n = tv_add(it, n, TV_I_TOGGLE, o->range_smooth, TV_ACT_SMOOTH, 0, "Range smoothing");
        n = tv_add(it, n, TV_I_TOGGLE, o->lowbw, TV_ACT_LOWBW, 0, "Low-bandwidth");
        n = tv_add(it, n, TV_I_SEP, 0, 0, 0, "");
        n = tv_add(it, n, TV_I_ACTION, 0, TV_ACT_QUIT, 0, "Quit");
        break;
    default:
        break;
    }
    return n;
}

/** tv_choice() - The value of menu @m's choice group. */
static int tv_choice(const tv_state_t *s, int m)
{
    switch (m)
    {
    case TV_M_CMAP:
        return s->opt.colormap;
    case TV_M_SCALE:
        return s->opt.scale;
    case TV_M_RANGE:
        return s->opt.range;
    case TV_M_VIEW:
        return s->opt.colors;
    default:
        return s->target_fps;
    }
}

/** tv_remember() - What Esc goes back to: the choice (and range lock) as it is now. */
static void tv_remember(tv_state_t *s)
{
    s->ui.saved      = tv_choice(s, s->ui.menu);
    s->ui.saved_lock = s->opt.range_locked;
    s->ui.saved_lo   = s->cur_min;
    s->ui.saved_hi   = s->cur_max;
}

/** tv_restore() - Undo a previewed, uncommitted choice in the open menu. */
static void tv_restore(tv_state_t *s)
{
    static const int set[TV_M_COUNT] = { TV_ACT_SET_CMAP, TV_ACT_SET_SCALE, TV_ACT_SET_RANGE,
                                         TV_ACT_SET_COLORS, TV_ACT_SET_FPS };
    int              m               = s->ui.menu;
    if (tv_choice(s, m) != s->ui.saved ||
        (m == TV_M_RANGE && s->opt.range_locked != s->ui.saved_lock))
    {
        tv_do(s, (tv_action_t) set[m], s->ui.saved);
        if (m == TV_M_RANGE)
        {
            s->opt.range_locked = s->ui.saved_lock; /* the limits themselves were never touched */
            s->cur_min          = s->ui.saved_lo;   /* and the range shown comes back with it */
            s->cur_max          = s->ui.saved_hi;
        }
    }
}

/** tv_first_sel() - Item to highlight when menu @m opens: the current choice in a menu made of
 * choices (Cmap, Scale, Range), else the first item. */
static int tv_first_sel(int m, const tv_item_t *it, int n)
{
    for (int k = 0; k < n && m <= TV_M_RANGE; k++)
    {
        if (it[k].kind == TV_I_CHOICE && it[k].on)
        {
            return k;
        }
    }
    for (int k = 0; k < n; k++)
    {
        if (it[k].kind != TV_I_SEP)
        {
            return k;
        }
    }
    return 0;
}

/**
 * tv_ui_open - Open dropdown menu @m or close any active menu and help overlay.
 * @s: Termview state structure.
 * @m: Menu index (`0 .. TV_M_COUNT - 1`) to open, or -1 to close all menus.
 *
 * Restores any uncommitted preview in the previously open menu, clears the
 * help box and hover state, and if @m names a valid menu, closes the range
 * minibuffer, selects the initial item, and snapshots the current choice so
 * `Esc` can revert previews.
 *
 * Return: Nothing.
 */
void tv_ui_open(tv_state_t *s, int m)
{
    if (s->ui.open)
    {
        tv_restore(s); /* leaving a menu undoes its preview */
    }
    s->ui.open  = 0;
    s->ui.help  = 0;
    s->ui.hover = 0;
    s->ui.gen++;
    if (m >= 0 && m < TV_M_COUNT)
    {
        tv_item_t it[TV_ITEMS];
        int       n    = tv_menu_items(s, m, it);
        s->mini.active = 0;
        s->ui.open     = 1;
        s->ui.menu     = m;
        s->ui.sel      = tv_first_sel(m, it, n);
        s->ui.top      = 0;
        tv_remember(s);
    }
    s->dirty |= TV_D_CHROME;
}

/** tv_close() - Close the open menu, keeping what it now shows. */
static void tv_close(tv_state_t *s)
{
    s->ui.open  = 0;
    s->ui.hover = 0;
    s->ui.gen++;
    s->dirty |= TV_D_CHROME;
}

/** tv_move() - Highlight item @to (separators skipped in the direction of travel), previewing a
 * choice. */
static void tv_move(tv_state_t *s, const tv_item_t *it, int n, int to)
{
    int step = (to >= s->ui.sel) ? 1 : -1;
    to       = (to < 0) ? 0 : (to >= n) ? n - 1 : to;
    while (to >= 0 && to < n && it[to].kind == TV_I_SEP)
    {
        to += step;
    }
    if (to < 0 || to >= n || to == s->ui.sel)
    {
        return;
    }
    s->ui.sel = to;
    if (it[to].kind == TV_I_CHOICE)
    {
        tv_do(s, (tv_action_t) it[to].act, it[to].arg);
    }
    s->dirty |= TV_D_CHROME;
}

/** tv_activate() - Choose item @k: commit a choice, flip a toggle, run an action. With @stay the
 * menu stays open (Space, or a click on a toggle). */
static void tv_activate(tv_state_t *s, const tv_item_t *it, int k, int stay)
{
    const tv_item_t *i = &it[k];
    s->ui.sel          = k;
    if (i->kind == TV_I_SEP)
    {
        return;
    }
    if (i->kind != TV_I_CHOICE)
    {
        tv_restore(s); /* a toggle or an action does not keep a choice passed over on the way */
        tv_remember(s);
    }
    if (i->kind == TV_I_ACTION && (!stay || i->act == TV_ACT_PROMPT || i->act == TV_ACT_QUIT))
    {
        tv_close(s);
        tv_do(s, (tv_action_t) i->act, i->arg);
        return;
    }
    tv_do(s, (tv_action_t) i->act, i->arg);
    if (s->ui.open)
    {
        tv_remember(s); /* committed: Esc no longer undoes it */
        if (!stay)
        {
            tv_close(s);
        }
    }
    s->dirty |= TV_D_CHROME;
}

/* -----------------------------------------------------------------------
 * Keys
 * ----------------------------------------------------------------------- */

/**
 * tv_passes - Check if a key event is a global session control key that bypasses open overlays.
 * @ev: Decoded key event to inspect.
 *
 * Allows quit (`q`, `Ctrl-C`, `Ctrl-\`), suspend (`Ctrl-Z`), and repaint
 * (`Ctrl-L`) keys to pass through when a dropdown menu or help box is open.
 *
 * Return: Non-zero if @ev should pass through to the main key handler, or 0 otherwise.
 */
static int tv_passes(const tv_event_t *ev)
{
    /* quit, suspend and repaint work whatever is open */
    return ev->key == 'q' || ev->key == TV_CTRL('c') || ev->key == TV_CTRL('\\') ||
           ev->key == TV_CTRL('z') || ev->key == TV_CTRL('l');
}

/**
 * tv_menu_key - Handle keyboard navigation and selection while a dropdown menu is open.
 * @s:  Termview state structure.
 * @ev: Decoded key event.
 *
 * Processes menu navigation (`Up`, `Down`, `Home`, `End`, `PgUp`, `PgDn`,
 * `Left`, `Right`, `Tab`), item activation (`Enter`, `Space`), menu switching
 * (`F2`-`F6`), help (`F1`, `?`), and cancellation (`Esc`).
 *
 * Return: 1 if the key was handled (or consumed by the menu), or 0 to pass it through.
 */
static int tv_menu_key(tv_state_t *s, const tv_event_t *ev)
{
    tv_item_t it[TV_ITEMS];
    int       n    = tv_menu_items(s, s->ui.menu, it);
    int       page = (s->lay.rows > 8) ? s->lay.rows - 6 : 2;
    uint32_t  k    = ev->key;
    if (k >= TV_K_F1 + 1 && k <= TV_K_F1 + 5)
    {
        int m = (int) (k - TV_K_F1 - 1);
        tv_ui_open(s, (m == s->ui.menu) ? -1 : m);
        return 1;
    }
    switch (k)
    {
    case TV_K_ESC:
        tv_ui_open(s, -1);
        return 1;
    case '\r':
    case '\n':
        tv_activate(s, it, s->ui.sel, 0);
        return 1;
    case ' ':
        tv_activate(s, it, s->ui.sel, 1);
        return 1;
    case TV_K_UP:
        tv_move(s, it, n, s->ui.sel - 1);
        return 1;
    case TV_K_DOWN:
        tv_move(s, it, n, s->ui.sel + 1);
        return 1;
    case TV_K_HOME:
        s->ui.sel = n; /* move "up" to the first selectable item */
        tv_move(s, it, n, 0);
        return 1;
    case TV_K_END:
        s->ui.sel = -1;
        tv_move(s, it, n, n - 1);
        return 1;
    case TV_K_PGUP:
        tv_move(s, it, n, s->ui.sel - page);
        return 1;
    case TV_K_PGDN:
        tv_move(s, it, n, s->ui.sel + page);
        return 1;
    case TV_K_LEFT:
        tv_ui_open(s, (s->ui.menu + TV_M_COUNT - 1) % TV_M_COUNT);
        return 1;
    case TV_K_RIGHT:
        tv_ui_open(s, (s->ui.menu + 1) % TV_M_COUNT);
        return 1;
    case '\t':
        tv_ui_open(s, (s->ui.menu + ((ev->mods & TV_MOD_SHIFT) ? TV_M_COUNT - 1 : 1)) % TV_M_COUNT);
        return 1;
    case '?':
    case TV_K_F1:
        tv_ui_open(s, -1);
        tv_do(s, TV_ACT_HELP, 0);
        return 1;
    default:
        return !tv_passes(ev);
    }
}

/**
 * tv_help_key - Handle keyboard scrolling and dismissal while the help overlay is open.
 * @s:  Termview state structure.
 * @ev: Decoded key event.
 *
 * Closes the help overlay on `Esc`, `?`, `F1`, or `Enter`, scrolls the key
 * table on navigation keys (`Up`, `Down`, `PgUp`, `PgDn`, `Home`, `End`), and
 * blocks non-global keys.
 *
 * Return: 1 if the key was handled (or consumed), or 0 to pass it through.
 */
static int tv_help_key(tv_state_t *s, const tv_event_t *ev)
{
    switch (ev->key)
    {
    case TV_K_ESC:
    case '?':
    case TV_K_F1:
    case '\r':
    case '\n':
        tv_do(s, TV_ACT_HELP, 0);
        return 1;
    case TV_K_UP:
    case TV_K_DOWN:
    case TV_K_PGUP:
    case TV_K_PGDN:
    case TV_K_HOME:
    case TV_K_END:
    {
        int page = (s->lay.rows > 8) ? s->lay.rows - 6 : 2;
        int d    = (ev->key == TV_K_UP)     ? -1
                   : (ev->key == TV_K_DOWN) ? 1
                   : (ev->key == TV_K_PGUP) ? -page
                   : (ev->key == TV_K_PGDN) ? page
                   : (ev->key == TV_K_HOME) ? -tv_help_nkeys
                                            : tv_help_nkeys;
        s->ui.help_top += d; /* clamped when drawn */
        s->ui.help_top = (s->ui.help_top < 0) ? 0 : s->ui.help_top;
        s->dirty |= TV_D_CHROME;
        return 1;
    }
    default:
        return !tv_passes(ev);
    }
}

/** tv_mini_check() - Why the typed limits cannot be used yet, or NULL. */
static const char *tv_mini_check(const tv_mini_t *m, double *lo, double *hi)
{
    if (tv_parse_double(m->text[0], lo) != 0 || tv_parse_double(m->text[1], hi) != 0)
    {
        return "invalid number";
    }
    return (*hi > *lo) ? NULL : "min must be < max";
}

/**
 * tv_mini_open - Open the status-row range minibuffer pre-filled with the current min and max.
 * @s:     Termview state structure.
 * @field: Initial field to focus (0 for minimum, 1 for maximum).
 *
 * Closes any open menu or help box, formats `s->cur_min` and `s->cur_max` into
 * `s->mini.text[]`, marks both fields fresh so the first character typed
 * replaces the pre-filled value, and flags the UI chrome dirty.
 *
 * Return: Nothing.
 */
void tv_mini_open(tv_state_t *s, int field)
{
    tv_mini_t *m = &s->mini;
    double     lo, hi;
    if (s->ui.open)
    {
        tv_ui_open(s, -1); /* without keeping a choice passed over */
    }
    s->ui.help = 0;
    s->ui.gen++;
    m->active = 1;
    m->field  = (field == 1) ? 1 : 0;
    snprintf(m->text[0], sizeof(m->text[0]), "%.6g", s->cur_min);
    snprintf(m->text[1], sizeof(m->text[1]), "%.6g", s->cur_max);
    m->fresh[0] = m->fresh[1] = 1;
    m->err                    = tv_mini_check(m, &lo, &hi);
    s->dirty |= TV_D_CHROME;
}

/**
 * tv_mini_key - Process a key event while the range minibuffer is active.
 * @s:  Termview state structure.
 * @ev: Decoded key event.
 *
 * Handles `Enter` (validates and locks range), `Esc` (cancels), `Tab`
 * (switches between min and max fields), `Backspace`, and printable ASCII
 * characters, leaving navigation and function keys to the main handler.
 *
 * Return: 1 if the key was consumed by the minibuffer, or 0 otherwise.
 */
static int tv_mini_key(tv_state_t *s, const tv_event_t *ev)
{
    tv_mini_t *m = &s->mini;
    uint32_t   k = ev->key;
    char      *t = m->text[m->field];
    size_t     n = strlen(t);
    double     lo, hi;
    if (ev->mods & (TV_MOD_ALT | TV_MOD_CTRL))
    {
        return 0;
    }
    if (k == '\r' || k == '\n')
    {
        if (tv_mini_check(m, &lo, &hi) == NULL)
        {
            tv_set_range(s, lo, hi);
            m->active = 0;
        }
    }
    else if (k == TV_K_ESC)
    {
        m->active = 0;
    }
    else if (k == '\t')
    {
        m->field ^= 1;
    }
    else if (k == 127 || k == 8)
    {
        if (n > 0 && !m->fresh[m->field])
        {
            t[n - 1] = '\0';
        }
        else if (m->fresh[m->field])
        {
            t[0] = '\0'; /* Backspace on the pre-filled text clears it */
        }
        m->fresh[m->field] = 0;
    }
    else if (k >= 32 && k <= 126)
    {
        if (m->fresh[m->field])
        {
            t[0] = '\0';
            n    = 0;
        }
        m->fresh[m->field] = 0;
        if (n + 1 < sizeof(m->text[0]))
        {
            t[n]     = (char) k;
            t[n + 1] = '\0';
        }
    }
    else
    {
        return 0; /* arrows, function keys: their usual meaning */
    }
    m->err = tv_mini_check(m, &lo, &hi);
    s->dirty |= TV_D_CHROME;
    return 1;
}

/**
 * tv_ui_key - Route a key event to the active UI overlay or menubar shortcuts.
 * @s:  Termview state structure.
 * @ev: Decoded key event.
 *
 * Delegates to the range minibuffer, help overlay, or open dropdown menu when
 * active, or opens the help overlay (`?`, `F1`) or dropdown menus (`F2`-`F6`,
 * `Tab`, `F9`) when the terminal layout is valid.
 *
 * Return: 1 if the UI handled the key event, or 0 to pass it to `tv_handle_key()`.
 */
int tv_ui_key(tv_state_t *s, const tv_event_t *ev)
{
    if (!s->lay.valid)
    {
        return 0; /* too small to show menus: keys keep their plain meaning */
    }
    if (s->mini.active && tv_mini_key(s, ev))
    {
        return 1;
    }
    if (s->ui.help)
    {
        return tv_help_key(s, ev);
    }
    if (s->ui.open)
    {
        return tv_menu_key(s, ev);
    }
    if ((ev->mods & TV_MOD_ALT) && ev->key < TV_K_UP)
    {
        return 0;
    }
    if (ev->key == '?' || ev->key == TV_K_F1)
    {
        tv_do(s, TV_ACT_HELP, 0);
        return 1;
    }
    if (ev->key >= TV_K_F1 + 1 && ev->key <= TV_K_F1 + 5)
    {
        tv_ui_open(s, (int) (ev->key - TV_K_F1 - 1));
        return 1;
    }
    if ((ev->key == '\t' && !(ev->mods & TV_MOD_SHIFT)) || ev->key == TV_K_F1 + 8)
    {
        tv_ui_open(s, s->ui.menu);
        return 1;
    }
    return 0;
}

/* -----------------------------------------------------------------------
 * Mouse
 * ----------------------------------------------------------------------- */

/**
 * tv_is_press - Check if a mouse event is a left, middle, or right button press.
 * @ev: Decoded mouse event to test.
 *
 * Distinguishes primary button press actions (`TV_MA_PRESS` with button
 * `<= TV_MB_RIGHT`) from releases, drags, moves, and wheel events.
 *
 * Return: Non-zero if @ev is a button press, or 0 otherwise.
 */
static int tv_is_press(const tv_event_t *ev)
{
    return ev->action == TV_MA_PRESS && ev->button <= TV_MB_RIGHT;
}

/**
 * tv_ui_wants_mouse - Check if a mouse event should be handled by the UI chrome rather than the image.
 * @s:  Termview state structure.
 * @ev: Decoded mouse event.
 *
 * Claims all mouse events while a dropdown menu or help box is open, and
 * claims button presses on the menubar row, status row, or any registered hit
 * rectangle when no overlay is open.
 *
 * Return: Non-zero if the UI claims @ev, or 0 if it belongs to the image canvas.
 */
int tv_ui_wants_mouse(const tv_state_t *s, const tv_event_t *ev)
{
    if (!s->lay.valid)
    {
        return 0; /* nothing of the interface is on screen */
    }
    if (s->ui.open || s->ui.help)
    {
        return 1;
    }
    int y = ev->y - 1;
    return tv_is_press(ev) &&
           (y == 0 || y == s->lay.status_row || tv_hit_at(s, ev->x - 1, y) != NULL);
}

/** tv_button() - A click on menubar button @b. */
static void tv_button(tv_state_t *s, int b)
{
    if (b == TV_QUIT_BTN)
    {
        tv_do(s, TV_ACT_QUIT, 0);
    }
    else if (b == TV_HELP_BTN)
    {
        int was = s->ui.help;
        tv_ui_open(s, -1); /* closes menus and the help box */
        if (!was)
        {
            tv_do(s, TV_ACT_HELP, 0);
        }
    }
    else
    {
        tv_ui_open(s, (s->ui.open && s->ui.menu == b) ? -1 : b);
    }
}

/**
 * tv_ui_mouse - Process a mouse event claimed by the UI chrome or open overlays.
 * @s:  Termview state structure.
 * @ev: Decoded mouse event.
 *
 * Updates button hover state, scrolls or closes the help overlay, tracks hover
 * and clicks across open dropdown menus and menubar buttons, or switches focus
 * between range minibuffer fields.
 *
 * Return: Nothing.
 */
void tv_ui_mouse(tv_state_t *s, const tv_event_t *ev)
{
    const tv_hit_t *h     = tv_hit_at(s, ev->x - 1, ev->y - 1);
    int             press = tv_is_press(ev);
    int             wheel = (ev->button == TV_MB_WHEEL_UP || ev->button == TV_MB_WHEEL_DOWN);
    int             d     = (ev->button == TV_MB_WHEEL_UP) ? -1 : 1;
    int             hover = (h && h->kind == TV_HIT_BUTTON) ? h->id + 1 : 0;

    if (!wheel && hover != s->ui.hover)
    {
        s->ui.hover = hover;
        s->dirty |= TV_D_CHROME;
    }
    if (s->ui.help)
    {
        if (wheel)
        {
            s->ui.help_top = (s->ui.help_top + 3 * d < 0) ? 0 : s->ui.help_top + 3 * d;
            s->dirty |= TV_D_CHROME;
        }
        else if (press && h && h->kind == TV_HIT_BUTTON)
        {
            tv_button(s, h->id);
        }
        else if (press && !(h && h->kind == TV_HIT_PANEL))
        {
            tv_do(s, TV_ACT_HELP, 0);
        }
        return;
    }
    if (s->ui.open)
    {
        tv_item_t it[TV_ITEMS];
        int       n = tv_menu_items(s, s->ui.menu, it);
        if (ev->action == TV_MA_MOVE || ev->action == TV_MA_DRAG)
        {
            if (h && h->kind == TV_HIT_ITEM && h->id < n && h->id != s->ui.sel)
            {
                tv_move(s, it, n, h->id);
            }
            else if (hover && hover - 1 < TV_M_COUNT && hover - 1 != s->ui.menu)
            {
                tv_ui_open(s, hover - 1); /* an open menu follows the pointer along the bar */
            }
        }
        else if (wheel)
        {
            tv_move(s, it, n, s->ui.sel + d);
        }
        else if (press)
        {
            if (h && h->kind == TV_HIT_ITEM && h->id < n)
            {
                tv_activate(s, it, h->id, it[h->id].kind == TV_I_TOGGLE);
            }
            else if (h && h->kind == TV_HIT_BUTTON)
            {
                tv_button(s, h->id);
            }
            else if (!(h && h->kind == TV_HIT_PANEL))
            {
                tv_ui_open(s, -1); /* a click elsewhere closes the menu and does nothing else */
            }
        }
        return;
    }
    if (press && h)
    {
        if (h->kind == TV_HIT_BUTTON)
        {
            tv_button(s, h->id);
        }
        else if (h->kind == TV_HIT_FIELD && s->mini.active)
        {
            s->mini.field = h->id;
            s->dirty |= TV_D_CHROME;
        }
    }
}

/* -----------------------------------------------------------------------
 * Menubar
 * ----------------------------------------------------------------------- */

/** tv_button_label() - Text of menubar button @b, long (@lng) or short. */
static void tv_button_label(const tv_state_t *s, int b, int lng, char *out, size_t n)
{
    const termview_options_t *o = &s->opt;
    switch (b)
    {
    case TV_M_CMAP:
        snprintf(out, n, lng ? " Cmap %s" : " %s", tv_cmap_names[o->colormap]);
        break;
    case TV_M_SCALE:
        snprintf(out, n, lng ? " Scale %s" : " %s", tv_scale_names[o->scale]);
        break;
    case TV_M_RANGE:
        snprintf(out, n, lng ? " Range %s" : " %s",
                 o->range_locked ? "locked" : tv_range_names[o->range]);
        break;
    case TV_M_VIEW:
        snprintf(out, n, " View");
        break;
    case TV_M_STREAM:
        snprintf(out, n, " Stream");
        break;
    case TV_HELP_BTN:
        snprintf(out, n, lng ? " ? Help " : " ? ");
        break;
    default:
        snprintf(out, n, lng ? " q Quit " : " q ");
        break;
    }
}

/** tv_button_width() - Columns of button @b: its label, plus the down marker for a menu. */
static int tv_button_width(const tv_state_t *s, int b, int lng)
{
    char lab[48];
    tv_button_label(s, b, lng, lab, sizeof(lab));
    return (int) strlen(lab) + ((b < TV_M_COUNT) ? 2 : 0);
}

/**
 * tv_draw_button - Draw a single menubar button and register its hit rectangle.
 * @s:   Termview state structure.
 * @b:   Button identifier (`0 .. TV_M_COUNT - 1`, `TV_HELP_BTN`, or `TV_QUIT_BTN`).
 * @lng: Non-zero to draw the full label, or 0 for the compact label.
 * @x:   Starting column index on row 0.
 * @end: Exclusive right-hand column limit.
 *
 * Renders the button label (plus the dropdown arrow for menu buttons) with
 * active, hover, or normal colors, records its left edge in `s->ui.btn_x[b]`,
 * and registers a `TV_HIT_BUTTON` region.
 *
 * Return: Nothing.
 */
static void tv_draw_button(tv_state_t *s, int b, int lng, int x, int end)
{
    char lab[48];
    tv_button_label(s, b, lng, lab, sizeof(lab));
    int open =
        (b < TV_M_COUNT) ? (s->ui.open && s->ui.menu == b) : (b == TV_HELP_BTN && s->ui.help);
    int      hov = (s->ui.open || s->ui.help) && s->ui.hover == b + 1; /* motion reports on */
    uint32_t bg  = open ? TV_C_ACCENT : hov ? TV_C_HOVER : TV_C_BAR;
    uint32_t fg  = open ? TV_C_BAR : (b == TV_QUIT_BTN) ? TV_C_DIM : TV_C_TEXT;
    int      x0  = x;
    x            = tv_text(s, 0, x, end, lab, fg, bg);
    if (b < TV_M_COUNT && x < end)
    {
        tv_put(s, 0, x++, TV_G_DOWN, open ? TV_C_BAR : TV_C_DIM, bg);
        if (x < end)
        {
            tv_put(s, 0, x++, ' ', fg, bg);
        }
    }
    s->ui.btn_x[b] = (int16_t) x0;
    tv_hit(s, x0, 0, x - 1, 0, TV_HIT_BUTTON, b);
}

/**
 * tv_render_menubar - Render the top menubar row with menu, help, and quit buttons.
 * @s: Termview state structure.
 *
 * Chooses long or compact button labels to fit `s->cols`, draws the five
 * dropdown menu buttons on the left and the Help/Quit buttons on the right,
 * and shows a subtle keyboard/mouse hint when space permits.
 *
 * Return: Nothing.
 */
static void tv_render_menubar(tv_state_t *s)
{
    static const char hint[] = "Tab: menu  (click)";
    int               cols   = s->cols;
    tv_fill(s, 0, 0, cols, TV_C_BAR);
    tv_hit(s, 0, 0, cols - 1, 0, TV_HIT_CHROME, 0);

    int lng = 1, left = 0, right = 0;
    for (int pass = 0; pass < 2; pass++, lng = 0)
    {
        left  = 0;
        right = tv_button_width(s, TV_HELP_BTN, lng) + tv_button_width(s, TV_QUIT_BTN, lng);
        for (int b = 0; b < TV_M_COUNT; b++)
        {
            left += tv_button_width(s, b, lng);
        }
        if (left + right <= cols)
        {
            break;
        }
    }
    lng      = (left + right <= cols) ? lng : 0;
    int rpos = (cols > right) ? cols - right : 0;
    int x    = 0;
    for (int b = 0; b < TV_M_COUNT && x < rpos; b++)
    {
        tv_draw_button(s, b, lng, x, rpos);
        x += tv_button_width(s, b, lng);
    }
    if (!s->ui.open && !s->ui.help && x + (int) sizeof(hint) < rpos)
    {
        tv_text(s, 0, x + 1, rpos, hint, TV_C_MUTED, TV_C_BAR);
    }
    tv_draw_button(s, TV_HELP_BTN, lng, rpos, cols);
    tv_draw_button(s, TV_QUIT_BTN, lng, rpos + tv_button_width(s, TV_HELP_BTN, lng), cols);
}

/* -----------------------------------------------------------------------
 * Dropdown menu and help box
 * ----------------------------------------------------------------------- */

/** tv_swatch() - Colour of cell @k of colormap @cm's preview strip (computed once). */
static uint32_t tv_swatch(int cm, int k)
{
    static uint32_t sw[COLORMAP_NB][TV_SWATCH];
    static int      ready;
    if (!ready)
    {
        for (int c = 0; c < COLORMAP_NB; c++)
        {
            for (int j = 0; j < TV_SWATCH; j++)
            {
                sw[c][j] = tv_rgb(tv_cmap_color((termview_colormap_t) c, j / (TV_SWATCH - 1.0)));
            }
        }
        ready = 1;
    }
    return sw[cm][k];
}

/**
 * tv_render_menu - Render the currently open dropdown menu box and its items.
 * @s: Termview state structure.
 *
 * Computes the menu box geometry aligned under `s->ui.btn_x[s->ui.menu]`,
 * clamps scrolling so `s->ui.sel` is visible, draws the bordered box with drop
 * shadow, renders each visible choice, toggle, action, or separator (plus
 * colormap swatches in the Cmap menu), and registers hit rectangles.
 *
 * Return: Nothing.
 */
static void tv_render_menu(tv_state_t *s)
{
    const tv_layout_t *l = &s->lay;
    tv_item_t          it[TV_ITEMS];
    int                n  = tv_menu_items(s, s->ui.menu, it);
    int                sw = (s->ui.menu == TV_M_CMAP) ? TV_SWATCH + 1 : 0;
    int                lw = 0;
    for (int k = 0; k < n; k++)
    {
        int w = (int) strlen(it[k].label);
        lw    = (w > lw) ? w : lw;
    }
    int w   = 4 + lw + sw + 4;   /* border, pad, mark, label, swatch, pad, border */
    int vis = l->status_row - 4; /* borders and shadow stay above the status row */
    vis     = (vis > n) ? n : vis;
    if (vis < 1 || n == 0)
    {
        return;
    }
    int x0 = s->ui.btn_x[s->ui.menu];
    x0     = (x0 + w + 1 > s->cols) ? s->cols - w - 1 : x0;
    x0     = (x0 < 0) ? 0 : x0;
    int y0 = 1, x1 = x0 + w - 1, y1 = y0 + vis + 1;

    int *top = &s->ui.top;
    *top     = (s->ui.sel < *top) ? s->ui.sel : *top;
    *top     = (s->ui.sel >= *top + vis) ? s->ui.sel - vis + 1 : *top;
    *top     = (*top > n - vis) ? n - vis : (*top < 0) ? 0 : *top;

    tv_box(s, x0, y0, x1, y1, TV_C_BAR);
    tv_hit(s, x0, y0, x1, y1, TV_HIT_PANEL, 0);
    if (*top > 0)
    {
        tv_put(s, y0, x1 - 2, TV_G_UP, TV_C_ACCENT, TV_C_BAR);
    }
    if (*top + vis < n)
    {
        tv_put(s, y1, x1 - 2, TV_G_DOWN, TV_C_ACCENT, TV_C_BAR);
    }
    for (int i = 0; i < vis; i++)
    {
        int              k   = *top + i;
        int              y   = y0 + 1 + i;
        const tv_item_t *t   = &it[k];
        int              sel = (k == s->ui.sel);
        uint32_t         bg  = sel ? TV_C_SELECT : TV_C_BAR;
        uint32_t         fg  = sel ? TV_C_BRIGHT : TV_C_TEXT;
        if (t->kind == TV_I_SEP)
        {
            for (int x = x0 + 1; x < x1; x++)
            {
                tv_put(s, y, x, TV_G_HLINE, TV_C_MUTED, TV_C_BAR);
            }
            continue;
        }
        tv_fill(s, y, x0 + 1, x1, bg);
        if (t->kind == TV_I_CHOICE && t->on)
        {
            tv_put(s, y, x0 + 2, TV_G_DOT, TV_C_ACCENT, bg);
        }
        else if (t->kind == TV_I_TOGGLE)
        {
            tv_text(s, y, x0 + 2, x1, t->on ? "[x]" : "[ ]", t->on ? TV_C_ACCENT : TV_C_DIM, bg);
        }
        tv_text(s, y, x0 + 6, x1, t->label, fg, bg);
        for (int j = 0; j < sw - 1; j++)
        {
            tv_put(s, y, x0 + 6 + lw + 1 + j, ' ', fg, tv_swatch(k, j));
        }
        tv_hit(s, x0 + 1, y, x1 - 1, y, TV_HIT_ITEM, k);
    }
}

/**
 * tv_render_help - Render the centered modal keyboard and mouse help dialog.
 * @s: Termview state structure.
 *
 * Sizes and centers a bordered dialog box over the image canvas, clamps
 * `s->ui.help_top` to the visible row count, draws scroll indicators when
 * needed, and renders the two-column `tv_help_keys` bindings table.
 *
 * Return: Nothing.
 */
static void tv_render_help(tv_state_t *s)
{
    static const char title[] = " Keys and mouse ";
    static const char foot[]  = " Esc: close ";
    int               kw = 0, dw = 0;
    for (int k = 0; k < tv_help_nkeys; k++)
    {
        int a = (int) strlen(tv_help_keys[k][0]), b = (int) strlen(tv_help_keys[k][1]);
        kw = (a > kw) ? a : kw;
        dw = (b > dw) ? b : dw;
    }
    int w   = kw + dw + 7;
    w       = (w > s->cols - 2) ? s->cols - 2 : w;
    int vis = s->rows - 5; /* menubar, borders and shadow; the status row stays visible */
    vis     = (vis > tv_help_nkeys) ? tv_help_nkeys : vis;
    if (vis < 1 || w < 12)
    {
        return;
    }
    int x0 = (s->cols - w - 1) / 2, y0 = (s->rows - vis - 2) / 2;
    y0     = (y0 < 1) ? 1 : y0;
    int x1 = x0 + w - 1, y1 = y0 + vis + 1;
    int top        = s->ui.help_top;
    top            = (top > tv_help_nkeys - vis) ? tv_help_nkeys - vis : top;
    s->ui.help_top = top;

    tv_box(s, x0, y0, x1, y1, TV_C_BAR);
    tv_hit(s, x0, y0, x1, y1, TV_HIT_PANEL, 0);
    tv_text(s, y0, x0 + 2, x1 - 1, title, TV_C_BRIGHT, TV_C_BAR);
    tv_text(s, y1, x0 + 2, x1 - 1, foot, TV_C_DIM, TV_C_BAR);
    if (top > 0)
    {
        tv_put(s, y0, x1 - 2, TV_G_UP, TV_C_ACCENT, TV_C_BAR);
    }
    if (top + vis < tv_help_nkeys)
    {
        tv_put(s, y1, x1 - 2, TV_G_DOWN, TV_C_ACCENT, TV_C_BAR);
    }
    for (int i = 0; i < vis; i++)
    {
        int y = y0 + 1 + i;
        tv_fill(s, y, x0 + 1, x1, TV_C_BAR);
        tv_text(s, y, x0 + 2, x1, tv_help_keys[top + i][0], TV_C_ACCENT, TV_C_BAR);
        tv_text(s, y, x0 + 4 + kw, x1, tv_help_keys[top + i][1], TV_C_TEXT, TV_C_BAR);
    }
}

/* -----------------------------------------------------------------------
 * Colorbar, status bar, minibuffer
 * ----------------------------------------------------------------------- */

/** tv_labels_ok() - The @n labels fit in @width and are all distinct. */
static int tv_labels_ok(char out[][16], int n, int width)
{
    for (int k = 0; k < n; k++)
    {
        if ((int) strlen(out[k]) > width)
        {
            return 0;
        }
        for (int j = 0; j < k; j++)
        {
            if (strcmp(out[j], out[k]) == 0)
            {
                return 0;
            }
        }
    }
    return 1;
}

/** tv_compact_g() - "%.*g" with a short exponent ("1.5e+07" -> "1.5e7", "2e-05" -> "2e-5"). */
static void tv_compact_g(char *out, int digits, double v)
{
    char  tmp[32];
    char *o = out;
    snprintf(tmp, sizeof(tmp), "%.*g", digits, v);
    for (const char *p = tmp; *p != '\0'; p++)
    {
        *o++ = *p;
        if (*p == 'e')
        {
            p++;
            if (*p == '-')
            {
                *o++ = '-';
            }
            if (*p == '+' || *p == '-')
            {
                p++;
            }
            while (*p == '0' && p[1] != '\0')
            {
                p++;
            }
            p--;
        }
    }
    *o = '\0';
}

/** tv_unsign_zero() - "-0", "-0.00", "-0e0": the sign of a zero carries no information. */
static void tv_unsign_zero(char *t)
{
    if (t[0] == '-' && strtod(t, NULL) == 0.0)
    {
        memmove(t, t + 1, strlen(t));
    }
}

/**
 * tv_labels() - Colorbar values @v[@n] as text no wider than @width, never cut: the most fixed
 * decimals (from those suited to the step down to none), else the most significant digits that
 * fit and stay distinct, else the most that fit.
 */
static void tv_labels(const double *v, int n, int width, char out[][16])
{
    double step = (n > 1) ? fabs(v[0] / (n - 1) - v[n - 1] / (n - 1)) : 0.0;
    int    dec  = (step > 0.0 && tv_finite(step)) ? (int) ceil(-log10(step)) + 1 : 2;
    dec         = (dec < 0) ? 0 : (dec > 6) ? 6 : dec;
    for (int d = dec; d >= 0; d--)
    {
        for (int k = 0; k < n; k++)
        {
            snprintf(out[k], 16, "%.*f", d, v[k]);
            tv_unsign_zero(out[k]);
        }
        if (tv_labels_ok(out, n, width))
        {
            return;
        }
    }
    int fit = 0; /* most digits whose labels fit, distinct or not */
    for (int d = width; d >= 1; d--)
    {
        int fits = 1;
        for (int k = 0; k < n; k++)
        {
            tv_compact_g(out[k], d, v[k]);
            tv_unsign_zero(out[k]);
            fits &= ((int) strlen(out[k]) <= width);
        }
        if (tv_labels_ok(out, n, width))
        {
            return;
        }
        fit = (fits && fit == 0) ? d : fit;
    }
    for (int k = 0; k < n; k++)
    {
        tv_compact_g(out[k], fit ? fit : 1, v[k]);
        tv_unsign_zero(out[k]);
    }
}

/**
 * tv_render_colorbar - Render the vertical colorbar and numeric tick labels.
 * @s: Termview viewer state.
 *
 * Paints the two-column colormap gradient along the right edge of the image
 * viewport and formats up to five evenly spaced value labels (top, bottom,
 * and quarters when height permits) using compact scientific/general notation.
 *
 * Return: void.
 */
static void tv_render_colorbar(tv_state_t *s)
{
    const tv_layout_t *l  = &s->lay;
    int                lc = l->bar_col + 4; /* labels: after gutter, 2 bar columns and a gap */
    for (int i = 0; i < l->img_rows; i++)
    {
        int      r = l->img_top + i;
        double   f = (l->img_rows > 1) ? 1.0 - (double) i / (l->img_rows - 1) : 0.5;
        uint32_t c = tv_rgb(s->lut[tv_lut_idx(f, 0.0, TV_LUT_N - 1)]);
        tv_fill(s, r, l->bar_col, s->cols, TV_C_TERM);
        tv_put(s, r, l->bar_col + 1, TV_G_FULL_BLOCK, c, TV_C_TERM);
        tv_put(s, r, l->bar_col + 2, TV_G_FULL_BLOCK, c, TV_C_TERM);
    }
    /* labels at the top, the bottom, and at quarters when there is room */
    int    nq = (l->img_rows >= 9) ? 4 : (l->img_rows >= 3) ? 2 : 1;
    double v[5];
    int    row[5];
    char   txt[5][16];
    for (int q = 0; q <= nq; q++)
    {
        double t = (double) q / nq; /* no max - min: it overflows for very wide ranges */
        row[q]   = (int) lround(t * (l->img_rows - 1));
        v[q]     = (q == 0)    ? s->cur_max
                   : (q == nq) ? s->cur_min
                               : s->cur_max * (1 - t) + s->cur_min * t;
        /* a zero crossing computes as rounding noise: show it as 0 */
        v[q] = (fabs(v[q]) <= 1e-12 * fmax(fabs(s->cur_max), fabs(s->cur_min))) ? 0.0 : v[q];
    }
    tv_labels(v, nq + 1, s->cols - lc, txt);
    for (int q = 0; q <= nq; q++)
    {
        tv_text(s, l->img_top + row[q], lc, s->cols, txt[q], TV_C_TEXT, TV_C_TERM);
    }
}

/**
 * tv_dtype_name - Return a short display string for an ImageStreamIO datatype.
 * @dt: ImageStreamIO datatype code (_DATATYPE_*).
 *
 * Maps numeric datatype codes to compact status-bar labels such as "f32" or
 * "u16".
 *
 * Return: Static short string describing @dt, or "?" if unknown.
 */
static const char *tv_dtype_name(uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
        return "u8";
    case _DATATYPE_INT8:
        return "i8";
    case _DATATYPE_UINT16:
        return "u16";
    case _DATATYPE_INT16:
        return "i16";
    case _DATATYPE_UINT32:
        return "u32";
    case _DATATYPE_INT32:
        return "i32";
    case _DATATYPE_UINT64:
        return "u64";
    case _DATATYPE_INT64:
        return "i64";
    case _DATATYPE_FLOAT:
        return "f32";
    case _DATATYPE_DOUBLE:
        return "f64";
    default:
        return "?";
    }
}

/**
 * tv_render_mini - Render the inline manual-range editor on the status row.
 * @s: Termview viewer state.
 * @r: Terminal row index of the status bar.
 *
 * Draws the Min and Max text input fields, cursor/selection highlight, mouse
 * hit regions, and either the active validation error message or keyboard
 * usage hints.
 *
 * Return: void.
 */
static void tv_render_mini(tv_state_t *s, int r)
{
    static const char hint[] = "Enter: lock  Tab: other field  Esc: cancel";
    tv_mini_t        *m      = &s->mini;
    int               x      = 1;
    tv_fill(s, r, 0, s->cols, TV_C_BAR);
    tv_hit(s, 0, r, s->cols - 1, r, TV_HIT_CHROME, 0);
    for (int f = 0; f < 2; f++)
    {
        x                 = tv_text(s, r, x, s->cols, f ? "  Max " : "Min ", TV_C_TEXT, TV_C_BAR);
        int         focus = (m->field == f);
        int         x0    = x;
        uint32_t    bg    = (focus && m->fresh[f]) ? TV_C_ACCENT : focus ? TV_C_SELECT : TV_C_FIELD;
        uint32_t    fg    = (focus && m->fresh[f]) ? TV_C_BAR : TV_C_BRIGHT;
        const char *t     = m->text[f];
        size_t      len   = strlen(t);
        t += (len > 12) ? len - 12 : 0; /* the end of a long entry */
        tv_fill(s, r, x, x + 14 < s->cols ? x + 14 : s->cols, TV_C_FIELD);
        x = tv_text(s, r, x + 1, x + 13 < s->cols ? x + 13 : s->cols, t, fg, bg);
        if (focus && !m->fresh[f] && x < s->cols)
        {
            tv_put(s, r, x, '_', TV_C_ACCENT, TV_C_FIELD);
        }
        x = x0 + 14;
        tv_hit(s, x0, r, x - 1, r, TV_HIT_FIELD, f);
    }
    if (m->err)
    {
        tv_text(s, r, x + 2, s->cols, m->err, TV_C_ERR, TV_C_BAR);
    }
    else
    {
        tv_text(s, r, x + 2, s->cols, hint, TV_C_DIM, TV_C_BAR);
    }
}

/** tv_render_status() - Segments by priority, dropped from the end when the row is too short. */
static void tv_render_status(tv_state_t *s, int64_t now_us)
{
    const tv_layout_t *l  = &s->lay;
    const tv_stream_t *st = s->st;
    int                r  = l->status_row;
    if (s->mini.active)
    {
        tv_render_mini(s, r);
        return;
    }
    tv_fill(s, r, 0, s->cols, TV_C_BAR);
    tv_hit(s, 0, r, s->cols - 1, r, TV_HIT_CHROME, 0);

    enum
    {
        NSEG = 9
    };
    char     seg[NSEG][96];
    uint32_t fg[NSEG];
    int64_t  age = tv_stale_age_us(s, now_us) / 1000000;
    if (s->paused || s->gone)
    {
        snprintf(seg[0], sizeof(seg[0]), "%s%s", s->paused ? "PAUSED" : "",
                 s->gone ? (s->paused ? " GONE" : "GONE") : "");
        fg[0] = s->gone ? TV_C_ERR : TV_C_ACCENT;
    }
    else if (age == 0)
    {
        snprintf(seg[0], sizeof(seg[0]), "LIVE");
        fg[0] = TV_C_OK;
    }
    else
    {
        snprintf(seg[0], sizeof(seg[0]), "STALE %lld%s",
                 (long long) ((age < 60)     ? age
                              : (age < 3600) ? age / 60
                                             : age / 3600),
                 (age < 60)     ? "s"
                 : (age < 3600) ? "m"
                                : "h");
        fg[0] = TV_C_WARN;
    }
    snprintf(seg[1], sizeof(seg[1]), "%s", st->name);
    fg[1] = TV_C_STREAM;
    snprintf(seg[2], sizeof(seg[2]), "%7.1f Hz", s->current_stream_fps);
    fg[2] = TV_C_TEXT;
    snprintf(seg[3], sizeof(seg[3]), "[%.4g : %.4g] %s", s->cur_min, s->cur_max,
             s->opt.range_locked ? "LOCK" : tv_range_names[s->opt.range]);
    fg[3] = s->opt.range_locked ? TV_C_WARN : TV_C_TEXT;
    snprintf(seg[4], sizeof(seg[4]), "%ux%u %s", st->xsize, st->ysize, tv_dtype_name(st->datatype));
    fg[4] = TV_C_DIM;
    snprintf(seg[5], sizeof(seg[5]), "CPU %.1f%%", s->current_cpu_usage);
    fg[5] = TV_C_DIM;
    snprintf(seg[6], sizeof(seg[6]), "UI %.0f/%d fps", s->ui_fps, s->target_fps);
    fg[6] = TV_C_DIM;
    snprintf(seg[7], sizeof(seg[7]), "zoom %.2fx", s->view.zoom);
    fg[7] = TV_C_DIM;
    snprintf(seg[8], sizeof(seg[8]), "%s%s%s%s", s->opt.flip_h ? " flipH" : "",
             s->opt.flip_v ? " flipV" : "", s->opt.lowbw ? " lowbw" : "",
             s->opt.range_smooth ? "" : " nosmooth");
    fg[8] = TV_C_ACCENT;

    /* the pill always shows; the others while they fit */
    int nseg = 1, width = 1 + 2 + (int) strlen(seg[0]);
    for (int k = 1; k < NSEG; k++)
    {
        const char *t = (k == 8 && seg[8][0] == ' ') ? seg[8] + 1 : seg[k];
        int         w = (int) strlen(t);
        if (w == 0 || width + 2 + w > s->cols)
        {
            break;
        }
        width += 2 + w;
        nseg = k + 1;
    }
    int x = 1;
    tv_put(s, r, x, TV_G_DOT, fg[0], TV_C_BAR);
    x = tv_text(s, r, x + 2, s->cols, seg[0], fg[0], TV_C_BAR);
    for (int k = 1; k < nseg; k++)
    {
        const char *t = (k == 8) ? seg[8] + 1 : seg[k];
        x             = tv_text(s, r, x + 2, s->cols, t, fg[k], TV_C_BAR);
    }
}

/**
 * tv_ui_render - Render all UI chrome and active overlays onto the screen grid.
 * @s:      Termview viewer state.
 * @now_us: Current monotonic timestamp in microseconds.
 *
 * Resets the mouse hit-region table for the current UI generation, renders the
 * colorbar, menu bar, and status bar (or mini-buffer), and paints any open
 * drop-down menu or help overlay on top.
 *
 * Return: void.
 */
void tv_ui_render(tv_state_t *s, int64_t now_us)
{
    s->ui.nhits    = 0;
    s->ui.hits_gen = s->ui.gen;
    tv_render_colorbar(s);
    tv_render_menubar(s);
    tv_render_status(s, now_us);
    if (s->ui.open)
    {
        tv_render_menu(s);
    }
    if (s->ui.help)
    {
        tv_render_help(s);
    }
}
