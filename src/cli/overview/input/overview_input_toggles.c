// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_toggles.c
 * @brief View switching, option toggles, interval controls, and dashboard focus navigation
 */

#include "overview_input_internal.h"
#include <limits.h>
#include <errno.h>

/** view_fkey - F2..F7: the view bound to that key. */
static int view_fkey(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) m;
    ov_set_view(lay, (ov_view_t) ov_view_by_fkey(key - OV_KEY_F1 + 1));
    return 1;
}

/** view_cycle - Ctrl+LEFT / RIGHT: previous / next view. */
static int view_cycle(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) m;
    ov_set_view(lay, (ov_view_t) ov_view_cycle(lay->view, (key == OV_KEY_CTRL_LEFT) ? -1 : 1));
    return 1;
}

/** focus_cycle - TAB: next panel on screen (DASH: the four; F5: list / tree; elsewhere nothing). */
static int focus_cycle(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    if (lay->view == OV_VIEW_DASHBOARD)
    {
        lay->focus = (ov_focus_t) (((int) lay->focus + 1) % OV_FOCUS_COUNT);
    }
    else if (lay->view == OV_VIEW_FPS)
    {
        lay->fps_param_focus = !lay->fps_param_focus;
    }
    else if (lay->view == OV_VIEW_STREAMS)
    {
        lay->strm_insp_focus = !lay->strm_insp_focus && lay->r_strm_insp.width > 0;
    }
    return 1;
}

/** subtab_cycle - Shift+TAB: next sub-tab; in F3 the inspector's next page. */
static int subtab_cycle(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    if (lay->view == OV_VIEW_STREAMS)
    {
        ov_insp_tab_step(lay, 1);
        return 1;
    }
    lay->subtab = (ov_subtab_t) ((lay->subtab + 1) % OV_SUBTAB_COUNT);
    return 1;
}

static const OV_KEYBIND view_keys[] = {
    { { OV_KEY_F2, OV_KEY_F3, OV_KEY_F4, OV_KEY_F5, OV_KEY_F6, OV_KEY_F7 }, 0, 0, view_fkey },
    { { OV_KEY_CTRL_LEFT, OV_KEY_CTRL_RIGHT }, 0, 0, view_cycle },
    { { OV_KEY_TAB }, 0, 0, focus_cycle },
    { { OV_KEY_BTAB }, 0, 0, subtab_cycle },
};

/** ov_input__handle_view_switch - views (F2..F7, Ctrl+arrows), TAB and Shift+TAB. Return: 1 consumed. */
int ov_input__handle_view_switch(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    return ov_key_dispatch(view_keys, OV_KEYBIND_N(view_keys), key, lay, m);
}

/**
 * ov_view_details_hint - in a view without the graph panel, say where DETAILS lives.
 * Return: 1 when the hint was given (key consumed, nothing toggled).
 */
static int ov_view_details_hint(OV_LAYOUT *lay)
{
    if (lay->view == OV_VIEW_DASHBOARD || lay->view == OV_VIEW_GRAPH)
    {
        return 0;
    }
    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "DETAILS is shown in F2:DASH and F6:GRAPH");
    return 1;
}

/** quit - q, x, ^C, ^D, ^Q, ^X. */
static int quit(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) lay, (void) m;
    return OV_KEY_QUIT;
}

/** cmdlog_toggle - G: show / hide the command log. */
static int cmdlog_toggle(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    lay->cmdlog_rows = (lay->cmdlog_rows > 0) ? 0 : 4;
    ov_buf_force_clear();
    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Command log %s",
                   lay->cmdlog_rows > 0 ? "shown" : "hidden");
    return 1;
}

/** ctrl_toggle - c: CONTROL mode on / off. */
static int ctrl_toggle(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    lay->ctrl_mode = !lay->ctrl_mode;
    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Control mode %s", lay->ctrl_mode ? "ON" : "OFF");
    return 1;
}

/** hover_toggle - m: mouse hover tracking on / off. */
static int hover_toggle(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    lay->mouse_hover = !lay->mouse_hover;
    ov_set_mouse_hover(lay->mouse_hover);
    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Mouse hover %s", lay->mouse_hover ? "ON" : "OFF");
    if (lay->mouse_hover)
    {
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN,
                       "Warning: Hover uses extra CPU on slow connections");
    }
    return 1;
}

/** help_open - h, F1: the help overlay. */
static int help_open(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    ov_help_open(lay);
    return 1;
}

static const OV_KEYBIND global_keys[] = {
    { { 'q', 'x', ctrl('c'), ctrl('d'), ctrl('x'), ctrl('q') }, 0, 0, quit },
    { { 'G' }, 0, 0, cmdlog_toggle },
    { { 'c' }, 0, 0, ctrl_toggle },
    { { 'm' }, 0, 0, hover_toggle },
    { { 'h', OV_KEY_F1 }, 0, 0, help_open },
};

/** ov_input__handle_global_keys - quit, log, CONTROL, hover, help. Return: 1 consumed, OV_KEY_QUIT. */
int ov_input__handle_global_keys(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    return ov_key_dispatch(global_keys, OV_KEYBIND_N(global_keys), key, lay, m);
}

/** scan_rate - + / = faster, - slower. */
static int scan_rate(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) lay, (void) m;
    ov_scan_set_interval(ov_scan_get_interval() * ((key == '-') ? 1.4f : 0.7f));
    return 1;
}

/** details_toggle - D: DETAILS / CONNECTIONS sub-tab. */
static int details_toggle(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    if (!ov_view_details_hint(lay))
    {
        lay->subtab = (lay->subtab == OV_SUB_DETAILS) ? OV_SUB_CONN : OV_SUB_DETAILS;
    }
    return 1;
}

/** compact_toggle - d: compact columns. */
static int compact_toggle(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    lay->compact_mode = !lay->compact_mode;
    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                   lay->compact_mode ? "Compact mode ON" : "Compact mode OFF");
    return 1;
}

/** lineage_cycle - L: trigger, input, full lineage. */
static int lineage_cycle(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    lay->lineage_mode = (lay->lineage_mode + 1) % 3;
    return 1;
}

/** cmdlog_resize - v / V: command log one row smaller / larger (V shows a hidden log). */
static int cmdlog_resize(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) m;
    int prev = lay->cmdlog_rows;
    int rows = (prev == 0 && key == 'V') ? 4 : prev + ((key == 'v') ? -1 : 1);
    rows     = (rows < 0) ? 0 : rows;
    rows     = (rows > lay->term_rows / 2) ? lay->term_rows / 2 : rows;
    if (rows != prev)
    {
        lay->cmdlog_rows = rows;
        ov_buf_force_clear();
    }
    return 1;
}

/** nudge - move split ratio *@r by @step within [0.1, 0.9]. */
static void nudge(float *r, float step)
{
    *r += step;
    *r = (*r < 0.1f) ? 0.1f : (*r > 0.9f) ? 0.9f : *r;
}

/** split_v - { / }: F5 list / tree split, DASH vertical split. */
static int split_v(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) m;
    nudge((lay->view == OV_VIEW_FPS)       ? &lay->fps_split_ratio
          : (lay->view == OV_VIEW_STREAMS) ? &lay->strm_split_ratio
                                           : &lay->dash_split_v_ratio,
          (key == '{') ? -0.05f : 0.05f);
    return 1;
}

/** split_h - ( / ): DASH horizontal split. */
static int split_h(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) m;
    nudge(&lay->dash_split_h_ratio, (key == '(') ? -0.05f : 0.05f);
    return 1;
}

/** pause_toggle - F: pause / resume the display. */
static int pause_toggle(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    lay->paused = !lay->paused;
    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Display %s", lay->paused ? "paused" : "resumed");
    return 1;
}

/** snapshot - W: write a snapshot file and log its path. */
static int snapshot(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key;
    char path[PATH_MAX];
    if (ov_model_export_snapshot(m, path, sizeof(path)) == 0)
    {
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_OK, "Snapshot written to %s", path);
    }
    else
    {
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_FAIL, "Snapshot failed: %s", strerror(errno));
    }
    return 1;
}

/** theme_next - F8, ^T: next theme, with the selector popup. */
static int theme_next(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    int next = (ov_theme_get_active_index() + 1) % ov_theme_count();
    ov_theme_set(next);
    lay->theme_popup.sel    = next;
    lay->theme_popup.active = 1;
    clock_gettime(CLOCK_MONOTONIC, &lay->theme_popup.ts);
    const ov_theme_t *th = ov_theme_get_active();
    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Theme: %s (%s)", th->name, th->desc);
    return 1;
}

/** loop_filter - ENTER on a loop list: lists show the selected loop's members only. */
static int loop_filter(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    lay->loop_filter_active = !lay->loop_filter_active;
    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                   lay->loop_filter_active ? "Loop isolation filter: ON"
                                           : "Loop isolation filter: OFF");
    return 1;
}

/** enter_details - ENTER: DETAILS of a list entry; on the sub-tab panel, toggle DETAILS. */
static int enter_details(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    if (!ov_view_details_hint(lay))
    {
        lay->subtab = (lay->focus != OV_FOCUS_GRAPH || lay->subtab != OV_SUB_DETAILS)
                          ? OV_SUB_DETAILS
                          : OV_SUB_CONN;
    }
    return 1;
}

/** freeze_toggle - SPACE: freeze the selection (relations stay highlighted), or release it. */
static int freeze_toggle(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    lay->freeze = !lay->freeze;
    if (lay->freeze)
    {
        lay->freeze_focus = lay->focus;
        for (int p = 0; p < OV_NB_LIST_PANELS; p++)
        {
            lay->panel[p].freeze_row = lay->panel[p].sel;
            lay->panel[p].freeze_tgt = lay->panel[p].tgt;
        }
    }
    return 1;
}

/** focus_adjacent - LEFT / RIGHT in DASH: the neighbouring panel (streams, procs, FPS, sub-tabs, wrap). */
static int focus_adjacent(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) m;
    int step   = (key == OV_KEY_LEFT) ? OV_FOCUS_COUNT - 1 : 1;
    lay->focus = (ov_focus_t) (((int) lay->focus + step) % OV_FOCUS_COUNT);
    return 1;
}

/** swallow - consume the key (LEFT / RIGHT where there is no neighbouring panel). */
static int swallow(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) lay, (void) m;
    return 1;
}

static const OV_KEYBIND toggle_keys[] = {
    { { '+', '=', '-' }, 0, 0, scan_rate },
    { { 'D' }, 0, 0, details_toggle },
    { { 'd' }, 0, 0, compact_toggle },
    { { 'L' }, 0, 0, lineage_cycle },
    { { 'v', 'V' }, 0, 0, cmdlog_resize },
    { { '{', '}' }, OV_CTX_F5, 0, split_v },
    { { '{', '}' }, OV_CTX_DASH, 0, split_v },
    { { '(', ')' }, OV_CTX_DASH, 0, split_h },
    { { 'F' }, 0, 0, pause_toggle },
    { { 'W' }, 0, 0, snapshot },
    { { OV_KEY_F8, ctrl('t') }, 0, 0, theme_next },
    { { '\n', '\r' }, OV_CTX_LOOP_LIST, 0, loop_filter },
    { { '\n', '\r' },
      OV_CTX_MODEL,
      OV_CTX_F5 | OV_CTX_F3,
      enter_details }, /* F5: ENTER edits, F3: enters the inspector (navigation group) */
    { { ' ' }, 0, 0, freeze_toggle },
    { { OV_KEY_LEFT, OV_KEY_RIGHT }, OV_CTX_DASH, 0, focus_adjacent },
    { { OV_KEY_LEFT, OV_KEY_RIGHT },
      0,
      OV_CTX_F5 | OV_CTX_F3,
      swallow }, /* F5: tree up / down, F3: inspector in / out (navigation group) */
};

/** ov_input__handle_misc_toggles - display toggles, splits, freeze, ENTER and LEFT / RIGHT. Return: 1 consumed. */
int ov_input__handle_misc_toggles(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    return ov_key_dispatch(toggle_keys, OV_KEYBIND_N(toggle_keys), key, lay, m);
}
