// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input.c
 * @brief Top-level keyboard and event router for milk-CTRL.
 */

#include "overview_input_internal.h"

/**
 * ov_handle_key_internal - route a key or mouse event to the appropriate subsystem.
 * @key: Key or mouse event code.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to current data model snapshot.
 *
 * Return: 1 if key was consumed and screen needs update, 0 if unhandled/ignored.
 */
static int ov_handle_key_internal(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (key == OV_KEY_NONE)
    {
        return 0;
    }

    /* 0. Theme selector popup: handle navigation, selection, and auto-dismissal */
    if (lay->theme_popup_active)
    {
        struct timespec now_ts;
        clock_gettime(CLOCK_MONOTONIC, &now_ts);
        double elapsed = (now_ts.tv_sec - lay->theme_popup_ts.tv_sec) +
                         (now_ts.tv_nsec - lay->theme_popup_ts.tv_nsec) * 1e-9;
        if (elapsed >= 1.0)
        {
            lay->theme_popup_active = 0;
        }
        else if (key == OV_KEY_ESC)
        {
            lay->theme_popup_active = 0;
            return 0;
        }
        else if (key == '\n' || key == '\r' || key == 10 || key == 13)
        {
            lay->theme_popup_active = 0;
            return 0;
        }
        else if (key == OV_KEY_UP || key == 'k' || key == OV_KEY_MOUSE_UP)
        {
            int count            = ov_theme_count();
            lay->theme_popup_sel = (lay->theme_popup_sel - 1 + count) % count;
            ov_theme_set(lay->theme_popup_sel);
            clock_gettime(CLOCK_MONOTONIC, &lay->theme_popup_ts);
            const ov_theme_t *th = ov_theme_get_active();
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "🎨 Theme: %s (%s)", th->name, th->desc);
            return 0;
        }
        else if (key == OV_KEY_DOWN || key == 'j' || key == OV_KEY_MOUSE_DOWN)
        {
            int count            = ov_theme_count();
            lay->theme_popup_sel = (lay->theme_popup_sel + 1) % count;
            ov_theme_set(lay->theme_popup_sel);
            clock_gettime(CLOCK_MONOTONIC, &lay->theme_popup_ts);
            const ov_theme_t *th = ov_theme_get_active();
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "🎨 Theme: %s (%s)", th->name, th->desc);
            return 0;
        }
        else if (key == ctrl('t') || key == OV_KEY_F8)
        {
            int count            = ov_theme_count();
            lay->theme_popup_sel = (lay->theme_popup_sel + 1) % count;
            ov_theme_set(lay->theme_popup_sel);
            clock_gettime(CLOCK_MONOTONIC, &lay->theme_popup_ts);
            const ov_theme_t *th = ov_theme_get_active();
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "🎨 Theme: %s (%s)", th->name, th->desc);
            return 0;
        }
        else if (key == OV_KEY_MOUSE_CLICK)
        {
            int mr = ov_mouse_row;
            int mc = ov_mouse_col;
            if (mr >= lay->r_theme_popup.row &&
                mr < lay->r_theme_popup.row + lay->r_theme_popup.height &&
                mc >= lay->r_theme_popup.col &&
                mc < lay->r_theme_popup.col + lay->r_theme_popup.width)
            {
                int item_idx = mr - (lay->r_theme_popup.row + 1);
                if (item_idx >= 0 && item_idx < ov_theme_count())
                {
                    lay->theme_popup_sel = item_idx;
                    ov_theme_set(item_idx);
                    clock_gettime(CLOCK_MONOTONIC, &lay->theme_popup_ts);
                    const ov_theme_t *th = ov_theme_get_active();
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "🎨 Theme: %s (%s)", th->name,
                                   th->desc);
                    return 0;
                }
            }
            lay->theme_popup_active = 0;
            return 0;
        }
        else
        {
            /* Other keys dismiss the popup and fall through */
            lay->theme_popup_active = 0;
        }
    }

    /* 1. Interactive help overlay: when open, modal navigation and search only */
    if (ov_input_handle_help_key(key, lay))
    {
        return 0;
    }

    /* 1b. Loop rename mode: all characters belong to the rename prompt */
    if (lay->renaming_loop)
    {
        ov_input__handle_loop_rename(key, lay, (OV_MODEL *) m);
        return 0;
    }

    /* 2. Filter editing mode: all characters belong to the filter prompt */
    if (lay->filter_editing)
    {
        ov_input__handle_filter_mode(key, lay, m);
        return 0;
    }

    /* 3. Filter activation ('/'), toggle ('f'), and filter clearing (ESC) */
    if (key == '/' || key == '?' || key == 'f' || key == ctrl('f') ||
        (key == 27 && ov_has_filter(lay)))
    {
        if (ov_input__handle_filter_mode(key, lay, m))
        {
            return 0;
        }
    }

    /* ESC — clear loop filter if active */
    if (key == 27 && lay->loop_filter_active)
    {
        lay->loop_filter_active = 0;
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Loop isolation filter: OFF");
        return 0;
    }

    /* Standalone ESC with no active filter: ignore silently */
    if (key == 27)
    {
        return 0;
    }

    /* 4. Global quit */
    if (key == 'q' || key == 'x' || key == 3 || key == ctrl('c') || key == 4 || key == ctrl('d') ||
        key == 24 || key == ctrl('x') || key == 17 || key == ctrl('q'))
    {
        return 1;
    }

    /* 5. Command log panel toggle — 'G' */
    if (key == 'G')
    {
        if (lay->cmdlog_rows > 0)
        {
            lay->cmdlog_rows = 0;
        }
        else
        {
            lay->cmdlog_rows = 4;
        }
        ov_buf_force_clear();
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Command log %s",
                       lay->cmdlog_rows > 0 ? "shown" : "hidden");
        return 0;
    }

    /* 6. Control mode toggle — 'c' */
    if (key == 'c')
    {
        lay->ctrl_mode = !lay->ctrl_mode;
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Control mode %s",
                       lay->ctrl_mode ? "ON" : "OFF");
        return 0;
    }

    /* 7. Mouse hover toggle — 'm' */
    if (key == 'm')
    {
        lay->mouse_hover = !lay->mouse_hover;
        ov_set_mouse_hover(lay->mouse_hover);
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Mouse hover %s",
                       lay->mouse_hover ? "ON" : "OFF");
        if (lay->mouse_hover)
        {
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN,
                           "Warning: Hover uses extra CPU on slow connections");
        }
        return 0;
    }

    /* 8. Help toggle — 'h' */
    if (key == 'h')
    {
        ov_help_open(lay);
        return 0;
    }
    int mouse_res = ov_input__handle_mouse(key, lay, m);
    if (mouse_res == 2)
    {
        return 1;
    }
    if (mouse_res)
    {
        return 0;
    }
    if (ov_input__handle_view_switch(key, lay))
    {
        return 0;
    }
    if (ov_input__handle_misc_toggles(key, lay, m))
    {
        return 0;
    }
    if (ov_input__handle_column_highlights(key, lay, m))
    {
        return 0;
    }
    if (ov_input__handle_sorting(key, lay))
    {
        return 0;
    }
    if (ov_input__handle_actions(key, lay, m))
    {
        return 0;
    }
    if (ov_input__handle_loop_actions(key, lay, (OV_MODEL *) m))
    {
        return 0;
    }

    if (ov_input__handle_ancestry_nav(key, lay, m) || ov_input__handle_navigation(key, lay, m))
    {
        /* Perform auto-scrolling if a navigation key was pressed */
        int *sel    = NULL;
        int *scroll = NULL;
        int  page_h = 10;
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS:
            sel    = &lay->sel_stream;
            scroll = &lay->scroll_stream;
            page_h = lay->r_streams.height - 3;
            break;
        case OV_FOCUS_PROCS:
            sel    = &lay->sel_proc;
            scroll = &lay->scroll_proc;
            page_h = lay->r_procs.height - 3;
            break;
        case OV_FOCUS_FPS:
            sel    = &lay->sel_fps;
            scroll = &lay->scroll_fps;
            page_h = lay->r_fps.height - 3;
            break;
        case OV_FOCUS_GRAPH:
            if (lay->graph_tab_mode == 1 || lay->view == OV_VIEW_LOOPS)
            {
                sel    = &lay->sel_loop;
                scroll = &lay->scroll_loop;
                page_h = (lay->r_graph.height - 3 >= 6)
                             ? ((lay->r_graph.height - 3 > 8) ? ((lay->r_graph.height - 3) / 2) : 3)
                             : (lay->r_graph.height - 3);
            }
            else
            {
                sel    = &lay->sel_graph;
                scroll = &lay->scroll_graph;
                page_h = lay->r_graph.height - 3;
            }
            break;
        default:
            break;
        }

        if (sel != NULL && scroll != NULL && page_h > 0)
        {
            if (*sel < *scroll)
            {
                *scroll = *sel;
            }
            if (*sel >= *scroll + page_h)
            {
                *scroll = *sel - page_h + 1;
            }
        }
        return 0;
    }

    if (key == '?')
    {
        char keys_str[128];
        char ctrl_str[128] = "";

        // Base keys always available
        snprintf(keys_str, sizeof(keys_str),
                 "Keys: q c h G v V ? TAB D L F W i <>[]sS +-= F2-F6 / ESC");

        if (lay->ctrl_mode)
        {
            if (lay->focus == OV_FOCUS_FPS)
            {
                snprintf(ctrl_str, sizeof(ctrl_str), " | CTRL (FPS): r s k K ^e");
            }
            else if (lay->focus == OV_FOCUS_PROCS)
            {
                snprintf(ctrl_str, sizeof(ctrl_str), " | CTRL (PROC): p ^s e z C k K ^e");
            }
            else if (lay->focus == OV_FOCUS_STREAMS)
            {
                snprintf(ctrl_str, sizeof(ctrl_str), " | CTRL (STRM): DEL ^e");
            }
        }

        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "%s%s", keys_str, ctrl_str);
        if (lay->cmdlog_rows == 0)
        {
            lay->cmdlog_rows = 4;
            ov_buf_force_clear();
        }
        return 0;
    }

    if (key == OV_KEY_SHIFT_UP || key == OV_KEY_SHIFT_DOWN || key == OV_KEY_SHIFT_LEFT ||
        key == OV_KEY_SHIFT_RIGHT || key == OV_KEY_CTRL_LEFT || key == OV_KEY_CTRL_RIGHT ||
        key == OV_KEY_BTAB || key == 27)
    {
        return 0;
    }

    if (key >= 32 && key <= 126)
    {
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN, "Unmapped key: '%c' (code %d)", key, key);
    }
    else
    {
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN, "Unmapped key code: %d", key);
    }
    return 0;
}

/**
 * ov_handle_key - process one key or mouse event and manage session history.
 * @key: Keycode from ov_get_key().
 * @lay: Mutable layout state.
 * @m:   Current data model snapshot (read-only).
 *
 * Return: 0 to continue, 1 if quit was requested.
 */
int ov_handle_key(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (key == OV_KEY_NONE)
    {
        return 0;
    }

    char          old_fps_name[80] = { 0 };
    const OV_FPS *old_fps          = ov_input_get_sel_fps(lay, m);
    if (old_fps != NULL)
    {
        strncpy(old_fps_name, old_fps->name, sizeof(old_fps_name) - 1);
    }

    int old_view = lay->view;

    int ret = ov_handle_key_internal(key, lay, m);

    if (lay->view == OV_VIEW_FPS)
    {
        char          new_fps_name[80] = { 0 };
        const OV_FPS *new_fps          = ov_input_get_sel_fps(lay, m);
        if (new_fps != NULL)
        {
            strncpy(new_fps_name, new_fps->name, sizeof(new_fps_name) - 1);
        }

        if (old_view != OV_VIEW_FPS || strcmp(old_fps_name, new_fps_name) != 0)
        {
            if (old_view == OV_VIEW_FPS && old_fps_name[0] != '\0')
            {
                ov_input_save_fps_history(lay, old_fps_name);
            }
            if (new_fps_name[0] != '\0')
            {
                ov_input_load_fps_history(lay, new_fps_name);
                lay->fps_param_focus = 0;
            }
        }
    }
    else if (old_view == OV_VIEW_FPS)
    {
        if (old_fps_name[0] != '\0')
        {
            ov_input_save_fps_history(lay, old_fps_name);
        }
    }

    return ret;
}
