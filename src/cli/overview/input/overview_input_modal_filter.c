// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_modal_filter.c
 * @brief Modal regex filter editing and interactive jump-search input handling
 */

#include "overview_input_internal.h"
#include <stdio.h>
#include <string.h>

/* Panel names as the filter messages show them (ov_focus_t index) */
static const char *const ov_panel_title[OV_NB_LIST_PANELS] = { "STREAMS", "PROCESSINFO", "FPS" };

/** ov_panel_names - names of the entries of list panel @p into @names (model order). Return: count. */
static int ov_panel_names(const OV_MODEL *m, ov_focus_t p, const char **names)
{
    int n = (p == OV_FOCUS_STREAMS) ? m->nb_streams
            : (p == OV_FOCUS_PROCS) ? m->nb_procs
            : (p == OV_FOCUS_FPS)   ? m->nb_fps
                                    : 0;
    for (int i = 0; i < n; i++)
    {
        names[i] = (p == OV_FOCUS_STREAMS) ? m->streams[i].name
                   : (p == OV_FOCUS_PROCS) ? m->procs[i].name
                                           : m->fps[i].name;
    }
    return n;
}

/**
 * ov_input__handle_filter_mode - process keyboard events during filter editing.
 * @key: Pressed key code.
 * @lay: Pointer to overview layout structure.
 * @m:   Pointer to current data model snapshot.
 *
 * Return: 1 if key was consumed, 0 otherwise.
 */
int ov_input__handle_filter_mode(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    char *active_filter = lay->filter;

    if (lay->filter_editing)
    {
        /* ESC or Ctrl+C — cancel filter edit, keep previous filter */
        if (key == 27 || key == 3 || key == ctrl('c'))
        {
            lay->filter_editing = 0;
            lay->filter[0]      = '\0';
            lay->filter_cursor  = 0;
            return 1;
        }

        /* ENTER — accept filter */
        if (key == '\n' || key == '\r' || key == OV_KEY_ENTER)
        {
            lay->filter_editing = 0;
            ov_focus_t target   = lay->filter_panel;

            /* Jump mode: jump to first matching item without applying persistent filter */
            if (lay->filter_jump)
            {
                lay->filter_jump = 0;
                if (lay->filter[0] != '\0' && m != NULL)
                {
                    const char *names[OV_MAX_NODES];
                    int         count = ov_panel_names(m, target, names);
                    int         fidx[OV_MAX_NODES];
                    int         n = ov_filter_build(lay->filter, names, count, fidx, OV_MAX_NODES);
                    if (n > 0)
                    {
                        int         found = fidx[0];
                        const char *what  = (target == OV_FOCUS_STREAMS) ? "stream"
                                            : (target == OV_FOCUS_PROCS) ? "process"
                                                                         : "FPS";
                        if (ov_sel_model(lay, m, target, found) >= 0)
                        {
                            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Jumped to %s: %s", what,
                                           names[found]);
                        }
                    }
                    else
                    {
                        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN,
                                       "No match found for jump query: /%s/", lay->filter);
                    }
                }
                lay->filter[0]     = '\0';
                lay->filter_cursor = 0;
                return 1;
            }

            /* Regular filter mode */
            if (target < OV_NB_LIST_PANELS)
            {
                OV_PANEL_STATE *ps = &lay->panel[target];
                snprintf(ps->filter, sizeof(ps->filter), "%s", lay->filter);
                ps->filter_on = (ps->filter[0] != '\0');
                ps->sel       = 0;
                ps->scroll    = 0;
                if (ps->filter_on)
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                   "%s regex filter applied: /%s/ ('f' toggle, ESC clear)",
                                   ov_panel_title[target], ps->filter);
                }
                else
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "%s filter cleared",
                                   ov_panel_title[target]);
                }
            }
            lay->filter_active = ov_filter_any_on(lay);
            lay->filter[0]     = '\0';
            lay->filter_cursor = 0;
            return 1;
        }

        /* Ctrl+U — clear filter text */
        if (key == ctrl('u') || key == 21)
        {
            lay->filter_cursor = 0;
            active_filter[0]   = '\0';
            return 1;
        }

        /* Backspace — delete last char */
        if (key == 127 || key == 8)
        {
            if (lay->filter_cursor > 0)
            {
                lay->filter_cursor--;
                active_filter[lay->filter_cursor] = '\0';
            }
            return 1;
        }

        /* Printable ASCII — append to filter */
        if (key >= 32 && key < 127 && lay->filter_cursor < 62)
        {
            active_filter[lay->filter_cursor] = (char) key;
            lay->filter_cursor++;
            active_filter[lay->filter_cursor] = '\0';
            return 1;
        }

        return 1; /* Ignore other keys during editing */
    }

    /* 'f' or Ctrl+F — toggle active filter without erasing */
    int in_loops = (lay->view == OV_VIEW_LOOPS ||
                    (lay->view == OV_VIEW_DASHBOARD && lay->subtab == OV_SUB_LOOPS &&
                     lay->focus == OV_FOCUS_GRAPH));
    if ((!in_loops && key == 'f') || key == ctrl('f'))
    {
        ov_focus_t target = ov_get_effective_filter_panel(lay);
        if (target == OV_FOCUS_GRAPH)
        {
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                           "Filter available on STREAMS, PROCESSINFO, or FPS panel");
            return 1;
        }

        OV_PANEL_STATE *ps = &lay->panel[target];
        if (ps->filter[0] != '\0')
        {
            ps->filter_on = !ps->filter_on;
            ps->sel       = 0;
            ps->scroll    = 0;
            if (ps->filter_on)
            {
                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "%s regex filter: ON (/%s/)",
                               ov_panel_title[target], ps->filter);
            }
            else
            {
                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                               "%s regex filter: OFF (paused, press 'f' to resume, ESC to clear)",
                               ov_panel_title[target]);
            }
        }
        else
        {
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                           "No %s filter set. Press [/] to set filter", ov_panel_title[target]);
        }
        lay->filter_active = ov_filter_any_on(lay);
        return 1;
    }

    /* '/' — enter filter editing mode (pre-filled with existing filter) */
    if (key == '/')
    {
        ov_focus_t target = ov_get_effective_filter_panel(lay);
        if (target == OV_FOCUS_GRAPH)
        {
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN,
                           "Filter available on STREAMS, PROCESSINFO, or FPS panel");
            return 1;
        }

        lay->filter_panel   = target;
        lay->filter_editing = 1;
        lay->filter_jump    = 0;

        const char *existing = ov_get_panel_filter_pattern(lay, target);
        strncpy(lay->filter, existing, sizeof(lay->filter) - 1);
        lay->filter[sizeof(lay->filter) - 1] = '\0';
        lay->filter_cursor                   = (int) strlen(lay->filter);

        const char *pname = (target == OV_FOCUS_STREAMS) ? "STREAMS"
                            : (target == OV_FOCUS_PROCS) ? "PROCESSINFO"
                                                         : "FPS";
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                       "Type %s regex filter (ENTER=apply, ESC=cancel, Ctrl+U=clear)", pname);
        return 1;
    }

    /* '?' — jump-search mode (#5): filter then
     * jump to first match on Enter */
    if (key == '?')
    {
        ov_focus_t target = ov_get_effective_filter_panel(lay);
        if (target == OV_FOCUS_GRAPH)
        {
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN,
                           "Jump search available on STREAMS, PROCESSINFO, or FPS panel");
            return 1;
        }

        lay->filter_panel   = target;
        lay->filter_editing = 1;
        lay->filter_jump    = 1;
        lay->filter[0]      = '\0';
        lay->filter_cursor  = 0;

        const char *pname = (target == OV_FOCUS_STREAMS) ? "STREAMS"
                            : (target == OV_FOCUS_PROCS) ? "PROCESSINFO"
                                                         : "FPS";
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                       "Type jump query for %s (ENTER=jump, ESC=cancel)", pname);
        return 1;
    }

    /* ESC — clear filter if one is defined */
    if (key == 27)
    {
        ov_focus_t target = ov_get_effective_filter_panel(lay);
        if (ov_has_panel_filter(lay, target))
        {
            ov_clear_panel_filter(lay, target);
            lay->panel[target].sel    = 0;
            lay->panel[target].scroll = 0;
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "%s regex filter cleared",
                           ov_panel_title[target]);
            return 1;
        }
        else if (ov_has_filter(lay))
        {
            ov_clear_all_filters(lay);
            for (int p = 0; p < OV_NB_LIST_PANELS; p++)
            {
                lay->panel[p].sel    = 0;
                lay->panel[p].scroll = 0;
            }
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "All regex filters cleared");
            return 1;
        }
    }

    return 0;
}
