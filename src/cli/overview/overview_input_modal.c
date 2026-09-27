// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_modal.c
 * @brief Modal text input handling (filtering, search, loop renaming).
 */

#include "overview_input_internal.h"

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
                    int         count = 0;
                    const char *names[OV_MAX_NODES];
                    if (target == OV_FOCUS_STREAMS)
                    {
                        count = m->nb_streams;
                        for (int i = 0; i < count; i++)
                        {
                            names[i] = m->streams[i].name;
                        }
                    }
                    else if (target == OV_FOCUS_PROCS)
                    {
                        count = m->nb_procs;
                        for (int i = 0; i < count; i++)
                        {
                            names[i] = m->procs[i].name;
                        }
                    }
                    else if (target == OV_FOCUS_FPS)
                    {
                        count = m->nb_fps;
                        for (int i = 0; i < count; i++)
                        {
                            names[i] = m->fps[i].name;
                        }
                    }
                    int fidx[OV_MAX_NODES];
                    int n = ov_filter_build(lay->filter, names, count, fidx, OV_MAX_NODES);
                    if (n > 0)
                    {
                        int found = fidx[0];
                        if (target == OV_FOCUS_STREAMS)
                        {
                            lay->sel_stream = found;
                            int page_h      = lay->r_streams.height - 3;
                            if (page_h > 0)
                            {
                                if (lay->sel_stream < lay->scroll_stream)
                                {
                                    lay->scroll_stream = lay->sel_stream;
                                }
                                if (lay->sel_stream >= lay->scroll_stream + page_h)
                                {
                                    lay->scroll_stream = lay->sel_stream - page_h + 1;
                                }
                            }
                            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Jumped to stream: %s",
                                           names[found]);
                        }
                        else if (target == OV_FOCUS_PROCS)
                        {
                            lay->sel_proc = found;
                            int page_h    = lay->r_procs.height - 3;
                            if (page_h > 0)
                            {
                                if (lay->sel_proc < lay->scroll_proc)
                                {
                                    lay->scroll_proc = lay->sel_proc;
                                }
                                if (lay->sel_proc >= lay->scroll_proc + page_h)
                                {
                                    lay->scroll_proc = lay->sel_proc - page_h + 1;
                                }
                            }
                            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Jumped to process: %s",
                                           names[found]);
                        }
                        else if (target == OV_FOCUS_FPS)
                        {
                            lay->sel_fps = found;
                            int page_h   = lay->r_fps.height - 3;
                            if (page_h > 0)
                            {
                                if (lay->sel_fps < lay->scroll_fps)
                                {
                                    lay->scroll_fps = lay->sel_fps;
                                }
                                if (lay->sel_fps >= lay->scroll_fps + page_h)
                                {
                                    lay->scroll_fps = lay->sel_fps - page_h + 1;
                                }
                            }
                            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Jumped to FPS: %s",
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
            if (target == OV_FOCUS_STREAMS)
            {
                strncpy(lay->filter_stream, lay->filter, sizeof(lay->filter_stream) - 1);
                lay->filter_stream[sizeof(lay->filter_stream) - 1] = '\0';
                lay->filter_stream_active = (lay->filter_stream[0] != '\0') ? 1 : 0;
                lay->sel_stream           = 0;
                lay->scroll_stream        = 0;
                if (lay->filter_stream_active)
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                   "STREAMS regex filter applied: /%s/ ('f' toggle, ESC clear)",
                                   lay->filter_stream);
                }
                else
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "STREAMS filter cleared");
                }
            }
            else if (target == OV_FOCUS_PROCS)
            {
                strncpy(lay->filter_proc, lay->filter, sizeof(lay->filter_proc) - 1);
                lay->filter_proc[sizeof(lay->filter_proc) - 1] = '\0';
                lay->filter_proc_active = (lay->filter_proc[0] != '\0') ? 1 : 0;
                lay->sel_proc           = 0;
                lay->scroll_proc        = 0;
                if (lay->filter_proc_active)
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                   "PROCESSINFO regex filter applied: /%s/ ('f' toggle, "
                                   "ESC clear)",
                                   lay->filter_proc);
                }
                else
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "PROCESSINFO filter cleared");
                }
            }
            else if (target == OV_FOCUS_FPS)
            {
                strncpy(lay->filter_fps, lay->filter, sizeof(lay->filter_fps) - 1);
                lay->filter_fps[sizeof(lay->filter_fps) - 1] = '\0';
                lay->filter_fps_active                       = (lay->filter_fps[0] != '\0') ? 1 : 0;
                lay->sel_fps                                 = 0;
                lay->scroll_fps                              = 0;
                if (lay->filter_fps_active)
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                   "FPS regex filter applied: /%s/ ('f' toggle, ESC clear)",
                                   lay->filter_fps);
                }
                else
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "FPS filter cleared");
                }
            }
            lay->filter_active =
                (lay->filter_stream_active || lay->filter_proc_active || lay->filter_fps_active);
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
    int in_loops =
        (lay->view == OV_VIEW_LOOPS || (lay->view == OV_VIEW_DASHBOARD &&
                                        lay->graph_tab_mode == 1 && lay->focus == OV_FOCUS_GRAPH));
    if ((!in_loops && key == 'f') || key == ctrl('f'))
    {
        ov_focus_t target = ov_get_effective_filter_panel(lay);
        if (target == OV_FOCUS_GRAPH)
        {
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                           "Filter available on STREAMS, PROCESSINFO, or FPS panel");
            return 1;
        }

        if (target == OV_FOCUS_STREAMS)
        {
            if (lay->filter_stream[0] != '\0')
            {
                lay->filter_stream_active = !lay->filter_stream_active;
                lay->sel_stream           = 0;
                lay->scroll_stream        = 0;
                if (lay->filter_stream_active)
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "STREAMS regex filter: ON (/%s/)",
                                   lay->filter_stream);
                }
                else
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                   "STREAMS regex filter: OFF (paused, press 'f' to resume, "
                                   "ESC to clear)");
                }
            }
            else
            {
                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                               "No STREAMS filter set. Press [/] to set filter");
            }
        }
        else if (target == OV_FOCUS_PROCS)
        {
            if (lay->filter_proc[0] != '\0')
            {
                lay->filter_proc_active = !lay->filter_proc_active;
                lay->sel_proc           = 0;
                lay->scroll_proc        = 0;
                if (lay->filter_proc_active)
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                   "PROCESSINFO regex filter: ON (/%s/)", lay->filter_proc);
                }
                else
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                   "PROCESSINFO regex filter: OFF (paused, press 'f' to resume, "
                                   "ESC to clear)");
                }
            }
            else
            {
                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                               "No PROCESSINFO filter set. Press [/] to set filter");
            }
        }
        else if (target == OV_FOCUS_FPS)
        {
            if (lay->filter_fps[0] != '\0')
            {
                lay->filter_fps_active = !lay->filter_fps_active;
                lay->sel_fps           = 0;
                lay->scroll_fps        = 0;
                if (lay->filter_fps_active)
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "FPS regex filter: ON (/%s/)",
                                   lay->filter_fps);
                }
                else
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                   "FPS regex filter: OFF (paused, press 'f' to resume, "
                                   "ESC to clear)");
                }
            }
            else
            {
                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                               "No FPS filter set. Press [/] to set filter");
            }
        }
        lay->filter_active =
            (lay->filter_stream_active || lay->filter_proc_active || lay->filter_fps_active);
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
            if (target == OV_FOCUS_STREAMS)
            {
                lay->sel_stream    = 0;
                lay->scroll_stream = 0;
                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "STREAMS regex filter cleared");
            }
            else if (target == OV_FOCUS_PROCS)
            {
                lay->sel_proc    = 0;
                lay->scroll_proc = 0;
                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "PROCESSINFO regex filter cleared");
            }
            else if (target == OV_FOCUS_FPS)
            {
                lay->sel_fps    = 0;
                lay->scroll_fps = 0;
                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "FPS regex filter cleared");
            }
            return 1;
        }
        else if (ov_has_filter(lay))
        {
            ov_clear_all_filters(lay);
            lay->sel_stream    = 0;
            lay->scroll_stream = 0;
            lay->sel_proc      = 0;
            lay->scroll_proc   = 0;
            lay->sel_fps       = 0;
            lay->scroll_fps    = 0;
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "All regex filters cleared");
            return 1;
        }
    }

    return 0;
}


/**
 * ov_input__handle_loop_rename - process keyboard events during loop renaming.
 * @key: Pressed key code.
 * @lay: Pointer to overview layout structure.
 * @m:   Pointer to current data model snapshot.
 *
 * Return: 1 if key was consumed, 0 otherwise.
 */
int ov_input__handle_loop_rename(int key, OV_LAYOUT *lay, OV_MODEL *m)
{
    if (!lay->renaming_loop)
    {
        return 0;
    }

    /* ESC or Ctrl+C — cancel loop rename */
    if (key == 27 || key == 3 || key == ctrl('c'))
    {
        lay->renaming_loop = 0;
        return 1;
    }

    /* ENTER — commit loop rename */
    if (key == '\n' || key == '\r' || key == OV_KEY_ENTER)
    {
        lay->renaming_loop = 0;
        if (lay->sel_loop >= 0 && lay->sel_loop < m->nb_loops)
        {
            ov_loop_rename(m, lay->sel_loop, lay->rename_buf);
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Loop L%02d renamed to '%s'",
                           m->loops[lay->sel_loop].loop_id, m->loops[lay->sel_loop].name);
        }
        return 1;
    }

    /* Ctrl+U — clear buffer */
    if (key == ctrl('u') || key == 21)
    {
        lay->rename_cursor = 0;
        lay->rename_buf[0] = '\0';
        return 1;
    }

    /* Backspace — delete character */
    if (key == 127 || key == 8)
    {
        if (lay->rename_cursor > 0)
        {
            lay->rename_cursor--;
            lay->rename_buf[lay->rename_cursor] = '\0';
        }
        return 1;
    }

    /* Printable ASCII */
    if (key >= 32 && key < 127 && lay->rename_cursor < (int) sizeof(lay->rename_buf) - 2)
    {
        lay->rename_buf[lay->rename_cursor++] = (char) key;
        lay->rename_buf[lay->rename_cursor]   = '\0';
        return 1;
    }

    return 1;
}


/**
 * ov_input__handle_loop_actions - process loop view action keys (rename, filter, graph).
 * @key: Pressed key code.
 * @lay: Pointer to overview layout structure.
 * @m:   Pointer to current data model snapshot.
 *
 * Return: 1 if key was consumed, 0 otherwise.
 */
int ov_input__handle_loop_actions(int key, OV_LAYOUT *lay, OV_MODEL *m)
{
    int in_loops =
        (lay->view == OV_VIEW_LOOPS || (lay->view == OV_VIEW_DASHBOARD &&
                                        lay->graph_tab_mode == 1 && lay->focus == OV_FOCUS_GRAPH));
    if (!in_loops)
    {
        return 0;
    }

    /* 'r' or 'R' — rename selected loop */
    if (key == 'r' || key == 'R')
    {
        if (lay->sel_loop >= 0 && lay->sel_loop < m->nb_loops)
        {
            lay->renaming_loop = 1;
            const OV_LOOP *lp  = &m->loops[lay->sel_loop];
            snprintf(lay->rename_buf, sizeof(lay->rename_buf), "%s",
                     lp->has_custom_name ? lp->custom_name : lp->name);
            lay->rename_cursor = (int) strlen(lay->rename_buf);
            return 1;
        }
    }

    /* 'f' — toggle loop isolation filter */
    if (key == 'f')
    {
        lay->loop_filter_active = !lay->loop_filter_active;
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                       lay->loop_filter_active ? "Loop isolation filter: ON"
                                               : "Loop isolation filter: OFF");
        return 1;
    }

    /* 'g' — switch to graph CONNECTIONS tab to view circuit tree */
    if (key == 'g')
    {
        lay->graph_tab_mode = 0;
        if (lay->view == OV_VIEW_LOOPS)
        {
            lay->view = OV_VIEW_GRAPH;
        }
        return 1;
    }

    return 0;
}
