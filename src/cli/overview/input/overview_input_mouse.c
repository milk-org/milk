// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_mouse.c
 * @brief Mouse event dispatcher, double-click timing, header badges, and tab clicks
 */

#include "overview_input_internal.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

/**
 * ov_input__handle_mouse - dispatch mouse events (click, drag, double-click, wheel).
 * @key: Mouse event key code (OV_KEY_MOUSE_*).
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: 1 if mouse event was consumed, 0 otherwise, 2 if exit requested.
 */
int ov_input__handle_mouse(
    int             key,
    OV_LAYOUT      *lay,
    const OV_MODEL *m)
{
    if (key == OV_KEY_MOUSE_CLICK)
    {
        int mr = ov_mouse_row;
        int mc = ov_mouse_col;

        static struct timespec last_click_ts = { 0, 0 };
        static int             last_click_r  = -1;
        static int             last_click_c  = -1;
        struct timespec        now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        int is_dbl = 0;
        if (mr == last_click_r && mc == last_click_c)
        {
            double dt =
                (now.tv_sec - last_click_ts.tv_sec) + (now.tv_nsec - last_click_ts.tv_nsec) / 1e9;
            if (dt < 0.3)
            {
                is_dbl       = 1;
                last_click_r = -1;
            }
        }
        if (!is_dbl)
        {
            last_click_ts = now;
            last_click_r  = mr;
            last_click_c  = mc;
        }

        /* Check for status bar exit and theme button clicks */
        if (mr == lay->term_rows)
        {
            int col_exit = lay->term_cols - 18;
            if (mc >= col_exit && mc < col_exit + 10)
            {
                return 2; /* exit request */
            }

            char th_buf[32];
            snprintf(th_buf, sizeof(th_buf), " [%s] ", ov_active_theme->id);
            int n_th   = (int) strlen(th_buf);
            int col_th = col_exit - n_th;
            if (mc >= col_th && mc < col_exit)
            {
                int count = ov_theme_count();
                int next  = (ov_theme_get_active_index() + 1) % count;
                ov_theme_set(next);
                lay->theme_popup_sel    = next;
                lay->theme_popup_active = 1;
                clock_gettime(CLOCK_MONOTONIC, &lay->theme_popup_ts);
                const ov_theme_t *th = ov_theme_get_active();
                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "🎨 Theme: %s (%s)", th->name,
                               th->desc);
                return 1;
            }
        }

        /* Check for header tab clicks */
        if (mr == lay->r_header.row && mc >= 1)
        {
            /* Check for CTRL mode toggle click */
            int commit_w    = (int) strlen(MILK_GIT_COMMIT) + 3;
            int shmdir_w    = (int) strlen(ov_get_shmdir()) + 8;
            int badge_start = lay->r_header.col + 18 + commit_w + shmdir_w;
            int badge_w     = lay->ctrl_mode ? 13 : 15;
            if (mc >= badge_start && mc < badge_start + badge_w)
            {
                lay->ctrl_mode = !lay->ctrl_mode;
                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Control mode %s",
                               lay->ctrl_mode ? "ON" : "OFF");
                return 1;
            }

            /* Check for HOVER badge click */
            int hover_badge_start = badge_start + badge_w + 1;
            int hover_badge_w     = lay->mouse_hover ? 15 : 16;
            if (mc >= hover_badge_start && mc < hover_badge_start + hover_badge_w)
            {
                lay->mouse_hover = !lay->mouse_hover;
                ov_set_mouse_hover(lay->mouse_hover);
                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "Mouse hover %s",
                               lay->mouse_hover ? "ON" : "OFF");
                return 1;
            }

            /* Check for header filter badge clicks */
            for (int fb = 0; fb < lay->r_filter_count; fb++)
            {
                int fb_start = lay->r_filter_start[fb];
                int fb_w     = lay->r_filter_width[fb];
                if (mc >= fb_start && mc < fb_start + fb_w)
                {
                    ov_focus_t fpanel = lay->r_filter_panel[fb];
                    const char *panel_full =
                        (fpanel == OV_FOCUS_STREAMS) ? "Streams"
                        : (fpanel == OV_FOCUS_PROCS) ? "Processes"
                        : (fpanel == OV_FOCUS_FPS)   ? "FPS"
                                                     : "Panel";
                    const char *fpat = (fpanel != OV_FOCUS_GRAPH)
                                           ? ov_get_panel_filter_pattern(lay, fpanel)
                                           : ov_get_filter_pattern(lay);
                    int is_act = (fpanel != OV_FOCUS_GRAPH)
                                     ? ov_is_panel_filter_active(lay, fpanel)
                                     : ov_is_filter_active(lay);

                    if (is_act)
                    {
                        if (fpanel == OV_FOCUS_STREAMS)
                        {
                            lay->filter_stream_active = 0;
                        }
                        else if (fpanel == OV_FOCUS_PROCS)
                        {
                            lay->filter_proc_active = 0;
                        }
                        else if (fpanel == OV_FOCUS_FPS)
                        {
                            lay->filter_fps_active = 0;
                        }
                        else
                        {
                            lay->filter_active = 0;
                        }
                        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                       "%s filter paused (press 'f' to resume)",
                                       panel_full);
                        lay->filter_active =
                            (lay->filter_stream_active || lay->filter_proc_active ||
                             lay->filter_fps_active);
                    }
                    else if (fpat[0] != '\0')
                    {
                        if (fpanel == OV_FOCUS_STREAMS)
                        {
                            lay->filter_stream_active = 1;
                        }
                        else if (fpanel == OV_FOCUS_PROCS)
                        {
                            lay->filter_proc_active = 1;
                        }
                        else if (fpanel == OV_FOCUS_FPS)
                        {
                            lay->filter_fps_active = 1;
                        }
                        else
                        {
                            lay->filter_active = 1;
                        }
                        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                       "%s filter resumed: /%s/", panel_full,
                                       fpat);
                        lay->filter_active =
                            (lay->filter_stream_active || lay->filter_proc_active ||
                             lay->filter_fps_active);
                    }
                    else
                    {
                        lay->filter_panel   = fpanel;
                        lay->filter_editing = 1;
                        lay->filter_jump    = 0;
                        lay->filter_cursor  = 0;
                        lay->filter[0]      = '\0';
                        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                       "Type %s regex filter (ENTER=apply, ESC=cancel)",
                                       panel_full);
                    }
                    return 1;
                }
            }
        }

        /* Check for tab selection & help button clicks (row 2) */
        if (mr == lay->r_tabs.row && mc >= 1)
        {
            int tab_widths[OV_VIEW_COUNT];
            int tabs_total_width = 0;
            for (int v = 0; v < OV_VIEW_COUNT; v++)
            {
                tab_widths[v] = (int) strlen(ov_view_label((ov_view_t) v)) + 9;
                tabs_total_width += tab_widths[v];
            }

            int help_width = 11;
            int help_col   = (lay->term_cols >= tabs_total_width + help_width)
                                 ? (lay->term_cols - help_width + 1)
                                 : (tabs_total_width + 1);

            /* Check help button click */
            if (mc >= help_col && mc < help_col + help_width)
            {
                if (lay->show_help)
                {
                    lay->show_help = 0;
                    ov_buf_force_clear();
                }
                else
                {
                    ov_help_open(lay);
                }
                return 1;
            }

            /* Check tab selection clicks */
            int tx = 1;
            for (int v = 0; v < OV_VIEW_COUNT; v++)
            {
                if (mc >= tx && mc < tx + tab_widths[v])
                {
                    lay->view = (ov_view_t) v;
                    if (lay->show_help)
                    {
                        lay->show_help = 0;
                        ov_buf_force_clear();
                    }
                    return 1;
                }
                tx += tab_widths[v];
            }
        }

        /* Preview-bar button clicks (row 3) */
        if (mr == 3 && lay->nb_preview_btns > 0)
        {
            for (int bi = 0; bi < lay->nb_preview_btns; bi++)
            {
                int bc = lay->preview_btns[bi].col;
                int bw = lay->preview_btns[bi].width;
                if (mc >= bc && mc < bc + bw)
                {
                    ov_input__exec_preview_btn(lay->preview_btns[bi].id, lay, m);
                    return 1;
                }
            }
        }

        /* Quick Action Buttons (Hover) */
        if (lay->mouse_hover && lay->hover_idx >= 0)
        {
            if (lay->hover_view == OV_FOCUS_STREAMS && lay->hover_idx < m->nb_streams)
            {
                if (mr >= lay->r_streams.row + 3 &&
                    mr < lay->r_streams.row + lay->r_streams.height)
                {
                    int idx = lay->scroll_stream + (mr - lay->r_streams.row - 3);
                    if (idx == lay->hover_idx)
                    {
                        if (mc >= lay->r_streams.col + lay->r_streams.width - 10 &&
                            mc < lay->r_streams.col + lay->r_streams.width)
                        {
                            ov_ctrl_stream_delete(&m->streams[idx], &lay->cmdlog);
                            return 1;
                        }
                    }
                }
            }
            else if (lay->hover_view == OV_FOCUS_PROCS && lay->hover_idx < m->nb_procs)
            {
                if (mr >= lay->r_procs.row + 3 &&
                    mr < lay->r_procs.row + lay->r_procs.height)
                {
                    int idx = lay->scroll_proc + (mr - lay->r_procs.row - 3);
                    if (idx == lay->hover_idx)
                    {
                        if (mc >= lay->r_procs.col + lay->r_procs.width - 8 &&
                            mc < lay->r_procs.col + lay->r_procs.width)
                        {
                            ov_ctrl_proc_kill(&m->procs[idx], &lay->cmdlog);
                            return 1;
                        }
                    }
                }
            }
            else if (lay->hover_view == OV_FOCUS_FPS && lay->hover_idx < m->nb_fps)
            {
                if (mr >= lay->r_fps.row + 3 &&
                    mr < lay->r_fps.row + lay->r_fps.height)
                {
                    int idx = lay->scroll_fps + (mr - lay->r_fps.row - 3);
                    if (idx == lay->hover_idx)
                    {
                        int right = lay->r_fps.col + lay->r_fps.width;
                        if (mc >= right - 8 && mc < right)
                        {
                            ov_ctrl_fps_conf_toggle(&m->fps[idx], &lay->cmdlog);
                            return 1;
                        }
                        else if (mc >= right - 16 && mc < right - 8)
                        {
                            if (m->fps[idx].run_alive)
                            {
                                ov_ctrl_fps_run_toggle(&m->fps[idx], &lay->cmdlog);
                            }
                            return 1;
                        }
                        else if (mc >= right - 24 && mc < right - 16)
                        {
                            if (!m->fps[idx].run_alive)
                            {
                                ov_ctrl_fps_run_toggle(&m->fps[idx], &lay->cmdlog);
                            }
                            return 1;
                        }
                    }
                }
            }
        }

        if (lay->view == OV_VIEW_DASHBOARD)
        {
            int h_split_row = lay->r_streams.row + lay->r_streams.height;
            if (mr == h_split_row - 1 || mr == h_split_row)
            {
                if (mc < lay->r_graph.col)
                {
                    lay->dash_split_h_dragging = 1;
                    return 1;
                }
            }
            int v_split_col = lay->r_streams.width;
            if (mc == v_split_col || mc == v_split_col + 1)
            {
                if (mr >= lay->r_streams.row &&
                    mr < lay->r_streams.row + lay->r_streams.height + lay->r_fps.height)
                {
                    lay->dash_split_v_dragging = 1;
                    return 1;
                }
            }
        }

        return ov_input_mouse_panel_click(lay, m, mr, mc, is_dbl);
    }

    if (key == OV_KEY_MOUSE_RELEASE)
    {
        lay->fps_split_dragging    = 0;
        lay->dash_split_v_dragging = 0;
        lay->dash_split_h_dragging = 0;
        lay->cmdlog_dragging       = 0;
        return 1;
    }

    if (key == OV_KEY_MOUSE_MOVE)
    {
        return 1;
    }

    if (key == OV_KEY_MOUSE_DRAG)
    {
        return ov_input_mouse_drag(lay, m, ov_mouse_row, ov_mouse_col);
    }

    if (key == OV_KEY_CTRL_SCROLL_UP || key == OV_KEY_CTRL_SCROLL_DOWN ||
        key == OV_KEY_MOUSE_UP || key == OV_KEY_MOUSE_DOWN)
    {
        return ov_input_mouse_wheel(key, lay, m);
    }

    return 0;
}
