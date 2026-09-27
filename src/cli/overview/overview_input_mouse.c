// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_mouse.c
 * @brief Mouse event handling (clicks, drags, scroll wheel, double-click).
 */

#include "overview_input_internal.h"

/**
 * ov_input__handle_mouse - dispatch mouse events (click, drag, double-click, wheel).
 * @key: Mouse event key code (OV_KEY_MOUSE_*).
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: 1 if mouse event was consumed, 0 otherwise.
 */
int ov_input__handle_mouse(int key, OV_LAYOUT *lay, const OV_MODEL *m)
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
                if (lay->mouse_hover)
                {
                    ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN,
                                   "Warning: Hover uses extra CPU on slow connections");
                }
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
                    if (fpanel == OV_FOCUS_GRAPH)
                    {
                        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN,
                                       "Filter available on STREAMS, PROCESSINFO, or FPS panel");
                        return 1;
                    }

                    lay->focus = fpanel;

                    const char *panel_full = (fpanel == OV_FOCUS_STREAMS) ? "STREAMS"
                                             : (fpanel == OV_FOCUS_PROCS) ? "PROCESSINFO"
                                                                          : "FPS";
                    const char *fpat       = ov_get_panel_filter_pattern(lay, fpanel);

                    if (fpat[0] != '\0')
                    {
                        if (fpanel == OV_FOCUS_STREAMS)
                        {
                            lay->filter_stream_active = !lay->filter_stream_active;
                            lay->sel_stream           = 0;
                            lay->scroll_stream        = 0;
                            if (lay->filter_stream_active)
                            {
                                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                               "STREAMS regex filter: ON (/%s/)",
                                               lay->filter_stream);
                            }
                            else
                            {
                                ov_cmdlog_push(
                                    &lay->cmdlog, OV_CMDLOG_INFO,
                                    "STREAMS regex filter: OFF (paused, click/'f' resume, "
                                    "ESC clear)");
                            }
                        }
                        else if (fpanel == OV_FOCUS_PROCS)
                        {
                            lay->filter_proc_active = !lay->filter_proc_active;
                            lay->sel_proc           = 0;
                            lay->scroll_proc        = 0;
                            if (lay->filter_proc_active)
                            {
                                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                               "PROCESSINFO regex filter: ON (/%s/)",
                                               lay->filter_proc);
                            }
                            else
                            {
                                ov_cmdlog_push(
                                    &lay->cmdlog, OV_CMDLOG_INFO,
                                    "PROCESSINFO regex filter: OFF (paused, click/'f' resume, "
                                    "ESC clear)");
                            }
                        }
                        else if (fpanel == OV_FOCUS_FPS)
                        {
                            lay->filter_fps_active = !lay->filter_fps_active;
                            lay->sel_fps           = 0;
                            lay->scroll_fps        = 0;
                            if (lay->filter_fps_active)
                            {
                                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                               "FPS regex filter: ON (/%s/)", lay->filter_fps);
                            }
                            else
                            {
                                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                               "FPS regex filter: OFF (paused, click/'f' resume, "
                                               "ESC clear)");
                            }
                        }
                        lay->filter_active = (lay->filter_stream_active ||
                                              lay->filter_proc_active || lay->filter_fps_active);
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

        /* --- Preview-bar button clicks (row 3) --- */
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

        /* --- Quick Action Buttons (Hover) --- */
        if (lay->mouse_hover && lay->hover_idx >= 0)
        {
            if (lay->hover_view == OV_FOCUS_STREAMS && lay->hover_idx < m->nb_streams)
            {
                if (mr >= lay->r_streams.row + 3 && mr < lay->r_streams.row + lay->r_streams.height)
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
                if (mr >= lay->r_procs.row + 3 && mr < lay->r_procs.row + lay->r_procs.height)
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
                if (mr >= lay->r_fps.row + 3 && mr < lay->r_fps.row + lay->r_fps.height)
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

        /* --- View-aware panel dispatch ---
         * In single-panel views (F3–F6) the panel rects
         * for non-active panels all overlap the full
         * screen, so we must check the view first and
         * route directly to the correct handler.
         * on the Dashboard all four rects are distinct
         * so the cascade works normally.
         */

        if (lay->view == OV_VIEW_DASHBOARD)
        {
            /* Check for dashboard horizontal split drag.
             * The split border is at h_split_row, which
             * coincides with r_graph.row (the tab header
             * row of the bottom-right panel).  Only start
             * a drag when the click is in the LEFT half
             * (under streams/fps); clicks on the RIGHT
             * half fall through so tab labels remain
             * clickable. */
            int h_split_row = lay->r_streams.row + lay->r_streams.height;
            if (mr == h_split_row - 1 || mr == h_split_row)
            {
                if (mc < lay->r_graph.col)
                {
                    lay->dash_split_h_dragging = 1;
                    return 1;
                }
            }
            /* Check for dashboard vertical split drag */
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

        if (lay->view == OV_VIEW_FPS)
        {
            /* F5: Check for split drag */
            if (mc == lay->r_fps_list.width || mc == lay->r_fps_list.width + 1)
            {
                lay->fps_split_dragging = 1;
                return 1;
            }

            /* F5: left = fps list, right = params */
            if (INSIDE(lay->r_fps_params, mr, mc))
            {
                lay->focus           = OV_FOCUS_FPS;
                lay->fps_param_focus = 1;
                int body_row         = mr - lay->r_fps_params.row - 2;
                if (body_row >= 0)
                {
                    int fsel = lay->sel_fps;
                    if (fsel >= 0 && fsel < m->nb_fps)
                    {
                        const OV_FPS   *fps = &m->fps[fsel];
                        fps_tree_item_t items[1024];
                        int nitems = ov_get_fps_tree_items(fps, lay->fps_param_path, items, 1024);
                        if (nitems > 0)
                        {
                            int idx = lay->fps_param_scroll + body_row;
                            if (idx >= nitems)
                            {
                                lay->fps_param_sel = nitems - 1;
                            }
                            else
                            {
                                lay->fps_param_sel = idx;
                            }
                        }
                    }
                }
            }
            else if (INSIDE(lay->r_fps, mr, mc))
            {
                lay->focus           = OV_FOCUS_FPS;
                lay->fps_param_focus = 0;
                int body_row         = mr - lay->r_fps.row - 3;
                if (body_row == -1 || body_row == -2)
                {
                    ov_input__fps_header_click(lay, mc);
                }
                else if (body_row >= 0)
                {
                    int idx = lay->scroll_fps + body_row;
                    if (idx < m->nb_fps)
                    {
                        if (lay->sel_fps != idx)
                        {
                            lay->sel_fps         = idx;
                            lay->sel_name_fps[0] = '\0';
                        }
                        if (is_dbl)
                        {
                            lay->graph_tab_mode = 2;
                        }
                    }
                }
            }
        }
        else if (lay->view == OV_VIEW_STREAMS && INSIDE(lay->r_streams, mr, mc))
        {
            lay->focus   = OV_FOCUS_STREAMS;
            int body_row = mr - lay->r_streams.row - 3;
            if (body_row == -1 || body_row == -2)
            {
                ov_input__streams_header_click(lay, mc);
            }
            else if (body_row >= 0)
            {
                int idx = lay->scroll_stream + body_row;
                if (idx < m->nb_streams)
                {
                    lay->sel_stream         = idx;
                    lay->sel_name_stream[0] = '\0';
                    if (is_dbl)
                    {
                        lay->graph_tab_mode = 2;
                    }
                }
            }
        }
        else if (lay->view == OV_VIEW_PROCS && INSIDE(lay->r_procs, mr, mc))
        {
            lay->focus   = OV_FOCUS_PROCS;
            int body_row = mr - lay->r_procs.row - 3;
            if (body_row == -1 || body_row == -2)
            {
                ov_input__procs_header_click(lay, mc);
            }
            else if (body_row >= 0)
            {
                int idx = lay->scroll_proc + body_row;
                if (idx < m->nb_procs)
                {
                    lay->sel_proc         = idx;
                    lay->sel_name_proc[0] = '\0';
                    if (is_dbl)
                    {
                        lay->graph_tab_mode = 2;
                    }
                }
            }
        }
        else if ((lay->view == OV_VIEW_GRAPH || lay->view == OV_VIEW_LOOPS) &&
                 INSIDE(lay->r_graph, mr, mc))
        {
            lay->focus = OV_FOCUS_GRAPH;

            /* Tab header click */
            if (lay->view == OV_VIEW_GRAPH && mr == lay->r_graph.row)
            {
                const char *dtabs[] = { "CONNECTIONS", "LOOPS", "DETAILS", "RESOURCES" };
                int         ti      = ov_input_hit_panel_tab(mc, lay->r_graph.col, dtabs, 4);
                if (ti >= 0)
                {
                    lay->graph_tab_mode = ti;
                    ov_scan_force_update();
                }
            }
            else
            {
                int body_row = mr - lay->r_graph.row - 2;
                if (body_row >= 0)
                {
                    if (lay->graph_tab_mode == 0)
                    {
                        int start_node = ov_input_get_graph_start_node(lay, m);
                        if (start_node >= 0)
                        {
                            SG_TREE_NODE rnodes[OV_MAX_NODES];
                            int          nb_rnodes =
                                sg_compute_render_tree(m, start_node, lay->lineage_mode, rnodes);

                            int idx = lay->scroll_graph + body_row;
                            if (idx < nb_rnodes)
                            {
                                lay->sel_graph         = idx;
                                const SG_TREE_NODE *rn = &rnodes[idx];

                                int proc_idx = -1;
                                if (rn->reader_name[0] != '\0')
                                {
                                    for (int i = 0; i < m->nb_procs; i++)
                                    {
                                        if (strcmp(m->procs[i].name, rn->reader_name) == 0)
                                        {
                                            proc_idx = i;
                                            break;
                                        }
                                    }
                                }

                                /* Estimate click position to decide stream vs proc */
                                int disp_len = 0;
                                for (int i = 0; rn->tree_prefix[i] != '\0';)
                                {
                                    disp_len++;
                                    i += utf8_char_length((unsigned char) rn->tree_prefix[i]);
                                }
                                if (rn->is_target)
                                {
                                    disp_len += 2;
                                }
                                for (int i = 0; rn->name[i] != '\0';)
                                {
                                    disp_len++;
                                    i += utf8_char_length((unsigned char) rn->name[i]);
                                }

                                int click_on_proc = 0;
                                if (mc - lay->r_graph.col - 1 > disp_len + 1)
                                {
                                    click_on_proc = 1;
                                }

                                if (click_on_proc && proc_idx >= 0)
                                {
                                    lay->focus            = OV_FOCUS_PROCS;
                                    lay->sel_proc         = proc_idx;
                                    lay->sel_name_proc[0] = '\0';
                                }
                                else if (rn->stream_idx >= 0)
                                {
                                    lay->focus              = OV_FOCUS_STREAMS;
                                    lay->sel_stream         = rn->stream_idx;
                                    lay->sel_name_stream[0] = '\0';
                                }

                                if (is_dbl)
                                {
                                    lay->view = OV_VIEW_DASHBOARD;
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (lay->view == OV_VIEW_DASHBOARD)
        {
            /* Dashboard: four non-overlapping rects */
            if (INSIDE(lay->r_streams, mr, mc))
            {
                lay->focus   = OV_FOCUS_STREAMS;
                int body_row = mr - lay->r_streams.row - 3;
                if (body_row == -1 || body_row == -2)
                {
                    ov_input__streams_header_click(lay, mc);
                }
                else if (body_row >= 0)
                {
                    int idx = lay->scroll_stream + body_row;
                    if (idx < m->nb_streams)
                    {
                        lay->sel_stream         = idx;
                        lay->sel_name_stream[0] = '\0';
                        if (is_dbl)
                        {
                            lay->graph_tab_mode = 2;
                        }
                    }
                }
            }
            else if (INSIDE(lay->r_procs, mr, mc))
            {
                lay->focus   = OV_FOCUS_PROCS;
                int body_row = mr - lay->r_procs.row - 3;
                if (body_row == -1 || body_row == -2)
                {
                    ov_input__procs_header_click(lay, mc);
                }
                else if (body_row >= 0)
                {
                    int idx = lay->scroll_proc + body_row;
                    if (idx < m->nb_procs)
                    {
                        lay->sel_proc         = idx;
                        lay->sel_name_proc[0] = '\0';
                        if (is_dbl)
                        {
                            lay->graph_tab_mode = 2;
                        }
                    }
                }
            }
            else if (INSIDE(lay->r_fps, mr, mc))
            {
                lay->focus   = OV_FOCUS_FPS;
                int body_row = mr - lay->r_fps.row - 3;
                if (body_row == -1 || body_row == -2)
                {
                    ov_input__fps_header_click(lay, mc);
                }
                else if (body_row >= 0)
                {
                    int idx = lay->scroll_fps + body_row;
                    if (idx < m->nb_fps)
                    {
                        lay->sel_fps         = idx;
                        lay->sel_name_fps[0] = '\0';
                        if (is_dbl)
                        {
                            lay->graph_tab_mode = 2;
                        }
                    }
                }
            }
            else if (INSIDE(lay->r_graph, mr, mc))
            {
                lay->focus = OV_FOCUS_GRAPH;

                /* Tab header click */
                if (mr == lay->r_graph.row)
                {
                    const char *dtabs[] = { "CONNECTIONS", "LOOPS", "DETAILS", "RESOURCES" };
                    int         ti      = ov_input_hit_panel_tab(mc, lay->r_graph.col, dtabs, 4);
                    if (ti >= 0)
                    {
                        lay->graph_tab_mode = ti;
                        ov_scan_force_update();
                    }
                }
                else
                {
                    int body_row = mr - lay->r_graph.row - 2;
                    if (body_row >= 0)
                    {
                        if (lay->graph_tab_mode == 0)
                        {
                            int start_node = ov_input_get_graph_start_node(lay, m);
                            if (start_node >= 0)
                            {
                                SG_TREE_NODE rnodes[OV_MAX_NODES];
                                int nb_rnodes = sg_compute_render_tree(m, start_node,
                                                                       lay->lineage_mode, rnodes);

                                int idx = lay->scroll_graph + body_row;
                                if (idx < nb_rnodes)
                                {
                                    lay->sel_graph         = idx;
                                    const SG_TREE_NODE *rn = &rnodes[idx];

                                    int proc_idx = -1;
                                    if (rn->reader_name[0] != '\0')
                                    {
                                        for (int i = 0; i < m->nb_procs; i++)
                                        {
                                            if (strcmp(m->procs[i].name, rn->reader_name) == 0)
                                            {
                                                proc_idx = i;
                                                break;
                                            }
                                        }
                                    }

                                    /* Estimate click position to decide stream vs proc */
                                    int disp_len = 0;
                                    for (int i = 0; rn->tree_prefix[i] != '\0';)
                                    {
                                        disp_len++;
                                        i += utf8_char_length((unsigned char) rn->tree_prefix[i]);
                                    }
                                    if (rn->is_target)
                                    {
                                        disp_len += 2;
                                    }
                                    for (int i = 0; rn->name[i] != '\0';)
                                    {
                                        disp_len++;
                                        i += utf8_char_length((unsigned char) rn->name[i]);
                                    }

                                    int click_on_proc = 0;
                                    if (mc - lay->r_graph.col - 1 > disp_len + 1)
                                    {
                                        click_on_proc = 1;
                                    }

                                    if (click_on_proc && proc_idx >= 0)
                                    {
                                        lay->focus            = OV_FOCUS_PROCS;
                                        lay->sel_proc         = proc_idx;
                                        lay->sel_name_proc[0] = '\0';
                                        if (is_dbl)
                                        {
                                            lay->view = OV_VIEW_PROCS;
                                        }
                                    }
                                    else if (rn->stream_idx >= 0)
                                    {
                                        lay->focus              = OV_FOCUS_STREAMS;
                                        lay->sel_stream         = rn->stream_idx;
                                        lay->sel_name_stream[0] = '\0';
                                        if (is_dbl)
                                        {
                                            lay->view = OV_VIEW_STREAMS;
                                        }
                                    }
                                }
                            }
                        }
                        else if (lay->graph_tab_mode == 1)
                        {
                            int max_rows = lay->r_graph.height - 3;
                            int list_rows =
                                (max_rows >= 6) ? ((max_rows > 8) ? (max_rows / 2) : 3) : max_rows;
                            if (body_row >= 0 && body_row < list_rows)
                            {
                                int li = lay->scroll_loop + body_row;
                                if (li >= 0 && li < m->nb_loops)
                                {
                                    lay->sel_loop = li;
                                    if (is_dbl)
                                    {
                                        lay->view = OV_VIEW_LOOPS;
                                    }
                                }
                            }
                        }
                        else if (lay->graph_tab_mode == 2)
                        {
                            ov_focus_t focus = lay->freeze ? lay->freeze_focus : lay->focus;
                            int        fsel  = lay->freeze ? lay->freeze_sel_fps : lay->sel_fps;

                            int active_fps = -1;
                            if (focus == OV_FOCUS_FPS && fsel >= 0 && fsel < m->nb_fps)
                            {
                                active_fps = fsel;
                            }
                            else if (focus != OV_FOCUS_STREAMS && focus != OV_FOCUS_PROCS)
                            {
                                if (fsel >= 0 && fsel < m->nb_fps)
                                {
                                    active_fps = fsel;
                                }
                            }

                            if (active_fps >= 0)
                            {
                                const OV_FPS *f = &m->fps[active_fps];
                                if (f->nb_disp_params > 0)
                                {
                                    int header_rows = 3 + (f->description[0] != '\0' ? 1 : 0);
                                    int param_row   = body_row - header_rows;
                                    if (param_row >= 0)
                                    {
                                        int                  dp     = lay->param_scroll + param_row;
                                        const OV_FPS_PARAMS *params = ov_fps_get_params(f->name);
                                        if (params != NULL && dp >= 0 &&
                                            dp < params->nb_disp_params)
                                        {
                                            lay->param_sel = dp;
                                            if (params->disp_param_type[dp] == FPTYPE_STREAMNAME)
                                            {
                                                int si = ov_find_stream_by_name(
                                                    m, params->disp_param_value[dp]);
                                                if (si >= 0)
                                                {
                                                    lay->focus              = OV_FOCUS_STREAMS;
                                                    lay->sel_stream         = si;
                                                    lay->sel_name_stream[0] = '\0';
                                                    if (is_dbl)
                                                    {
                                                        lay->view = OV_VIEW_STREAMS;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }


        /* Check if clicking on cmdlog border to start dragging */
        {
            int cmdlog_top =
                (lay->cmdlog_rows > 0) ? (lay->term_rows - lay->cmdlog_rows) : lay->term_rows;
            if (mr == cmdlog_top - 1 || mr == cmdlog_top)
            {
                lay->cmdlog_dragging = 1;
            }
        }

        return 1;
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
        int mr = ov_mouse_row;
        int mc = ov_mouse_col;

        /* Global: Command log panel height drag */
        int cmdlog_top =
            (lay->cmdlog_rows > 0) ? (lay->term_rows - lay->cmdlog_rows) : lay->term_rows;
        if (lay->cmdlog_dragging || (mr == cmdlog_top - 1 || mr == cmdlog_top))
        {
            lay->cmdlog_dragging = 1;
            int new_h            = lay->term_rows - 1 - mr;
            if (new_h < 0)
            {
                new_h = 0;
            }
            if (new_h > lay->term_rows / 2)
            {
                new_h = lay->term_rows / 2;
            }
            if (new_h != lay->cmdlog_rows)
            {
                lay->cmdlog_rows = new_h;
                ov_buf_force_clear();
            }
            return 1;
        }

        if (lay->view == OV_VIEW_FPS)
        {
            if (lay->fps_split_dragging ||
                (mc >= lay->r_fps_list.width - 1 && mc <= lay->r_fps_list.width + 2))
            {
                lay->fps_split_dragging = 1;
                float ratio             = (float) mc / lay->term_cols;
                if (ratio < 0.1f)
                {
                    ratio = 0.1f;
                }
                if (ratio > 0.9f)
                {
                    ratio = 0.9f;
                }
                lay->fps_split_ratio = ratio;
                return 1;
            }
        }
        if (lay->view == OV_VIEW_DASHBOARD)
        {
            int h_split_row = lay->r_streams.row + lay->r_streams.height;
            int v_split_col = lay->r_streams.width;

            int handled = 0;
            if (lay->dash_split_h_dragging || (mr >= h_split_row - 1 && mr <= h_split_row + 1))
            {
                lay->dash_split_h_dragging = 1;
                int log_h                  = lay->cmdlog_rows;
                if (log_h < 0)
                {
                    log_h = 0;
                }
                int body_top = 4;
                int body_h   = lay->term_rows - 4 - log_h;
                if (body_h < 4)
                {
                    body_h = 4;
                }
                float ratio = (float) (mr - body_top) / body_h;
                if (ratio < 0.1f)
                {
                    ratio = 0.1f;
                }
                if (ratio > 0.9f)
                {
                    ratio = 0.9f;
                }
                lay->dash_split_h_ratio = ratio;
                handled                 = 1;
            }
            if (lay->dash_split_v_dragging || (mc >= v_split_col - 1 && mc <= v_split_col + 2))
            {
                lay->dash_split_v_dragging = 1;
                float ratio                = (float) mc / lay->term_cols;
                if (ratio < 0.1f)
                {
                    ratio = 0.1f;
                }
                if (ratio > 0.9f)
                {
                    ratio = 0.9f;
                }
                lay->dash_split_v_ratio = ratio;
                handled                 = 1;
            }
            if (handled)
            {
                return 1;
            }
        }
        return 1; /* Ignore other drags for now */
    }

    /* Ctrl+scroll: cycle views (#12) */
    if (key == OV_KEY_CTRL_SCROLL_UP || key == OV_KEY_CTRL_SCROLL_DOWN)
    {
        int v = (int) lay->view;
        if (key == OV_KEY_CTRL_SCROLL_UP)
        {
            v--;
            if (v < 0)
            {
                v = OV_VIEW_COUNT - 1;
            }
        }
        else
        {
            v++;
            if (v >= OV_VIEW_COUNT)
            {
                v = 0;
            }
        }
        lay->view = (ov_view_t) v;
        return 1;
    }

    if (key == OV_KEY_MOUSE_UP || key == OV_KEY_MOUSE_DOWN)
    {
        int mr  = ov_mouse_row;
        int mc  = ov_mouse_col;
        int dir = (key == OV_KEY_MOUSE_UP) ? -3 : 3;

        int *sel    = NULL;
        int *scroll = NULL;
        int  count  = 0;
        int  page_h = 10;

        /* View-aware dispatch (same rationale as click
         * handler — in single-panel views the panel
         * rects overlap).
         */
        if (lay->view == OV_VIEW_FPS)
        {
            if (INSIDE(lay->r_fps_params, mr, mc))
            {
                sel      = &lay->fps_param_sel;
                scroll   = &lay->fps_param_scroll;
                count    = 0;
                int fsel = lay->sel_fps;
                if (fsel >= 0 && fsel < m->nb_fps)
                {
                    const OV_FPS   *fps = &m->fps[fsel];
                    fps_tree_item_t items[1024];
                    count = ov_get_fps_tree_items(fps, lay->fps_param_path, items, 1024);
                }
                page_h = lay->r_fps_params.height - 3;
            }
            else if (INSIDE(lay->r_fps, mr, mc))
            {
                sel    = &lay->sel_fps;
                scroll = &lay->scroll_fps;
                count  = ov_input_get_filtered_count(OV_FOCUS_FPS, lay, m);
                page_h = lay->r_fps.height - 3;
            }
        }
        else if (lay->view == OV_VIEW_STREAMS)
        {
            sel    = &lay->sel_stream;
            scroll = &lay->scroll_stream;
            count  = ov_input_get_filtered_count(OV_FOCUS_STREAMS, lay, m);
            page_h = lay->r_streams.height - 3;
        }
        else if (lay->view == OV_VIEW_PROCS)
        {
            sel    = &lay->sel_proc;
            scroll = &lay->scroll_proc;
            count  = ov_input_get_filtered_count(OV_FOCUS_PROCS, lay, m);
            page_h = lay->r_procs.height - 3;
        }
        else if (lay->view == OV_VIEW_GRAPH)
        {
            if (lay->graph_tab_mode == 0)
            {
                sel            = &lay->sel_graph;
                scroll         = &lay->scroll_graph;
                int start_node = ov_input_get_graph_start_node(lay, m);
                if (start_node >= 0)
                {
                    SG_RENDER_NODE rnodes[OV_MAX_NODES];
                    count = sg_compute_render_nodes(m, start_node, lay->lineage_mode, rnodes);
                }
                else
                {
                    count = m->nb_edges;
                }
                page_h = lay->r_graph.height - 3;
            }
            else if (lay->graph_tab_mode == 1)
            {
                sel    = &lay->sel_loop;
                scroll = &lay->scroll_loop;
                count  = m->nb_loops;
                page_h = (lay->r_graph.height - 3 >= 6)
                             ? ((lay->r_graph.height - 3 > 8) ? ((lay->r_graph.height - 3) / 2) : 3)
                             : (lay->r_graph.height - 3);
            }
            else if (lay->graph_tab_mode == 2)
            {
                page_h = lay->r_graph.height - 3;
                lay->scroll_detail += dir;
                if (lay->scroll_detail < 0)
                {
                    lay->scroll_detail = 0;
                }
                if (lay->scroll_detail > lay->detail_total_lines - page_h)
                {
                    lay->scroll_detail = lay->detail_total_lines - page_h;
                }
                if (lay->scroll_detail < 0)
                {
                    lay->scroll_detail = 0;
                }
                return 1;
            }
            else
            {
                /* RESOURCES panel */
                return 1;
            }
        }
        else if (lay->view == OV_VIEW_LOOPS)
        {
            sel    = &lay->sel_loop;
            scroll = &lay->scroll_loop;
            count  = m->nb_loops;
            page_h = lay->r_graph.height - 3;
        }
        else if (lay->view == OV_VIEW_DASHBOARD)
        {
            /* Dashboard: non-overlapping rects */
            if (INSIDE(lay->r_streams, mr, mc))
            {
                sel    = &lay->sel_stream;
                scroll = &lay->scroll_stream;
                count  = ov_input_get_filtered_count(OV_FOCUS_STREAMS, lay, m);
                page_h = lay->r_streams.height - 3;
            }
            else if (INSIDE(lay->r_procs, mr, mc))
            {
                sel    = &lay->sel_proc;
                scroll = &lay->scroll_proc;
                count  = ov_input_get_filtered_count(OV_FOCUS_PROCS, lay, m);
                page_h = lay->r_procs.height - 3;
            }
            else if (INSIDE(lay->r_fps, mr, mc))
            {
                sel    = &lay->sel_fps;
                scroll = &lay->scroll_fps;
                count  = ov_input_get_filtered_count(OV_FOCUS_FPS, lay, m);
                page_h = lay->r_fps.height - 3;
            }
            else if (INSIDE(lay->r_graph, mr, mc))
            {
                if (lay->graph_tab_mode == 0)
                {
                    sel            = &lay->sel_graph;
                    scroll         = &lay->scroll_graph;
                    int start_node = ov_input_get_graph_start_node(lay, m);
                    if (start_node >= 0)
                    {
                        SG_RENDER_NODE rnodes[OV_MAX_NODES];
                        count = sg_compute_render_nodes(m, start_node, lay->lineage_mode, rnodes);
                    }
                    else
                    {
                        count = m->nb_edges;
                    }
                    page_h = lay->r_graph.height - 3;
                }
                else if (lay->graph_tab_mode == 1)
                {
                    sel    = &lay->sel_loop;
                    scroll = &lay->scroll_loop;
                    count  = m->nb_loops;
                    page_h =
                        (lay->r_graph.height - 3 >= 6)
                            ? ((lay->r_graph.height - 3 > 8) ? ((lay->r_graph.height - 3) / 2) : 3)
                            : (lay->r_graph.height - 3);
                }
                else if (lay->graph_tab_mode == 2)
                {
                    page_h = lay->r_graph.height - 3;
                    lay->scroll_detail += dir;
                    if (lay->scroll_detail < 0)
                    {
                        lay->scroll_detail = 0;
                    }
                    if (lay->scroll_detail > lay->detail_total_lines - page_h)
                    {
                        lay->scroll_detail = lay->detail_total_lines - page_h;
                    }
                    if (lay->scroll_detail < 0)
                    {
                        lay->scroll_detail = 0;
                    }
                    return 1;
                }
                else
                {
                    /* RESOURCES panel */
                    return 1;
                }
            }
        }

        if (sel != NULL && scroll != NULL)
        {
            *scroll += dir;
            if (*scroll < 0)
            {
                *scroll = 0;
            }
            if (page_h > 0 && *scroll > count - page_h)
            {
                *scroll = count - page_h;
                if (*scroll < 0)
                {
                    *scroll = 0;
                }
            }

            *sel += dir;
            if (*sel < *scroll)
            {
                *sel = *scroll;
            }
            if (page_h > 0 && *sel >= *scroll + page_h)
            {
                *sel = *scroll + page_h - 1;
            }
            if (*sel >= count)
            {
                *sel = count - 1;
            }
            if (*sel < 0)
            {
                *sel = 0;
            }

            if (sel == &lay->sel_stream)
            {
                lay->sel_name_stream[0] = '\0';
            }
            else if (sel == &lay->sel_proc)
            {
                lay->sel_name_proc[0] = '\0';
            }
            else if (sel == &lay->sel_fps)
            {
                lay->sel_name_fps[0] = '\0';
            }
        }
        return 1;
    }

    return 0;
}
#undef INSIDE
