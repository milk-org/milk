// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_fps.c
 * @brief FPS panel rendering for milk-CTRL.
 */

#include "overview_render_internal.h"

/**
 * ov_fps__render_rows - render filtered FPS module rows and scrollbar.
 * @lay:           Pointer to layout structure.
 * @m:             Pointer to data model snapshot.
 * @rel:           Pointer to related entities lookup.
 * @r:             Bounding rectangle of FPS panel.
 * @fidx:          Array of visible FPS indices.
 * @filt_n:        Count of matching FPS modules.
 * @active_filter: Active filter string.
 */
static void ov_fps__render_rows(
    const OV_LAYOUT  *lay,
    const OV_MODEL   *m,
    const OV_RELATED *rel,
    OV_RECT           r,
    const int        *fidx,
    int               filt_n,
    const char       *active_filter)
{
    int hrow = r.row + 1;
    int hs   = lay->hscroll_fps;
    int8_t local_depth[OV_MAX_FPS];
    memset(local_depth, 0, sizeof(local_depth));
    {
        int eff_sel = -1;
        if (lay->mouse_hover && lay->hover_global_fps >= 0)
        {
            for (int i = 0; i < filt_n; i++)
            {
                if (fidx[i] == lay->hover_global_fps)
                {
                    eff_sel = i;
                    break;
                }
            }
        }
        else if (lay->freeze && lay->freeze_focus == OV_FOCUS_FPS && lay->freeze_sel_fps >= 0 &&
                 lay->freeze_sel_fps < filt_n)
        {
            eff_sel = lay->freeze_sel_fps;
        }
        else if (lay->focus == OV_FOCUS_FPS && lay->sel_fps >= 0 && lay->sel_fps < filt_n)
        {
            eff_sel = lay->sel_fps;
        }
        if (eff_sel >= 0)
        {
            int root_fi   = fidx[eff_sel];
            int root_node = m->fps[root_fi].node_idx;
            if (root_node >= 0)
            {
                int8_t node_depths[OV_MAX_NODES];
                sg_compute_node_depths(m, root_node, SG_MODE_FPS, node_depths);
                for (int fi = 0; fi < m->nb_fps; fi++)
                {
                    int n = m->fps[fi].node_idx;
                    if (n >= 0 && node_depths[n] != 127)
                    {
                        local_depth[fi] = node_depths[n];
                    }
                }
            }
        }
    }

    int max_rows = r.height - 4;
    int start    = lay->scroll_fps;

    regex_t re;
    int     has_re = 0;
    if (active_filter[0] != '\0')
    {
        if (regcomp(&re, active_filter, REG_EXTENDED | REG_ICASE) == 0)
        {
            has_re = 1;
        }
    }

    if (filt_n == 0 && active_filter[0] != '\0')
    {
        int row = hrow + 2;
        ov_buf_pos(row, r.col + 1);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        char msg[128];
        snprintf(msg, sizeof(msg), "  No matching FPS modules for '/%s/'", active_filter);
        ov_buf_printf("%s", msg);
        render_pad_spaces((int) strlen(msg), r.width);
        for (int i = 1; i < max_rows; i++)
        {
            clear_row(hrow + 2 + i, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
        if (has_re)
        {
            regfree(&re);
        }
        render_scroll_indicators(r, 0, max_rows, 0, OV_FG_FPS);
        return;
    }

    for (int i = 0; i < max_rows; i++)
    {
        int row = hrow + 2 + i;
        int ffi = start + i;
        if (ffi < filt_n)
        {
            int           fi     = fidx[ffi];
            const OV_FPS *f      = &m->fps[fi];
            int           is_sel = (ffi == lay->sel_fps &&
                                    (lay->focus == OV_FOCUS_FPS || lay->focus == OV_FOCUS_GRAPH));
            int           is_frozen =
                (lay->freeze && lay->freeze_focus == OV_FOCUS_FPS && ffi == lay->freeze_sel_fps);
            ov_focus_t eff_focus = lay->freeze ? lay->freeze_focus : lay->focus;
            int        has_rel   = (rel != NULL && bget(rel->fps, fi));
            int        is_rel    = (!is_sel && !is_frozen && eff_focus != OV_FOCUS_FPS && has_rel);
            int        is_loop_member = 0;
            if ((lay->graph_tab_mode == 1 || lay->view == OV_VIEW_LOOPS) && lay->sel_loop >= 0 &&
                lay->sel_loop < m->nb_loops)
            {
                uint32_t active_mask = (UINT32_C(1) << lay->sel_loop);
                if (f->loop_mask & active_mask)
                {
                    is_loop_member = 1;
                }
            }

            ov_rgb_t row_bg = OV_BG_PANEL;

            if (is_sel)
            {
                row_bg = OV_BG_SELECTED;
            }
            else if (is_frozen)
            {
                row_bg = OV_BG_FROZEN;
            }
            else if (is_loop_member)
            {
                row_bg = (f->nb_loops > 1) ? OV_BG_LOOP_SHARED : OV_BG_LOOP;
            }
            else if (is_rel)
            {
                row_bg = OV_BG_RELATED;
            }
            else if (f->is_new > 0)
            {
                row_bg = OV_BG_NEW_ITEM;
            }
            else if (lay->mouse_hover && lay->hover_global_fps == fi)
            {
                row_bg = OV_BG_HOVER;
            }
            row_bg = zebra_bg(row_bg, i);

            int hs_rem  = hs;
            int printed = 1;
            int avail   = r.width - 2;

            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(row_bg);

            /* Focus ring accent strip */
            int panel_focused = (lay->focus == OV_FOCUS_FPS);
            render_focus_strip(row, r.col + 1, panel_focused, OV_FG_FPS, row_bg);

            /* Multi-select marker */
            int fsi = (int) (f - m->fps);
            if (fsi >= 0 && fsi < 200 && lay->multi_sel_fps[fsi])
            {
                ov_theme_fg(OV_FG_ACTIVE);
                ov_buf_printf("\xe2\x97\x86"); /* ◆ */
                printed += 1;
            }

            /* Ancestry column */
            int8_t sdepth      = local_depth[fi];
            char   anc_str[64] = "";
            if (sdepth != 0 && !is_sel && !is_frozen)
            {
                int abs_d = sdepth < 0 ? -sdepth : sdepth;
                if (abs_d > 99)
                {
                    abs_d = 99;
                }
                if (sdepth < 0)
                {
                    snprintf(anc_str, sizeof(anc_str),
                             abs_d < 10 ? "\xe2\x97\x80%d  " : "\xe2\x97\x80%d ", abs_d);
                }
                else
                {
                    snprintf(anc_str, sizeof(anc_str),
                             abs_d < 10 ? "%d\xe2\x96\xb6  " : "%d\xe2\x96\xb6 ", abs_d);
                }
            }
            else if (is_sel || is_frozen)
            {
                snprintf(anc_str, sizeof(anc_str), "\xe2\x97\x8f   ");
            }
            else if (eff_focus == OV_FOCUS_STREAMS && is_rel && rel != NULL)
            {
                int is_written = bget(rel->fps_writes, fi);
                snprintf(anc_str, sizeof(anc_str),
                         is_written ? "\xe2\x96\xb6   " : "\xe2\x97\x80   ");
            }
            else if (f->nb_loops > 1)
            {
                snprintf(anc_str, sizeof(anc_str), "\xe2\xae\x82   "); /* ⮂ */
            }
            else if (f->nb_loops == 1)
            {
                snprintf(anc_str, sizeof(anc_str), "\xe2\x86\xba   "); /* ↺ */
            }
            else
            {
                snprintf(anc_str, sizeof(anc_str), "    ");
            }

            ov_rgb_t anc_color = (f->nb_loops > 1) ? OV_FG_LOOP_SHARED
                                                   : ((f->nb_loops == 1) ? OV_FG_LOOP : OV_FG_WARN);
            ov_render_cell(0, 0, anc_color, row_bg, anc_str, &hs_rem, &printed, avail,
                           lay->highlight_col_fps, lay->col_collapsed_fps);

#define FPS_FIELD_WITH_COL(vcol_idx, logical_idx, color, bg_color, fmt, ...)                      \
    do                                                                                            \
    {                                                                                             \
        char _fb[128];                                                                            \
        int  _fl = snprintf(_fb, sizeof(_fb), fmt, ##__VA_ARGS__);                                \
        ov_render_cell(logical_idx, vcol_idx, (color), (bg_color), _fb, &hs_rem, &printed, avail, \
                       lay->highlight_col_fps, lay->col_collapsed_fps);                           \
    } while (0)

#define FPS_FIELD(color, fmt, ...)                                                   \
    do                                                                               \
    {                                                                                \
        int logical_idx = ov_get_logical_col_fps(vcol, lay->compact_mode);           \
        FPS_FIELD_WITH_COL(vcol, logical_idx, (color), cell_bg, fmt, ##__VA_ARGS__); \
        vcol++;                                                                      \
    } while (0)

#define FPS_PID_FIELD(pid_val, fmt, ...) \
    do \
    { \
        pid_t    _pval    = (pid_t) (pid_val); \
        int      _match   = (_spid > 0 && _pval == _spid); \
        int      _idx     = ov_find_proc_by_pid(m, _pval); \
        int      _crashed = (_idx >= 0 && \
                             m->procs[_idx].loopstat == PROCESSINFO_LOOPSTAT_CRASHED); \
        ov_rgb_t prev_bg  = cell_bg; \
        if (_match) \
        { \
            if (_crashed) \
            { \
                cell_bg = OV_FG_ERROR; \
            } \
            else \
            { \
                cell_bg = OV_BG_PID_MATCH; \
            } \
            ov_buf_bold(); \
        } \
        ov_rgb_t _fg; \
        if (_crashed) \
        { \
            _fg = _match ? (ov_rgb_t) { 255, 255, 255 } : OV_FG_ERROR; \
        } \
        else if (_match) \
        { \
            _fg = (ov_rgb_t) { 0, 0, 0 }; \
        } \
        else \
        { \
            _fg = ov_pid_color(_pval); \
        } \
        FPS_FIELD(_fg, fmt, ##__VA_ARGS__); \
        if (_match) \
        { \
            ov_buf_reset_attr(); \
            cell_bg = prev_bg; \
        } \
    } while (0)

            pid_t    _spid   = (rel != NULL) ? rel->sel_pid : 0;
            int      vcol    = 1;
            ov_rgb_t cell_bg = row_bg;

            /* FPS Name with regex match highlighting */
            {
                char       name_cell[128];
                regmatch_t pm[1];
                if (has_re && regexec(&re, f->name, 1, pm, 0) == 0)
                {
                    int b_len = pm[0].rm_so;
                    if (b_len > 18)
                    {
                        b_len = 18;
                    }
                    int m_len = pm[0].rm_eo - pm[0].rm_so;
                    if (b_len + m_len > 18)
                    {
                        m_len = 18 - b_len;
                    }
                    int tail_len = 18 - (b_len + m_len);
                    if (tail_len < 0)
                    {
                        tail_len = 0;
                    }
                    snprintf(name_cell, sizeof(name_cell), "%.*s\x01%.*s\x02%.*s ", b_len, f->name,
                             m_len, f->name + b_len, tail_len, f->name + b_len + m_len);
                }
                else
                {
                    snprintf(name_cell, sizeof(name_cell), "%-18.18s ", f->name);
                }
                FPS_FIELD(OV_FG_FPS, "%s", name_cell);
            }

            char tmx_str[4] = { (f->tmux_flags & OV_TMUX_CTRL) ? 'c' : '-',
                                (f->tmux_flags & OV_TMUX_CONF) ? 'C' : '-',
                                (f->tmux_flags & OV_TMUX_RUN) ? 'r' : '-', '\0' };
            FPS_FIELD(OV_FG_DIM, "%3s ", tmx_str);

            if (f->confpid > 0)
            {
                FPS_PID_FIELD(f->confpid, "%7d ", (int) f->confpid);
            }
            else
            {
                FPS_FIELD(OV_FG_DIM, "%7s ", "-");
            }
            if (f->runpid > 0)
            {
                FPS_PID_FIELD(f->runpid, "%7d ", (int) f->runpid);
            }
            else
            {
                FPS_FIELD(OV_FG_DIM, "%7s ", "-");
            }
            FPS_FIELD(OV_FG_TEXT, "%3d ", f->nb_stream_params);

            /* MEM */
            char memstr[16];
            format_mem_kb(memstr, sizeof(memstr), f->mem_rss_kb);
            FPS_FIELD(OV_FG_TEXT, "%5s ", memstr);

            /* Detailed columns (hidden in compact) */
            if (!lay->compact_mode)
            {
                if (lay->view == OV_VIEW_FPS)
                {
                    FPS_FIELD(OV_FG_DIM, "%-30.30s ", f->description);
                }
                else
                {
                    FPS_FIELD(OV_FG_DIM, "%-20.20s ", f->description);
                }
                /* FPS Hz sparkline */
                if (f->hz_hist_idx > 0)
                {
                    render_sparkline(f->hz_hist, f->hz_hist_idx, OV_SPARKLINE_LEN, 8, OV_FG_FPS);
                    printed += 8;
                    ov_buf_printf(" ");
                    printed += 1;
                }
            }

#undef FPS_PID_FIELD
#undef FPS_FIELD_WITH_COL
#undef FPS_FIELD

            /* When cross-highlighted by a stream selection, iterate all
             * stream params of this FPS that match the selected stream */
            int n5 = 0;
            if (has_rel && eff_focus == OV_FOCUS_STREAMS)
            {
                uint32_t mask = rel->fps_param_mask[fi];
                for (int sp = 0; mask != 0 && sp < f->nb_stream_params; sp++, mask >>= 1)
                {
                    if (!(mask & 1))
                    {
                        continue;
                    }

                    const char *kname = f->stream_param_name[sp];

                    if (strcmp(kname, "procinfo.triggersname") == 0)
                    {
                        ov_buf_bg(120, 80, 10);
                        ov_buf_fg(255, 210, 80);
                        ov_buf_bold();
                        int w = snprintf(NULL, 0, " [TRIG]");
                        ov_buf_printf(" [TRIG]");
                        ov_buf_reset_attr();
                        ov_theme_bg(row_bg);
                        n5 += w;
                    }
                    else
                    {
                        ov_theme_fg(OV_FG_CONN);
                        int w = snprintf(NULL, 0, " :%s", kname);
                        ov_buf_printf(" :%s", kname);
                        n5 += w;
                    }
                }
            }

            if (lay->mouse_hover && lay->hover_view == OV_FOCUS_FPS && lay->hover_idx == fi)
            {
                snprintf((char *) lay->hover_tooltip, sizeof(lay->hover_tooltip),
                         "FPS: %s | RunPID: %d | ConfPID: %d | Mem: %" PRId64 " KB", f->name,
                         f->runpid, f->confpid, (int64_t) f->mem_rss_kb);

                int btn_w = 24; /* " [Run]  [Stop]  [Conf] " */
                int rem   = r.width - (printed + n5);
                if (rem >= btn_w)
                {
                    render_pad_spaces(printed + n5, r.width - btn_w);

                    ov_theme_bg(OV_FG_ACTIVE);
                    ov_theme_fg(OV_FG_TEXT);
                    ov_buf_printf(" [Run] ");

                    ov_theme_bg(OV_FG_ERROR);
                    ov_theme_fg(OV_FG_TEXT);
                    ov_buf_printf(" [Stop] ");

                    ov_theme_bg(OV_BG_HEADER);
                    ov_theme_fg(OV_FG_TEXT);
                    ov_buf_printf(" [Conf] ");

                    printed = r.width;
                    n5      = 0;
                }
            }

            render_pad_spaces(printed + n5, r.width);
            ov_buf_reset_attr();
        }
        else
        {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
    }
    render_scroll_indicators(r, lay->scroll_fps, max_rows, filt_n, OV_FG_FPS);
    if (has_re)
    {
        regfree(&re);
    }
}

/**
 * ov_render_fps_panel - render the entire FPS panel (border, header, rows, footer).
 * @lay: Pointer to layout structure.
 * @m:   Pointer to current data model snapshot.
 * @rel: Pointer to relationship lookup tables.
 */
void ov_render_fps_panel(
    const OV_LAYOUT  *lay,
    const OV_MODEL   *m,
    const OV_RELATED *rel)
{
    OV_RECT r = lay->r_fps;

    /* Build filtered index array */
    int         fidx[OV_MAX_FPS];
    int         filt_n        = ov_filter_fps(lay, m, rel, fidx, OV_MAX_FPS);
    const char *active_filter = ov_get_active_filter_for(lay, OV_FOCUS_FPS);

    int loop_id = (lay->loop_filter_active && lay->sel_loop >= 0 && lay->sel_loop < m->nb_loops)
                      ? m->loops[lay->sel_loop].loop_id
                      : -1;
    ov_draw_panel_border_filter(r.row, r.col, r.height, r.width, "FPS", OV_FG_FPS,
                                lay->focus == OV_FOCUS_FPS, 0, lay->ctrl_blink, loop_id,
                                lay->filter_fps, lay->filter_fps_active, filt_n, m->nb_fps);

    ov_fps__render_header(lay, r);
    ov_fps__render_rows(lay, m, rel, r, fidx, filt_n, active_filter);

    int max_rows = r.height - 4;
    ov_fps__render_footer(lay, m, r, fidx, filt_n, max_rows);

    ov_buf_reset_attr();
}
