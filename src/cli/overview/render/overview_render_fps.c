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
static void ov_fps__render_rows(const OV_LAYOUT  *lay,
                                const OV_MODEL   *m,
                                const OV_RELATED *rel,
                                OV_RECT           r,
                                const int        *fidx,
                                int               filt_n,
                                const char       *active_filter)
{
    int    hrow = r.row + 1;
    int8_t local_depth[OV_MAX_FPS];
    memset(local_depth, 0, sizeof(local_depth));
    {
        int eff_sel = -1;
        if (lay->mouse_hover && lay->panel[OV_FOCUS_FPS].hover_idx >= 0)
        {
            for (int i = 0; i < filt_n; i++)
            {
                if (fidx[i] == lay->panel[OV_FOCUS_FPS].hover_idx)
                {
                    eff_sel = i;
                    break;
                }
            }
        }
        else if (lay->freeze && lay->freeze_focus == OV_FOCUS_FPS &&
                 lay->panel[OV_FOCUS_FPS].freeze_row >= 0 &&
                 lay->panel[OV_FOCUS_FPS].freeze_row < filt_n)
        {
            eff_sel = lay->panel[OV_FOCUS_FPS].freeze_row;
        }
        else if (lay->focus == OV_FOCUS_FPS && lay->panel[OV_FOCUS_FPS].sel >= 0 &&
                 lay->panel[OV_FOCUS_FPS].sel < filt_n)
        {
            eff_sel = lay->panel[OV_FOCUS_FPS].sel;
        }
        if (eff_sel >= 0)
        {
            int root_fi   = fidx[eff_sel];
            int root_node = m->fps[root_fi].node_idx;
            if (root_node >= 0)
            {
                const int8_t *node_depths = ov_node_depths(m, root_node, SG_MODE_FPS);
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
    int start    = lay->panel[OV_FOCUS_FPS].scroll;

    regex_t re;
    int     has_re = 0;
    if (active_filter[0] != '\0')
    {
        if (regcomp(&re, active_filter, REG_EXTENDED | REG_ICASE) == 0)
        {
            has_re = 1;
        }
    }

    if (filt_n == 0)
    {
        char msg[192];
        if (active_filter[0] != '\0')
        {
            snprintf(msg, sizeof(msg), "  No matching FPS modules for '/%s/'", active_filter);
        }
        else
        {
            snprintf(msg, sizeof(msg),
                     "  No FPS \xe2\x80\x94 start a pipeline, or milk-CTRL -d DIR (now %s)",
                     ov_get_shmdir());
        }
        render_empty_rows(r, hrow + 2, max_rows,
                          (active_filter[0] != '\0') ? OV_FG_WARN : OV_FG_DIM, msg);
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
            int           is_sel = (ffi == lay->panel[OV_FOCUS_FPS].sel &&
                                    (lay->focus == OV_FOCUS_FPS || lay->focus == OV_FOCUS_GRAPH));
            int           is_frozen = (lay->freeze && lay->freeze_focus == OV_FOCUS_FPS &&
                                       ffi == lay->panel[OV_FOCUS_FPS].freeze_row);
            ov_focus_t    eff_focus = lay->freeze ? lay->freeze_focus : lay->focus;
            int           has_rel   = (rel != NULL && bget(rel->fps, fi));
            int           is_rel = (!is_sel && !is_frozen && eff_focus != OV_FOCUS_FPS && has_rel);
            int           is_loop_member = 0;
            if ((lay->subtab == OV_SUB_LOOPS || lay->view == OV_VIEW_LOOPS) &&
                lay->sub[OV_SUB_LOOPS].sel >= 0 && lay->sub[OV_SUB_LOOPS].sel < m->nb_loops)
            {
                uint32_t active_mask = (UINT32_C(1) << lay->sub[OV_SUB_LOOPS].sel);
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
            else if (lay->mouse_hover && lay->panel[OV_FOCUS_FPS].hover_idx == fi)
            {
                row_bg = OV_BG_HOVER;
            }
            row_bg = zebra_bg(row_bg, i);

            int printed = 1;
            int avail   = r.width - 2;

            ov_buf_pos(row, r.col + 1);
            ov_theme_bg(row_bg);

            /* Focus ring accent strip */
            int panel_focused = (lay->focus == OV_FOCUS_FPS);
            render_focus_strip(row, r.col + 1, panel_focused, OV_FG_FPS, row_bg);

            OV_ROW_CTX rc = { lay,
                              m,
                              rel,
                              fi,
                              is_sel,
                              is_frozen,
                              is_rel,
                              eff_focus,
                              local_depth[fi],
                              (rel != NULL) ? rel->sel_pid : 0,
                              has_re ? &re : NULL };
            printed       = ov_table_row(&ov_fps_table, &rc, row_bg, avail);

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
                        ov_theme_bg(OV_BG_STALE);
                        ov_theme_fg(OV_FG_WARN);
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
                ov_row_tooltip("FPS: %s | RunPID: %d | ConfPID: %d | Mem: %" PRId64 " KB", f->name,
                               f->runpid, f->confpid, (int64_t) f->mem_rss_kb);

                int btn_w = 24; /* " [Run]  [Stop]  [Conf] " */
                int rem   = r.width - (printed + n5);
                if (lay->ctrl_mode && rem >= btn_w) /* action buttons: CONTROL mode only */
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
    render_scroll_indicators(r, lay->panel[OV_FOCUS_FPS].scroll, max_rows, filt_n, OV_FG_FPS);
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
void ov_render_fps_panel(const OV_LAYOUT *lay, const OV_MODEL *m, const OV_RELATED *rel)
{
    OV_RECT r = lay->r_fps;

    /* Build filtered index array */
    int         fidx[OV_MAX_FPS];
    int         filt_n        = ov_rows(lay, m, OV_FOCUS_FPS, fidx, OV_MAX_FPS);
    const char *active_filter = ov_get_active_filter_for(lay, OV_FOCUS_FPS);

    int  loop_id = (lay->loop_filter_active && lay->sub[OV_SUB_LOOPS].sel >= 0 &&
                    lay->sub[OV_SUB_LOOPS].sel < m->nb_loops)
                       ? m->loops[lay->sub[OV_SUB_LOOPS].sel].loop_id
                       : -1;
    char ftitle[48];
    ov_draw_panel_border_filter(
        r.row, r.col, r.height, r.width, ov_msel_title(lay, m, ftitle, sizeof(ftitle)), OV_FG_FPS,
        lay->focus == OV_FOCUS_FPS, 0, lay->ctrl_blink, loop_id, lay->panel[OV_FOCUS_FPS].filter,
        lay->panel[OV_FOCUS_FPS].filter_on, filt_n, m->nb_fps);

    ov_table_header(&ov_fps_table, lay, r, OV_FG_FPS_HDR);
    ov_fps__render_rows(lay, m, rel, r, fidx, filt_n, active_filter);

    int max_rows = r.height - 4;
    ov_fps__render_footer(lay, m, r, fidx, filt_n, max_rows);

    ov_buf_reset_attr();
}
