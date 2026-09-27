// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_procs.c
 * @brief PROCS panel rendering for milk-CTRL.
 */

#include "overview_render_internal.h"

/**
 * ov_procs__filter - Filter processes based on current layout and relations
 * @lay:      Layout configuration
 * @m:        Data model containing processes
 * @rel:      Related items bitmasks
 * @filt_idx: Output array of filtered process indices
 * @has_re:   Output flag set to 1 if regex compilation succeeded
 * @re:       Output compiled regex structure
 *
 * Return: Number of filtered processes in @filt_idx.
 */
static int ov_procs__filter(const OV_LAYOUT  *lay,
                            const OV_MODEL   *m,
                            const OV_RELATED *rel,
                            int              *filt_idx,
                            int              *has_re,
                            regex_t          *re)
{
    int         filt_n        = ov_filter_procs(lay, m, rel, filt_idx, OV_MAX_PROCS);
    const char *active_filter = ov_get_active_filter_for(lay, OV_FOCUS_PROCS);

    *has_re = 0;
    if (active_filter[0] != '\0')
    {
        if (regcomp(re, active_filter, REG_EXTENDED | REG_ICASE) == 0)
        {
            *has_re = 1;
        }
    }
    return filt_n;
}

/**
 * ov_procs__render_rows - Render process rows in the overview
 * @lay:      Layout configuration
 * @m:        Data model containing processes
 * @rel:      Related items bitmasks
 * @hrow:     Screen row for headers
 * @r:        Bounding rectangle of panel
 * @filt_idx: Array of filtered process indices
 * @filt_n:   Count of matching processes
 * @has_re:   Flag indicating whether regex is compiled
 * @re:       Compiled regex pointer or NULL
 */
static void ov_procs__render_rows(const OV_LAYOUT  *lay,
                                  const OV_MODEL   *m,
                                  const OV_RELATED *rel,
                                  int               hrow,
                                  OV_RECT           r,
                                  const int        *filt_idx,
                                  int               filt_n,
                                  int               has_re,
                                  const regex_t    *re)
{
    int8_t local_depth[OV_MAX_PROCS];
    memset(local_depth, 0, sizeof(local_depth));
    {
        int eff_sel = -1;
        if (lay->freeze && lay->freeze_focus == OV_FOCUS_PROCS && lay->freeze_sel_proc >= 0 &&
            lay->freeze_sel_proc < filt_n)
        {
            eff_sel = lay->freeze_sel_proc;
        }
        else if (lay->focus == OV_FOCUS_PROCS && lay->sel_proc >= 0 && lay->sel_proc < filt_n)
        {
            eff_sel = lay->sel_proc;
        }
        if (eff_sel >= 0)
        {
            int root_pi   = filt_idx[eff_sel];
            int root_node = m->procs[root_pi].node_idx;
            if (root_node >= 0)
            {
                int8_t node_depths[OV_MAX_NODES];
                sg_compute_node_depths(m, root_node, SG_MODE_FULL, node_depths);
                for (int pi = 0; pi < m->nb_procs; pi++)
                {
                    int n = m->procs[pi].node_idx;
                    if (n >= 0 && node_depths[n] != 127)
                    {
                        local_depth[pi] = node_depths[n];
                    }
                }
            }
        }
    }

    int max_rows = r.height - 4;
    int start    = lay->scroll_proc;

    const char *active_filter = ov_get_active_filter_for(lay, OV_FOCUS_PROCS);
    if (filt_n == 0 && active_filter[0] != '\0')
    {
        int row = hrow + 2;
        ov_buf_pos(row, r.col + 1);
        ov_theme_bg(OV_BG_PANEL);
        ov_theme_fg(OV_FG_WARN);
        char msg[128];
        snprintf(msg, sizeof(msg), "  No matching processes for '/%s/'", active_filter);
        ov_buf_printf("%s", msg);
        render_pad_spaces((int) strlen(msg), r.width);
        for (int i = 1; i < max_rows; i++)
        {
            clear_row(hrow + 2 + i, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
        render_scroll_indicators(r, 0, max_rows, 0, OV_FG_PROC);
        return;
    }

    for (int i = 0; i < max_rows; i++)
    {
        int row = hrow + 2 + i;
        int fi  = start + i;
        if (fi < filt_n)
        {
            int    pi     = filt_idx[fi];
            int8_t sdepth = local_depth[pi];
            ov_procs_render_single_row(lay, m, rel, row, i, fi, pi, sdepth, has_re, re, r);
        }
        else
        {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
        }
    }
    render_scroll_indicators(r, lay->scroll_proc, max_rows, filt_n, OV_FG_PROC);
    ov_buf_reset_attr();
}

/**
 * ov_render_procs_panel - Render the processes panel in the overview
 * @lay: Layout configuration
 * @m:   Data model containing processes
 * @rel: Related items bitmasks
 */
void ov_render_procs_panel(const OV_LAYOUT *lay, const OV_MODEL *m, const OV_RELATED *rel)
{
    OV_RECT r = lay->r_procs;

    int     filt_idx[OV_MAX_PROCS];
    int     has_re;
    regex_t re;
    int     filt_n = ov_procs__filter(lay, m, rel, filt_idx, &has_re, &re);

    int loop_id = (lay->loop_filter_active && lay->sel_loop >= 0 && lay->sel_loop < m->nb_loops)
                      ? m->loops[lay->sel_loop].loop_id
                      : -1;
    ov_draw_panel_border_filter(r.row, r.col, r.height, r.width, "PROCESSINFO", OV_FG_PROC,
                                lay->focus == OV_FOCUS_PROCS, 0, lay->ctrl_blink, loop_id,
                                lay->filter_proc, lay->filter_proc_active, filt_n, m->nb_procs);

    int hrow = r.row + 1;

    ov_procs__render_header(lay, r);
    ov_procs__render_rows(lay, m, rel, hrow, r, filt_idx, filt_n, has_re, &re);

    int max_rows = r.height - 4;
    ov_procs__render_footer(lay, m, r, filt_idx, filt_n, max_rows);
    if (has_re)
    {
        regfree(&re);
    }
}
