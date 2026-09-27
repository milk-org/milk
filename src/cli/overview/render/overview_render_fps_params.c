// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_fps_params.c
 * @brief FPS parameter tree panel for milk-CTRL F5 view.
 */

#include <string.h>
#include <stdint.h>

#include "overview_render_internal.h"
#include "overview_render_fps_params.h"
#include "fps_types.h"

/**
 * render_param_breadcrumb - draw the FPS name / param count header.
 * @lay: Layout state.
 * @fps: FPS data entry.
 * @r:   Panel rectangle (r_fps_params).
 */
static void render_param_breadcrumb(const OV_LAYOUT *lay, const OV_FPS *fps, OV_RECT r)
{
    int      focused   = (lay->fps_param_focus == 1);
    ov_rgb_t border_fg = focused ? OV_FG_FPS : OV_FG_DIM;

    /* Draw border and title in one call */
    char title[256];
    if (lay->fps_param_path[0] != '\0')
    {
        snprintf(title, sizeof(title), "PARAMS  %s / %s", fps->name, lay->fps_param_path);
    }
    else
    {
        snprintf(title, sizeof(title), "PARAMS  %s", fps->name);
    }
    ov_draw_panel_border(r.row, r.col, r.height, r.width, title, border_fg, focused, 0);

    /* Header row: col labels */
    int hrow = r.row + 1;
    ov_buf_pos(hrow, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_DIM);

    int kw = r.width - 18; /* keyword column width */
    if (kw < 6)
    {
        kw = 6;
    }
    ov_buf_printf(" %-*s %4s  %-*s", kw, "PARAMETER", "TYPE", r.width - kw - 12, "VALUE");
    ov_buf_reset_attr();
}

/**
 * ov_render_fps_params_panel - draw parameter tree panel.
 * @lay: Layout state (fps_param_focus/sel/scroll used).
 * @m:   Data model.
 */
void ov_render_fps_params_panel(OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_RECT r = lay->r_fps_params;

    /* Guard: need a valid FPS selection with params */
    int fsel = lay->sel_fps;
    if (fsel < 0 || fsel >= m->nb_fps)
    {
        ov_draw_panel_border(r.row, r.col, r.height, r.width, "PARAMS", OV_FG_DIM, 0, 0);
        return;
    }

    const OV_FPS        *fps    = &m->fps[fsel];
    const OV_FPS_PARAMS *params = ov_fps_get_params(fps->name);

    fps_tree_item_t items[1024];
    int             nitems = ov_get_fps_tree_items(fps, lay->fps_param_path, items, 1024);

    if (lay->fps_param_focus == 1 && nitems > 0)
    {
        if (lay->fps_param_sel < 0)
        {
            lay->fps_param_sel = 0;
        }
        else if (lay->fps_param_sel >= nitems)
        {
            lay->fps_param_sel = nitems - 1;
        }
    }

    render_param_breadcrumb(lay, fps, r);

    if (nitems <= 0)
    {
        if (lay->fps_param_path[0] != '\0')
        {
            ov_buf_pos(r.row + 2, r.col + 2);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("(Empty directory)");
        }
        else
        {
            ov_buf_pos(r.row + 2, r.col + 2);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("(No params)");
        }
        return;
    }

    /* Clamp scroll */
    int max_rows = r.height - 3; /* header row + borders */
    if (max_rows < 1)
    {
        max_rows = 1;
    }

    int scroll = lay->fps_param_scroll;
    int psel   = lay->fps_param_sel;

    /* Keep cursor visible */
    if (psel < scroll)
    {
        scroll = psel;
    }
    if (psel >= scroll + max_rows)
    {
        scroll = psel - max_rows + 1;
    }
    if (scroll < 0)
    {
        scroll = 0;
    }
    if (scroll > nitems - max_rows)
    {
        scroll = nitems - max_rows;
    }
    if (scroll < 0)
    {
        scroll = 0;
    }
    lay->fps_param_scroll = scroll;

    /* Keyword column width */
    int kw = r.width / 2 - 2;
    if (kw > 35)
    {
        kw = 35;
    }
    if (kw < 12)
    {
        kw = 12;
    }

    for (int i = 0; i < max_rows; i++)
    {
        int row      = r.row + 2 + i;
        int list_idx = scroll + i;

        if (list_idx >= nitems)
        {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
            continue;
        }

        int is_sel = (lay->fps_param_focus == 1 && list_idx == psel);

        ov_rgb_t row_bg = is_sel ? OV_BG_SELECTED : OV_BG_PANEL;

        ov_buf_pos(row, r.col + 1);
        ov_theme_bg(row_bg);

        fps_tree_item_t *item = &items[list_idx];

        if (item->is_dir)
        {
            /* Directory row */
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("  ");

            ov_theme_fg(is_sel ? OV_FG_TEXT : (ov_rgb_t) { 220, 200, 100 });
            ov_buf_printf("%-*.*s/", kw - 1, kw - 1, item->name);

            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("%4s  ", "DIR ");

            int vw = r.width - kw - 12;
            if (vw < 4)
            {
                vw = 4;
            }
            ov_buf_printf("%-*s", vw, "");
        }
        else
        {
            /* Leaf parameter row */
            int         pi  = item->param_idx;
            uint64_t    fl  = (params != NULL && pi >= 0 && pi < params->nb_disp_params)
                                  ? params->disp_param_flags[pi]
                                  : 0;
            uint32_t    pt  = (params != NULL && pi >= 0 && pi < params->nb_disp_params)
                                  ? params->disp_param_type[pi]
                                  : 0;
            const char *val = (params != NULL && pi >= 0 && pi < params->nb_disp_params)
                                  ? params->disp_param_value[pi]
                                  : "";

            /* Write-status badge */
            int ws = (fl & FPFLAG_WRITESTATUS) != 0;
            int wc = (fl & FPFLAG_WRITECONF) != 0;

            if (ws)
            {
                ov_theme_fg(is_sel ? OV_FG_ACTIVE : (ov_rgb_t) { 80, 200, 80 });
                ov_buf_printf("W ");
            }
            else if (wc)
            {
                ov_theme_fg(is_sel ? OV_FG_TEXT : (ov_rgb_t) { 220, 180, 60 });
                ov_buf_printf("C ");
            }
            else
            {
                ov_theme_fg(is_sel ? OV_FG_DIM : (ov_rgb_t) { 160, 60, 60 });
                ov_buf_printf("NW");
            }
            ov_buf_printf(" ");

            /* Parameter keyword */
            ov_theme_fg(is_sel ? OV_FG_TEXT : OV_FG_FPS);
            ov_buf_printf("%-*.*s ", kw, kw, item->name);

            /* Type badge */
            ov_rgb_t tc = fps_param_type_color(pt);
            if (is_sel)
            {
                tc = OV_FG_TEXT;
            }
            ov_theme_fg(tc);
            ov_buf_printf("%4s  ", fps_param_type_badge(pt));

            /* Value */
            int vw = r.width - kw - 14;
            if (vw < 4)
            {
                vw = 4;
            }

            if (pt == FPTYPE_ONOFF)
            {
                int is_on = (val[0] == 'O' && val[1] == 'N');
                if (is_on)
                {
                    ov_theme_fg(is_sel ? OV_FG_ACTIVE : (ov_rgb_t) { 60, 220, 60 });
                    ov_buf_printf("%-*s", vw, "ON");
                }
                else
                {
                    ov_theme_fg(is_sel ? OV_FG_DIM : (ov_rgb_t) { 160, 60, 60 });
                    ov_buf_printf("%-*s", vw, "OFF");
                }
            }
            else
            {
                ov_theme_fg(is_sel ? OV_FG_TEXT : OV_FG_DIM);
                ov_buf_printf("%-*.*s", vw, vw, val);
            }
        }

        ov_buf_reset_attr();
    }

    render_scroll_indicators(r, scroll, max_rows, nitems, OV_FG_FPS);

    /* Footer: item count */
    char fbuf[48];
    snprintf(fbuf, sizeof(fbuf), " %d/%d items ", psel + 1, nitems);
    int flen = (int) strlen(fbuf);
    int bcol = r.col + r.width - flen - 2;
    if (bcol > r.col + 1)
    {
        ov_buf_pos(r.row + r.height - 1, bcol);
        ov_theme_fg(OV_FG_DIM);
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_printf("%s", fbuf);
    }

    ov_buf_reset_attr();
}
