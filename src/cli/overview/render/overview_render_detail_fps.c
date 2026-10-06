// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_detail_fps.c
 * @brief   FPS module and parameter inspector pane for milk-CTRL.
 */

#include "overview_render_detail_internal.h"
#include <stdio.h>
#include <string.h>

/**
 * ov_fps__render_detail_fps - render comprehensive inspector pane for selected FPS module.
 * @lay:      Pointer to layout structure.
 * @m:        Pointer to data model snapshot.
 * @fsel:     Selected FPS index in model.
 * @r:        Bounding rectangle of detail panel.
 * @max_rows: Maximum visible rows in panel.
 * @row:      Base row coordinate on terminal.
 *
 * Return: 1 on success, 0 otherwise.
 */
int ov_fps__render_detail_fps(OV_LAYOUT      *lay,
                              const OV_MODEL *m,
                              int             fsel,
                              OV_RECT         r,
                              int             max_rows,
                              int             row)
{
    const OV_FPS *f = &m->fps[fsel];

    const char *tabs[] = { "CONNECTIONS", "LOOPS", "DETAILS", "RESOURCES" };
    ov_draw_panel_tabs(r.row, r.col, r.height, r.width, tabs, 4, lay->graph_tab_mode, OV_FG_FPS,
                       lay->focus == OV_FOCUS_GRAPH);

    int ri       = 0;
    int line_idx = 0;

    /* Name */
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_TITLE);
        H_ov_buf_bold();
        int n = snprintf(NULL, 0, " %s", f->name);
        H_ov_buf_printf(" %s", f->name);
        H_ov_buf_reset_attr();
        H_ov_theme_bg(OV_BG_PANEL);
        H_render_pad_spaces(n, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }

    /* Description */
    if (f->description[0] != '\0')
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_TEXT);
        int n = snprintf(NULL, 0, " %s", f->description);
        H_ov_buf_printf(" %s", f->description);
        H_render_pad_spaces(n, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }

    /* Conf / Run status */
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_DIM);
        ov_pid_status_t cs = pid_get_status(f->confpid);
        ov_pid_status_t rs = pid_get_status(f->runpid);
        const char *cst = (cs == OV_PID_ALIVE) ? "ALIVE" : (cs == OV_PID_ZOMBIE) ? "ZOMB" : "dead";
        const char *rst = (rs == OV_PID_ALIVE) ? "ALIVE" : (rs == OV_PID_ZOMBIE) ? "ZOMB" : "dead";
        int n = snprintf(NULL, 0, " Conf: %s (PID %d)  Run: %s (PID %d)", cst, (int) f->confpid,
                         rst, (int) f->runpid);
        H_ov_buf_printf(" Conf: %s (PID %d)  Run: %s (PID %d)", cst, (int) f->confpid, rst,
                        (int) f->runpid);
        H_render_pad_spaces(n, r.width);
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;
    }

    /* Parameters */
    if (f->nb_disp_params > 0)
    {
        H_ov_buf_pos(row + ri, r.col + 1);
        H_ov_theme_bg(OV_BG_PANEL);
        H_ov_theme_fg(OV_FG_TITLE);
        H_ov_buf_bold();
        int n = snprintf(NULL, 0, " Parameters (%d):", f->nb_disp_params);
        H_ov_buf_printf(" Parameters (%d):", f->nb_disp_params);
        H_ov_buf_reset_attr();
        H_ov_theme_bg(OV_BG_PANEL);

        /* Hint: ↑↓ ENTER (only when graph focused) */
        if (lay->focus == OV_FOCUS_GRAPH && lay->graph_tab_mode == 2)
        {
            H_ov_theme_fg(OV_FG_DIM);
            int h = snprintf(NULL, 0, "  [↑↓ ENTER]");
            H_ov_buf_printf("  [↑↓ ENTER]");
            H_render_pad_spaces(n + h, r.width);
        }
        else
        {
            H_render_pad_spaces(n, r.width);
        }
        if (!skip_draw)
        {
            ri++;
        }
        line_idx++;

        /* Clamp param_sel */
        if (lay->param_sel >= f->nb_disp_params)
        {
            lay->param_sel = f->nb_disp_params - 1;
        }

        /* Auto-scroll to keep selection visible */
        {
            int vis_rows = max_rows - ri;
            if (vis_rows < 1)
            {
                vis_rows = 1;
            }
            if (lay->param_sel >= 0)
            {
                if (lay->param_sel < lay->param_scroll)
                {
                    lay->param_scroll = lay->param_sel;
                }
                if (lay->param_sel >= lay->param_scroll + vis_rows)
                {
                    lay->param_scroll = lay->param_sel - vis_rows + 1;
                }
            }
            if (lay->param_scroll < 0)
            {
                lay->param_scroll = 0;
            }
        }

        const OV_FPS_PARAMS *params = ov_fps_get_params(f->name);
        for (int dp = 0; dp < f->nb_disp_params; dp++)
        {
            /* Skip rows above scroll window */
            if (dp < lay->param_scroll)
            {
                line_idx++;
                continue;
            }

            int is_sel =
                (dp == lay->param_sel && lay->focus == OV_FOCUS_GRAPH && lay->graph_tab_mode == 2);
            int header_rows = 3 + (f->description[0] != '\0' ? 1 : 0);
            int is_hover    = (lay->mouse_hover && lay->hover_view == OV_FOCUS_GRAPH &&
                               lay->graph_tab_mode == 2 &&
                               lay->hover_idx == header_rows + (dp - lay->param_scroll));

            ov_rgb_t row_bg = is_sel ? OV_BG_SELECTED : (is_hover ? OV_BG_HOVER : OV_BG_PANEL);

            H_ov_buf_pos(row + ri, r.col + 1);
            H_ov_theme_bg(row_bg);

            uint64_t pfl =
                (params != NULL && dp < params->nb_disp_params) ? params->disp_param_flags[dp] : 0;
            uint32_t pt =
                (params != NULL && dp < params->nb_disp_params) ? params->disp_param_type[dp] : 0;
            const char *pname =
                (params != NULL && dp < params->nb_disp_params) ? params->disp_param_name[dp] : "";
            const char *pval =
                (params != NULL && dp < params->nb_disp_params) ? params->disp_param_value[dp] : "";

            /* Writability indicator */
            int writable = (pfl & FPFLAG_WRITESTATUS) != 0;
            if (is_sel)
            {
                H_ov_theme_fg(writable ? OV_FG_ACTIVE : OV_FG_DIM);
                H_ov_buf_printf(writable ? " \xe2\x9c\x8e" : " \xf0\x9f\x94\x92");
            }
            else
            {
                H_ov_buf_printf("  ");
            }

            /* Type badge */
            const char *tbadge = "???";
            ov_rgb_t    tcolor = OV_FG_DIM;
            if (pt == FPTYPE_INT64 || pt == FPTYPE_INT32)
            {
                tbadge = "INT";
                tcolor = (ov_rgb_t) { 120, 180, 255 };
            }
            else if (pt == FPTYPE_UINT64 || pt == FPTYPE_UINT32)
            {
                tbadge = "UINT";
                tcolor = (ov_rgb_t) { 100, 160, 220 };
            }
            else if (pt == FPTYPE_FLOAT64 || pt == FPTYPE_FLOAT32)
            {
                tbadge = "FLT";
                tcolor = (ov_rgb_t) { 180, 200, 100 };
            }
            else if (pt == FPTYPE_ONOFF)
            {
                tbadge = "ON/OFF";
                tcolor = (ov_rgb_t) { 220, 180, 60 };
            }
            else if (pt == FPTYPE_STREAMNAME)
            {
                tbadge = "STRM";
                tcolor = OV_FG_STREAM;
            }
            else if (pt == FPTYPE_FPSNAME)
            {
                tbadge = "FPS";
                tcolor = OV_FG_FPS;
            }
            else if (FPTYPE_IS_STRING(pt))
            {
                tbadge = "STR";
                tcolor = (ov_rgb_t) { 200, 160, 120 };
            }
            else if (pt == FPTYPE_PID)
            {
                tbadge = "PID";
                tcolor = OV_FG_PROC;
            }
            else if (pt == FPTYPE_TIMESPEC)
            {
                tbadge = "TIME";
                tcolor = (ov_rgb_t) { 160, 180, 200 };
            }

            H_ov_theme_fg(tcolor);
            H_ov_buf_printf("[%-6s]", tbadge);

            /* Parameter name */
            H_ov_theme_fg(is_sel ? OV_FG_BRIGHT : OV_FG_CONN);
            H_ov_buf_printf(" %-20.20s", pname);

            /* Value */
            if (pt == FPTYPE_STREAMNAME)
            {
                int      s_idx  = ov_find_stream_by_name(m, pval);
                ov_rgb_t vcolor = (s_idx >= 0) ? OV_FG_STREAM : OV_FG_DIM;
                if (is_sel || is_hover)
                {
                    H_ov_theme_fg(vcolor);
                    H_ov_buf_bold();
                }
                else
                {
                    H_ov_theme_fg(vcolor);
                }
            }
            else if (pt == FPTYPE_ONOFF)
            {
                int      is_on  = (strcmp(pval, "ON") == 0 || strcmp(pval, "1") == 0);
                ov_rgb_t vcolor = is_on ? (ov_rgb_t) { 100, 255, 100 } : OV_FG_DIM;
                if (is_sel || is_hover)
                {
                    H_ov_theme_fg(vcolor);
                    H_ov_buf_bold();
                }
                else
                {
                    H_ov_theme_fg(vcolor);
                }
            }
            else
            {
                H_ov_theme_fg(is_sel ? OV_FG_BRIGHT : OV_FG_TEXT);
            }

            int n2 = snprintf(NULL, 0, " = %s", pval);
            H_ov_buf_printf(" = %s", pval);

            if ((is_sel || is_hover) && (pt == FPTYPE_STREAMNAME || pt == FPTYPE_ONOFF))
            {
                H_ov_buf_reset_attr();
                H_ov_theme_bg(row_bg); // Restore background after reset
            }

            /* Pad remainder */
            /* 2 (icon) + 8 (badge) + 1 (sp)
             * + 20 (name) + n2 (val) */
            H_render_pad_spaces(2 + 8 + 1 + 20 + n2, r.width);

            if (!skip_draw)
            {
                ri++;
            }
            line_idx++;
        } // for disp_params
    } // if nb_disp_params > 0

    lay->detail_total_lines = line_idx;
    for (; ri < max_rows; ri++)
    {
        clear_row(row + ri, r.col + 1, r.width - 2, OV_BG_PANEL);
    }
    H_ov_buf_reset_attr();
    return 1;
} // ov_fps__render_detail_fps
