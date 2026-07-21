// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_layout.c
 * @brief Panel layout computation for milk-CTRL
 */

#include "overview_defs.h"
#include "overview_ansi.h"
#include "overview_layout.h"

/**
 * @brief Compute panel layout from terminal dimensions.
 *
 * Calculates column widths and row counts for
 * the overview dashboard panels.
 */
void ov_layout_compute(OV_LAYOUT *lay)
{
    ov_get_terminal_size(&lay->term_rows, &lay->term_cols);

    int W = lay->term_cols;
    int H = lay->term_rows;

    /* Header: 1 row at top */
    lay->r_header = (OV_RECT) { 1, 1, 1, W };

    /* Status: 1 row at bottom */
    lay->r_status = (OV_RECT) { H, 1, 1, W };

    /* Command log strip above status bar */
    int log_h = lay->cmdlog_rows;
    if (log_h < 0)
    {
        log_h = 0;
    }
    if (log_h > 0)
    {
        lay->r_cmdlog = (OV_RECT) { H - log_h, 1, log_h, W };
    }
    else
    {
        lay->r_cmdlog = (OV_RECT) { 0, 0, 0, 0 };
    }

    /* Usable height excludes header + status + log */
    int body_top;
    int body_h;

    if (lay->view == OV_VIEW_DASHBOARD)
    {
        /* Row 2 = preview bar for selected item */
        /* Row 3 = highlighted column description line */
        body_top = 4;
        body_h   = H - 4 - log_h;
        if (body_h < 4)
        {
            body_h = 4;
        }
        /* 2x2 grid layout using adjustable splits */
        int dash_w = (int) (W * lay->dash_split_v_ratio);
        int dash_h = (int) (body_h * lay->dash_split_h_ratio);

        if (dash_w < 20)
        {
            dash_w = 20;
        }
        if (dash_w > W - 20)
        {
            dash_w = W - 20;
        }

        if (dash_h < 4)
        {
            dash_h = 4;
        }
        if (dash_h > body_h - 4)
        {
            dash_h = body_h - 4;
        }

        lay->r_streams = (OV_RECT) { body_top, 1, dash_h, dash_w };
        lay->r_procs   = (OV_RECT) { body_top, dash_w + 1, dash_h, W - dash_w };
        lay->r_fps     = (OV_RECT) { body_top + dash_h, 1, body_h - dash_h, dash_w };
        lay->r_graph   = (OV_RECT) { body_top + dash_h, dash_w + 1, body_h - dash_h, W - dash_w };
    }
    else
    {
        /* Row 2 (or row 4 for FPS view) = highlighted column description line */
        body_top = 3;
        body_h   = H - 3 - log_h;
        if (lay->view == OV_VIEW_FPS)
        {
            if (body_h < 6)
            {
                body_h = 6;
            }
        }
        else
        {
            if (body_h < 4)
            {
                body_h = 4;
            }
        }
        /* Full-screen for single-view modes */
        lay->r_streams = (OV_RECT) { body_top, 1, body_h, W };
        lay->r_procs   = lay->r_streams;
        lay->r_fps     = lay->r_streams;
        lay->r_graph   = lay->r_streams;

        /* F5 split */
        if (lay->view == OV_VIEW_FPS)
        {
            int lw = (int) (W * lay->fps_split_ratio);
            if (lw < 20)
            {
                lw = 20;
            }
            if (lw > W - 20)
            {
                lw = W - 20;
            }
            lay->r_fps_list   = (OV_RECT) { body_top + 2, 1, body_h - 2, lw };
            lay->r_fps_params = (OV_RECT) { body_top + 2, lw + 1, body_h - 2, W - lw };
            /* Keep r_fps pointing to list for panel rendering */
            lay->r_fps = lay->r_fps_list;
        }
        else
        {
            lay->r_fps_list   = lay->r_fps;
            lay->r_fps_params = lay->r_fps;
        }
    }
}
