/**
 * @file overview_layout.c
 * @brief Panel layout computation for milkCTRL
 */

#include "overview_defs.h"
#include "overview_ansi.h"
#include "overview_layout.h"

void ov_layout_compute(OV_LAYOUT *lay)
{
    ov_get_terminal_size(&lay->term_rows,
                         &lay->term_cols);

    int W = lay->term_cols;
    int H = lay->term_rows;

    /* Header: 1 row at top */
    lay->r_header = (OV_RECT){1, 1, 1, W};

    /* Status: 1 row at bottom */
    lay->r_status = (OV_RECT){H, 1, 1, W};

    /* Command log strip above status bar */
    int log_h = lay->cmdlog_rows;
    if (log_h < 0)
    {
        log_h = 0;
    }
    if (log_h > 0)
    {
        lay->r_cmdlog = (OV_RECT){
            H - log_h, 1, log_h, W
        };
    }
    else
    {
        lay->r_cmdlog = (OV_RECT){0, 0, 0, 0};
    }

    /* Usable height excludes header + status + log */
    int body_top;
    int body_h;

    if (lay->view == OV_VIEW_DASHBOARD)
    {
        /* Row 2 = preview bar for selected item */
        body_top = 3;
        body_h   = H - 3 - log_h;
        if (body_h < 4)
        {
            body_h = 4;
        }
        /* 2x2 grid layout */
        int half_w = W / 2;
        int half_h = body_h / 2;

        lay->r_streams = (OV_RECT){
            body_top, 1,
            half_h, half_w
        };
        lay->r_procs = (OV_RECT){
            body_top, half_w + 1,
            half_h, W - half_w
        };
        lay->r_fps = (OV_RECT){
            body_top + half_h, 1,
            body_h - half_h, half_w
        };
        lay->r_graph = (OV_RECT){
            body_top + half_h, half_w + 1,
            body_h - half_h, W - half_w
        };
    }
    else
    {
        body_top = 2;
        body_h   = H - 2 - log_h;
        if (body_h < 4)
        {
            body_h = 4;
        }
        /* Full-screen for single-view modes */
        lay->r_streams = (OV_RECT){
            body_top, 1, body_h, W
        };
        lay->r_procs   = lay->r_streams;
        lay->r_fps     = lay->r_streams;
        lay->r_graph   = lay->r_streams;

        /* F5 split: ~40% list, ~60% params */
        if (lay->view == OV_VIEW_FPS)
        {
            int lw = (W * 2) / 5;
            if (lw < 20)
            {
                lw = 20;
            }
            lay->r_fps_list   = (OV_RECT){
                body_top, 1, body_h, lw
            };
            lay->r_fps_params = (OV_RECT){
                body_top, lw + 1, body_h, W - lw
            };
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
