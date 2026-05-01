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

    int body_top = 2;
    int body_h   = H - 2;

    if (lay->view == OV_VIEW_DASHBOARD)
    {
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
        /* Full-screen for single-view modes */
        lay->r_streams = (OV_RECT){
            body_top, 1, body_h, W
        };
        lay->r_procs   = lay->r_streams;
        lay->r_fps     = lay->r_streams;
        lay->r_graph   = lay->r_streams;
    }
}
