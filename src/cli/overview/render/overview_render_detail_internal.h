// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef OVERVIEW_RENDER_DETAIL_INTERNAL_H
#define OVERVIEW_RENDER_DETAIL_INTERNAL_H

/**
 * @file overview_render_detail_internal.h
 * @brief Internal macros and helper declarations for detail pane renderers.
 */

#include "overview_render_internal.h"
#include <sched.h>
#include "fps_types.h"

#define skip_draw (line_idx < lay->scroll_detail || ri >= max_rows)

#define H_ov_buf_pos(r, c) \
    if (!(skip_draw))      \
    ov_buf_pos(r, c)
#define H_ov_theme_bg(c) \
    if (!(skip_draw))    \
    ov_theme_bg(c)
#define H_ov_theme_fg(c) \
    if (!(skip_draw))    \
    ov_theme_fg(c)
#define H_ov_buf_bold() \
    if (!(skip_draw))   \
    ov_buf_bold()
#define H_ov_buf_reset_attr() \
    if (!(skip_draw))         \
    ov_buf_reset_attr()
#define H_ov_buf_printf(...) \
    if (!(skip_draw))        \
    ov_buf_printf(__VA_ARGS__)
#define H_render_pad_spaces(n, w) \
    if (!(skip_draw))             \
    render_pad_spaces(n, w)

#define H_detail_row(ri, line_idx, row, col, width, ...) \
    do                                                   \
    {                                                    \
        H_ov_buf_pos((row) + (ri), (col) + 1);           \
        {                                                \
            int _n = snprintf(NULL, 0, __VA_ARGS__);     \
            H_ov_buf_printf(__VA_ARGS__);                \
            H_render_pad_spaces(_n, (width));            \
        }                                                \
        if (!skip_draw)                                  \
        {                                                \
            (ri)++;                                      \
        }                                                \
        (line_idx)++;                                    \
    } while (0)

int ov_fps__render_detail_stream(
    OV_LAYOUT      *lay,
    const OV_MODEL *m,
    int             ssel,
    OV_RECT         r,
    int             max_rows,
    int             row);

int ov_fps__render_detail_proc(
    OV_LAYOUT      *lay,
    const OV_MODEL *m,
    int             psel,
    OV_RECT         r,
    int             max_rows,
    int             row);

int ov_fps__render_detail_fps(
    OV_LAYOUT      *lay,
    const OV_MODEL *m,
    int             fsel,
    OV_RECT         r,
    int             max_rows,
    int             row);

#endif /* OVERVIEW_RENDER_DETAIL_INTERNAL_H */
