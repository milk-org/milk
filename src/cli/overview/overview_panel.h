// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_panel.h
 * @brief State of one list panel (streams, processes or FPS); OV_LAYOUT holds one per panel
 */

#ifndef OVERVIEW_PANEL_H
#define OVERVIEW_PANEL_H

#include <stdint.h>

#include "overview_target.h"

/* Indexed by ov_focus_t: OV_FOCUS_STREAMS, OV_FOCUS_PROCS, OV_FOCUS_FPS */
typedef struct
{
    OV_TARGET tgt;           /* cursor identity; sel is the drawn row derived from it */
    OV_TARGET freeze_tgt;    /* frozen entry identity; freeze_row derived from it */
    int       sel;           /* cursor row */
    int       scroll;        /* first drawn row */
    int       hscroll;       /* horizontal column offset */
    int       freeze_row;    /* row of the frozen entry, or -1 */
    int       sort_key;      /* 0 = name, 1..N = column-specific */
    int       sort_dir;      /* 0 = ascending, 1 = descending */
    int       highlight_col; /* column cursor (visible index) */
    uint32_t  collapsed;     /* collapsed columns, one bit per logical column */
    int       hover_idx;     /* model index under the mouse, or -1 */
    int       filter_on;     /* regex filter applied (the pattern is kept when off) */
    char      filter[64];    /* regex filter pattern */
} OV_PANEL_STATE;

#endif /* OVERVIEW_PANEL_H */
