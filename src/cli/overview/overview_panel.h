// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_panel.h
 * @brief State of one list panel (streams, processes or FPS) and of each sub-tab; OV_LAYOUT holds them
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
    int       hscroll;       /* scrollable columns skipped on the left */
    int       freeze_row;    /* row of the frozen entry, or -1 */
    int       sort_key;      /* 0 = name, 1..N = column-specific */
    int       sort_dir;      /* 0 = ascending, 1 = descending */
    int       highlight_col; /* column cursor (table index) */
    uint32_t  collapsed;     /* collapsed columns, one bit per logical column */
    int       hover_idx;     /* model index under the mouse, or -1 */
    int       filter_on;     /* regex filter applied (the pattern is kept when off) */
    char      filter[64];    /* regex filter pattern */
} OV_PANEL_STATE;

/* Sub-tabs of the bottom-right panel (DASH) and of F6 */
typedef enum
{
    OV_SUB_CONN = 0, /* CONNECTIONS */
    OV_SUB_LOOPS,    /* LOOPS (F7 shows its list too) */
    OV_SUB_DETAILS,  /* DETAILS: text, or the FPS parameter list */
    OV_SUB_RES,      /* RESOURCES */
    OV_SUBTAB_COUNT,
} ov_subtab_t;

/* Pages of the F3 stream inspector */
typedef enum
{
    OV_INSP_INFO = 0,
    OV_INSP_SEMS,
    OV_INSP_TAB_COUNT
} ov_insp_tab_t;

/* List cursor of one sub-tab, indexed by ov_subtab_t */
typedef struct
{
    int sel;    /* cursor row (-1 = none) */
    int scroll; /* first drawn row */
} OV_SUBTAB_STATE;

#endif /* OVERVIEW_PANEL_H */
