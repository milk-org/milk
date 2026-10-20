// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_keymap.h
 * @brief Key table: one entry per documented key; the help overlay, -h and the status-line hints read it
 */

#ifndef OVERVIEW_KEYMAP_H
#define OVERVIEW_KEYMAP_H

#include <stddef.h>

/* Help overlay sections */
enum
{
    HS_INTRO = 0, /* Introduction & Overview    */
    HS_NAV,       /* Global & Navigation        */
    HS_STREAMS,   /* Streams Panel (STRM)       */
    HS_PROCS,     /* Processes Panel (PROC)     */
    HS_FPS,       /* FPS Panel (FPS)            */
    HS_GRAPH,     /* Graph & Lineage (CONN)     */
    HS_CMDLOG,    /* Command Log & Display      */
    HS_MOUSE,     /* Mouse Interactions         */
    HS_COLORS,    /* Theme & Colors             */
    HS_COUNT
};

/* -h key sections */
typedef enum
{
    OV_KG_VIEWS = 0,
    OV_KG_NAV,
    OV_KG_COLUMNS,
    OV_KG_LOOPS,
    OV_KG_CTRL,
    OV_KG_COUNT,
} ov_keygroup_t;

#define OV_KF_CTRL 0x01 /* acts in CONTROL mode only */

typedef struct
{
    const char *display; /* keys as documented: "s / S / A" */
    const char *label;   /* one line: help list and -h row */
    const char *detail;  /* help detail pane, NULL = label */
    int         section; /* help section HS_*, -1 = -h only */
    int         group;   /* -h section ov_keygroup_t, -1 = help only */
    int         flags;   /* OV_KF_* */
    const char *footer;  /* status-line hint, NULL = none */
} OV_KEYDEF;

extern const OV_KEYDEF   ov_keys[];
extern const int         ov_nb_keys;
extern const char *const ov_keygroup_title[OV_KG_COUNT];

size_t ov_key_footer(char *buf, size_t n);

#endif /* OVERVIEW_KEYMAP_H */
