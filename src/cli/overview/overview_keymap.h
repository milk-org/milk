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

/* Context bits (ov_key_ctx) */
#define OV_CTX_MODEL (1U << 0)         /* a model is loaded */
#define OV_CTX_CTRL (1U << 1)          /* CONTROL mode */
#define OV_CTX_FROZEN (1U << 2)        /* a selection is frozen */
#define OV_CTX_DASH (1U << 3)          /* view F2 */
#define OV_CTX_F5 (1U << 4)            /* view F5 */
#define OV_CTX_LOOP_LIST (1U << 5)     /* F7, or the LOOPS sub-tab focused */
#define OV_CTX_FOCUS_STREAMS (1U << 6) /* focus on a panel */
#define OV_CTX_FOCUS_PROCS (1U << 7)
#define OV_CTX_FOCUS_FPS (1U << 8)
#define OV_CTX_FOCUS_GRAPH (1U << 9)
#define OV_CTX_PARAM_TREE (1U << 10)  /* F5 parameter tree focused */
#define OV_CTX_MSEL (1U << 11)        /* FPS batch marks set */
#define OV_CTX_FOCUS_SHOWN (1U << 12) /* the focused panel is on screen */
#define OV_CTX_FOCUS_LIST (1U << 13)  /* the focus is a list panel (streams, processes, FPS) */

/* Status-line hint of a key: shown while the context has every @need bit and no @forbid bit */
typedef struct
{
    const char *key;     /* as shown: "k" */
    const char *meaning; /* one or two words: "term" */
    unsigned    need;
    unsigned    forbid;
} OV_KEYHINT;

#define OV_NOHINT { NULL, NULL, 0, 0 }

typedef struct
{
    const char *display; /* keys as documented: "s / S / A" */
    const char *label;   /* one line: help list and -h row */
    const char *detail;  /* help detail pane, NULL = label */
    int         section; /* help section HS_*, -1 = -h only */
    int         group;   /* -h section ov_keygroup_t, -1 = help only */
    int         flags;   /* OV_KF_* */
    OV_KEYHINT  hint;    /* status-line hint, OV_NOHINT = none */
} OV_KEYDEF;

extern const OV_KEYDEF   ov_keys[];
extern const int         ov_nb_keys;
extern const char *const ov_keygroup_title[OV_KG_COUNT];

int ov_key_hints(unsigned ctx, const OV_KEYHINT **out, int max);

#endif /* OVERVIEW_KEYMAP_H */
