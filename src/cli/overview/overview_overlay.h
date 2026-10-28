// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_overlay.h
 * @brief State of the overlays and prompts drawn over the panels (help, theme popup, loop rename)
 */

#ifndef OVERVIEW_OVERLAY_H
#define OVERVIEW_OVERLAY_H

#include <stdint.h>
#include <time.h>

/* Help overlay state */
typedef struct
{
    int      show;
    int      mode;          /* 0 = Controls & keybindings, 1 = Intro to milk-CTRL */
    int      intro_scroll;  /* scroll row in full intro view */
    int      sel;           /* cursor row in help */
    uint32_t expand;        /* bitmask: 1=expanded */
    char     search[64];    /* keyword/topic search query */
    int      search_active; /* 1 if typing in search prompt */
    int      search_cursor; /* cursor position in search query */
} OV_HELP_STATE;

/* Theme selector popup */
typedef struct
{
    int             active;
    int             sel;
    struct timespec ts;
    OV_RECT         rect;
} OV_THEME_POPUP;

/* Loop rename prompt */
typedef struct
{
    int  active;
    char buf[64];
    int  cursor;
} OV_RENAME_STATE;

/* F5 parameter-tree history: heap, allocated on first use, defined in input/overview_input_history.c */
typedef struct OV_INPUT_HISTORY OV_INPUT_HISTORY;

#endif /* OVERVIEW_OVERLAY_H */
