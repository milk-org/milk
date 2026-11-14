// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_target.h
 * @brief Identity of a selected list entry: rows move, identities do not.
 *
 * Purpose:     a cursor, a frozen entry, an FPS batch mark or a pending y/N action keeps naming
 *              the same entity while rows move between scans, sorts and filter changes; it does
 *              not pass to an entity that takes its row, or a recreated stream (new inode) or a
 *              new process with the same PID (new start time).
 * Does:        OV_TARGET: a name, plus the inode for a stream, the PID and start ticks for a
 *              process ("" = none). ov_act_t lists the destructive and batch verbs that ask y/N
 *              first; OV_CONFIRM is a pending confirmation, its target captured when it opens.
 *              OV_NB_LIST_PANELS and OV_MSEL_MAX size the panel and batch-mark arrays.
 * Used by:     overview_layout.h and overview_panel.h (OV_LAYOUT); the identity resolver
 *              (render/overview_render_select.c: ov_tgt_set, ov_tgt_find) and the view cache
 *              (render/overview_render_view.c); the y/N verbs (input/overview_input_confirm.c,
 *              overview_input_actions.c, overview_input_mouse.c, render/overview_render_popup.c).
 * See also:    render/overview_render_select.c; docs/guide/milk-CTRL.md, "CONTROL mode".
 */

#ifndef OVERVIEW_TARGET_H
#define OVERVIEW_TARGET_H

#include <stdint.h>

#define OV_TGT_NAMELEN 100  /* >= stream, process and FPS name lengths */
#define OV_NB_LIST_PANELS 3 /* indexed by ov_focus_t STREAMS, PROCS, FPS */

/* name "" = none; inode set for streams, pid (and start ticks once known) for processes */
typedef struct
{
    char     name[OV_TGT_NAMELEN];
    uint64_t inode;
    int      pid;
    uint64_t start;
} OV_TARGET;

#define OV_MSEL_MAX 200 /* FPS batch marks, == OV_MAX_FPS */

/* Destructive and batch verbs that ask y/N first */
typedef enum
{
    OV_ACT_NONE = 0,
    OV_ACT_PROC_SIGKILL,
    OV_ACT_PROC_REMOVE,
    OV_ACT_PROC_CLEANUP,
    OV_ACT_FPS_SIGKILL,
    OV_ACT_FPS_ERASE,
    OV_ACT_STREAM_DELETE,
    OV_ACT_BATCH_SIGNAL, /* arg: signal number */
    OV_ACT_BATCH_RUN,    /* arg: 1 start the stopped ones, 0 stop */
    OV_ACT_BATCH_CONF,   /* arg: 1 start the stopped ones, 0 stop */
} ov_act_t;

/* Pending confirmation (verb NONE = closed); the target identity is captured when it opens */
typedef struct
{
    ov_act_t  verb;
    int       arg;
    int       panel;
    OV_TARGET tgt;
} OV_CONFIRM;

#endif /* OVERVIEW_TARGET_H */
