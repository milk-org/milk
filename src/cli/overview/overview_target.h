// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_target.h
 * @brief Identity of a selected list entry: rows move, identities do not
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

#endif /* OVERVIEW_TARGET_H */
