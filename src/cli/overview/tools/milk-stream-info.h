// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef MILK_STREAM_INFO_H
#define MILK_STREAM_INFO_H

/**
 * @file milk-stream-info.h
 * @brief Header for milk-stream-info tool.
 */

#include "overview_defs.h"
#include "overview_data.h"
#include "milk_help.h"

#define SI_ONELINE                         \
    "print detailed info and connections " \
    "for a shared-memory stream"

#define SI_DESC_LONG                                                \
    "Scan the ImageStreamIO shared-memory area and the FPS\n"       \
    "registry to build a connection graph, then print a rich\n"     \
    "diagnostic view for the specified stream: type, dimensions,\n" \
    "memory footprint, counters, semaphores, and connections\n"     \
    "(written by, triggers, read by, FPS linkage)."

/* Replace local ANSI macros with milk_help.h equivalents */
#define C_RST MH_RST
#define C_BOLD MH_BOLD
#define C_DIM MH_DIM
#define C_TITLE MH_TITLE
#define C_HDR MH_HDR
#define C_LABEL MH_DFLT
#define C_NAME MH_CMD
#define C_PROC MH_NOTE
#define C_FPS MH_NOTE
#define C_VAL MH_BOLD
#define C_ALIVE "\033[1;32m"
#define C_DEAD MH_ERR
#define C_WARN MH_ERR
#define C_SEP MH_DFLT

void print_stream_info(
    const OV_MODEL *m,
    int             si);

#endif /* MILK_STREAM_INFO_H */
