// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef MILK_STREAM_GRAPH_H
#define MILK_STREAM_GRAPH_H

/**
 * @file milk-stream-graph.h
 * @brief Internal declarations for milk-stream-graph tool.
 */

#include "overview_defs.h"
#include "overview_data.h"
#include "stream_graph.h"

/* =========================================================
 * Version and descriptions
 * ========================================================= */

#define SG_VERSION "1.0.0"
#define SG_ONELINE "stream dependency graph with loop detection"

#define SG_DESC_LONG                                                  \
    "Compute and display ancestor/descendant lineage for a stream.\n" \
    "Supports trigger, input, and full traversal modes with\n"        \
    "text, TrueColor ANSI, JSON, and interactive output formats."

/* =========================================================
 * ANSI escape helpers (TrueColor)
 * ========================================================= */

#define SGC_RESET "\033[0m"
#define SGC_BOLD "\033[1m"
#define SGC_DIM "\033[2m"
#define SGC_BLINK "\033[5m"

/* TrueColor foreground */
#define SGC_FG(r, g, b) "\033[38;2;" #r ";" #g ";" #b "m"

#define SGC_STREAM SGC_FG(100, 200, 255)
#define SGC_PROC SGC_FG(120, 220, 120)
#define SGC_FPS SGC_FPS_COLOR
#define SGC_FPS_COLOR SGC_FG(240, 200, 80)
#define SGC_LOOP SGC_FG(255, 80, 80)
#define SGC_DEPTH SGC_FG(160, 160, 160)
#define SGC_HEADER SGC_FG(200, 180, 255)
#define SGC_TEXT SGC_FG(200, 200, 200)
#define SGC_ARROW SGC_FG(140, 140, 180)

/* Box-drawing chars */
#define SG_TREE_V "│"
#define SG_TREE_BR "├"
#define SG_TREE_END "└"
#define SG_TREE_H "─"

/* =========================================================
 * Output mode
 * ========================================================= */

typedef enum
{
    OUT_TEXT   = 0,
    OUT_PRETTY = 1,
    OUT_JSON   = 2,
} sg_output_t;

/* =========================================================
 * Function declarations
 * ========================================================= */

void sg_scan_model(OV_MODEL *model);

void sg_print_text(const OV_MODEL   *m,
                   const char       *stream_name,
                   sg_mode_t         mode,
                   const SG_LINEAGE *lin);

void sg_print_json(const OV_MODEL   *m,
                   const char       *stream_name,
                   sg_mode_t         mode,
                   const SG_LINEAGE *lin);

void sg_print_pretty(const OV_MODEL   *m,
                     const char       *stream_name,
                     sg_mode_t         mode,
                     const SG_LINEAGE *lin);

void sg_interactive(OV_MODEL *model, const char *initial_stream, sg_mode_t mode);

#endif /* MILK_STREAM_GRAPH_H */
