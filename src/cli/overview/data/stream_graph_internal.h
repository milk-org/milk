// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef STREAM_GRAPH_INTERNAL_H
#define STREAM_GRAPH_INTERNAL_H

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

#include "overview_data.h"
#include "stream_graph.h"

#define SG_BITS_PER_WORD 64
#define SG_BSET_WORDS(n) (((n) + SG_BITS_PER_WORD - 1) / SG_BITS_PER_WORD)

static inline void sg_bset(uint64_t *words, int idx)
{
    words[idx / SG_BITS_PER_WORD] |= (UINT64_C(1) << (idx % SG_BITS_PER_WORD));
}

static inline int sg_bget(const uint64_t *words, int idx)
{
    return (int) ((words[idx / SG_BITS_PER_WORD] >> (idx % SG_BITS_PER_WORD)) & 1);
}

typedef struct
{
    int node;
    int depth;
} sg_bfs_item_t;

/**
 * sg_edge_matches_mode_from_stream - check if edge qualifies for traversal
 * @e:    edge to test
 * @mode: traversal mode
 *
 * Return: 1 if edge should be followed.
 */
static inline int sg_edge_matches_mode_from_stream(const OV_EDGE *e, sg_mode_t mode)
{
    switch (mode)
    {
    case SG_MODE_TRIGGER:
        return (e->type == OV_EDGE_STREAM_TRIGGERS_PROC || e->type == OV_EDGE_PROC_TRIGGER_STREAM);

    case SG_MODE_INPUT:
        return (e->type == OV_EDGE_FPS_INPUT_STREAM || e->type == OV_EDGE_STREAM_READ_BY_PROC);

    case SG_MODE_FULL:
        return (e->type == OV_EDGE_STREAM_TRIGGERS_PROC || e->type == OV_EDGE_PROC_TRIGGER_STREAM ||
                e->type == OV_EDGE_FPS_INPUT_STREAM || e->type == OV_EDGE_FPS_OUTPUT_STREAM ||
                e->type == OV_EDGE_STREAM_READ_BY_PROC || e->type == OV_EDGE_PROC_WRITES_STREAM);

    case SG_MODE_FPS:
        return (e->type == OV_EDGE_FPS_INPUT_STREAM || e->type == OV_EDGE_FPS_OUTPUT_STREAM);
    }

    return 0;
}

#endif /* STREAM_GRAPH_INTERNAL_H */
