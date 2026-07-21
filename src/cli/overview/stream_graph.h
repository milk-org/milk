// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file stream_graph.h
 * @brief Stream dependency graph traversal API
 *
 * Provides BFS-based ancestor/descendant lineage
 * computation for shared-memory streams.  Supports
 * three traversal modes (trigger, input, full) and
 * detects cycles (loops) in the stream graph.
 *
 * Used by both the standalone milk-stream-graph tool
 * and the milk-CTRL CONNECTIONS panel.
 *
 * Dependencies: overview_data.h (OV_MODEL)
 */

#ifndef STREAM_GRAPH_H
#define STREAM_GRAPH_H

#include "overview_data.h"

/* =========================================================
 * Constants
 * ========================================================= */

#define SG_MAX_LINEAGE 128
#define SG_MAX_DEPTH 16

/* =========================================================
 * Traversal modes
 * ========================================================= */

/**
 * sg_mode_t - lineage traversal mode.
 *
 * @SG_MODE_TRIGGER: follow trigger edges only.
 * @SG_MODE_INPUT:   follow FPS input/read edges.
 * @SG_MODE_FULL:    follow all FPS stream params
 *                   (both input and output).
 * @SG_MODE_FPS:     follow only FPS-stream edges.
 */
typedef enum
{
    SG_MODE_TRIGGER = 0,
    SG_MODE_INPUT   = 1,
    SG_MODE_FULL    = 2,
    SG_MODE_FPS     = 3,
} sg_mode_t;

/* =========================================================
 * Result structures
 * ========================================================= */

/**
 * SG_LINEAGE_ENTRY - one node in the lineage result.
 *
 * @stream_idx: index in model->streams[]
 * @depth:      BFS distance from root (always positive)
 * @via_name:   name of intermediary proc/FPS
 * @is_loop:    1 if this node closes a cycle
 */
typedef struct
{
    int  stream_idx;
    int  depth;
    char via_name[40];
    int  is_loop;
} SG_LINEAGE_ENTRY;

/**
 * SG_RENDER_NODE - flattened graph node for vertical rendering.
 */
typedef struct
{
    int  node_idx;
    int  depth;
    int  order;
    int  type;
    char name[64];
    int  is_loop;
} SG_RENDER_NODE;

/**
 * SG_TREE_NODE - flattened tree node for lineage rendering.
 */
typedef struct
{
    int  stream_idx;       /* Index in m->streams */
    int  is_target;        /* 1 if this is the target root stream */
    int  is_target_proc;   /* 1 if the reader_name matches the target process */
    int  depth;            /* Graph depth */
    char name[64];         /* Stream name */
    char reader_name[64];  /* Name of the process/fps reading this stream */
    char tree_prefix[128]; /* Prefix strings for rendering (e.g. "├── ") */
} SG_TREE_NODE;

/**
 * SG_LINEAGE - complete lineage result for one stream.
 *
 * @ancestors:      upstream streams
 * @nb_ancestors:   count of ancestors
 * @descendants:    downstream streams
 * @nb_descendants: count of descendants
 * @has_loop:       1 if any cycle was detected
 * @cycle_path:     stream indices forming the cycle
 * @cycle_len:      length of cycle_path
 */
typedef struct
{
    SG_LINEAGE_ENTRY ancestors[SG_MAX_LINEAGE];
    int              nb_ancestors;

    SG_LINEAGE_ENTRY descendants[SG_MAX_LINEAGE];
    int              nb_descendants;

    int has_loop;
    int cycle_path[SG_MAX_DEPTH];
    int cycle_len;
} SG_LINEAGE;

/* =========================================================
 * API
 * ========================================================= */

/**
 * sg_compute_lineage - BFS traversal for stream lineage.
 * @m:          system model (streams, FPS, procs, graph)
 * @stream_idx: index of the root stream
 * @mode:       traversal mode (trigger/input/full)
 * @out:        result (cleared on entry)
 *
 * Traverses the directed graph in both directions
 * (upstream for ancestors, downstream for descendants).
 * Detects cycles where the root stream appears as its
 * own ancestor or descendant.
 */
void sg_compute_lineage(const OV_MODEL *m, int stream_idx, sg_mode_t mode, SG_LINEAGE *out);

/**
 * sg_compute_node_depths - Generic BFS traversal for any node type.
 * @m:          system model (streams, FPS, procs, graph)
 * @start_node: index of the root node
 * @mode:       traversal mode (trigger/input/full)
 * @node_depths: array of size OV_MAX_NODES to store results
 *
 * Computes upstream (negative) and downstream (positive) depths
 * from start_node to all other reachable nodes.
 */
void sg_compute_node_depths(const OV_MODEL *m, int start_node, sg_mode_t mode, int8_t *node_depths);

/**
 * sg_mode_label - human-readable label for a mode.
 * @mode: traversal mode
 *
 * Return: static string like "Trigger", "Input", "Full".
 */
const char *sg_mode_label(sg_mode_t mode);

/**
 * sg_compute_render_nodes - Computes a flattened, sorted list of nodes for vertical graph rendering.
 * @m:          system model
 * @start_node: root node to traverse from
 * @mode:       traversal mode
 * @out_nodes:  pre-allocated array (size OV_MAX_NODES) to store result
 *
 * Returns: number of nodes placed in out_nodes.
 */
int sg_compute_render_nodes(const OV_MODEL *m,
                            int             start_node,
                            sg_mode_t       mode,
                            SG_RENDER_NODE *out_nodes);

/**
 * sg_compute_render_tree - Computes a top-down flattened list of stream nodes representing the lineage.
 * @m:          system model
 * @start_node: root node to traverse from
 * @mode:       traversal mode
 * @out_nodes:  pre-allocated array (size OV_MAX_NODES) to store result
 *
 * Returns: number of nodes placed in out_nodes.
 */
int sg_compute_render_tree(const OV_MODEL *m,
                           int             start_node,
                           sg_mode_t       mode,
                           SG_TREE_NODE   *out_nodes);

#endif /* STREAM_GRAPH_H */
