// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_data_adj.c
 * @brief   Adjacency lists of the link graph: each node's out-edges and in-edges, in edge order.
 *
 * Purpose:     graph walks visit the edges of one node without scanning every edge of the model.
 * Does:        ov_model_adjacency() fills OV_MODEL.adj_out and adj_in and their offset arrays by
 *              a counting sort (compressed sparse rows). Each node keeps its edges in edge order,
 *              so a walk visits them in the order they were found.
 * Used by:     ov_build_graph() (overview_data_graph.c), as its last step; declared in
 *              overview_data.h. The walks read the lists through OV_ADJ_B/OV_ADJ_E:
 *              stream_graph*.c, overview_data_loops*.c, and the lineage, loop and health code of
 *              render/ and input/. milk-fps-info also compiles this file.
 * Uses:        overview_data_internal.h (OV_MODEL).
 * See also:    overview_data_graph.c (the edges); overview_data.h (the adjacency fields).
 * Constraints: not reentrant: one static work array.
 */

#include "overview_data_internal.h"

/** csr_build - offsets @off[0..n] and edge lists @lst of the edges grouped by @end (source or target). */
static void csr_build(const OV_MODEL *m, int by_src, int *off, int *lst)
{
    static int pos[OV_MAX_NODES + 1];
    int        n = m->nb_nodes;
    memset(off, 0, (size_t) (n + 1) * sizeof(int));
    for (int ei = 0; ei < m->nb_edges; ei++)
    {
        int v = by_src ? m->edges[ei].src_node : m->edges[ei].tgt_node;
        if (v >= 0 && v < n)
        {
            off[v + 1]++;
        }
    }
    for (int v = 0; v < n; v++)
    {
        off[v + 1] += off[v];
    }
    memcpy(pos, off, (size_t) (n + 1) * sizeof(int));
    for (int ei = 0; ei < m->nb_edges; ei++) /* in edge order: traversals keep their order */
    {
        int v = by_src ? m->edges[ei].src_node : m->edges[ei].tgt_node;
        if (v >= 0 && v < n)
        {
            lst[pos[v]++] = ei;
        }
    }
}

/**
 * ov_model_adjacency - build the out- and in-edge lists of @m's graph (ov_build_graph does it last;
 * a model whose edges are set by hand calls it before traversing).
 */
void ov_model_adjacency(OV_MODEL *m)
{
    csr_build(m, 1, m->adj_out_off, m->adj_out);
    csr_build(m, 0, m->adj_in_off, m->adj_in);
}
