// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_data_adj.c
 * @brief   Adjacency lists of the link graph: each node's out-edges and in-edges, in edge order.
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
