// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "stream_graph_internal.h"

/**
 * @brief Get the label string for a graph display mode.
 */
const char *sg_mode_label(sg_mode_t mode)
{
    switch (mode)
    {
    case SG_MODE_TRIGGER:
        return "Trigger";
    case SG_MODE_INPUT:
        return "Input";
    case SG_MODE_FULL:
        return "Full";
    case SG_MODE_FPS:
        return "FPS";
    }
    return "Unknown";
}

/* =========================================================
 * Generic node BFS (all node types)
 * ========================================================= */

/**
 * sg_compute_node_depths - Compute signed graph topological depths from start node
 * @m:           Pointer to data model
 * @start_node:  Source node index
 * @mode:        Graph mode filter
 * @node_depths: Output array of depth offsets (negative for ancestors, positive for descendants)
 */
void sg_compute_node_depths(const OV_MODEL *m, int start_node, sg_mode_t mode, int8_t *node_depths)
{
    for (int i = 0; i < OV_MAX_NODES; i++)
    {
        node_depths[i] = 127;
    }
    if (start_node < 0 || start_node >= m->nb_nodes)
    {
        return;
    }
    node_depths[start_node] = 0;

    ov_node_type_t start_type = m->nodes[start_node].type;

    /* Downstream */
    uint64_t visited[SG_BSET_WORDS(OV_MAX_NODES)];
    memset(visited, 0, sizeof(visited));
    sg_bset(visited, start_node);

    sg_bfs_item_t queue[OV_MAX_NODES];
    int           qhead = 0, qtail = 0;
    queue[qtail].node  = start_node;
    queue[qtail].depth = 0;
    qtail++;

    while (qhead < qtail)
    {
        sg_bfs_item_t cur = queue[qhead++];

        if (cur.node != start_node)
        {
            int d = cur.depth;
            if (m->nodes[cur.node].type != start_type)
            {
                d++;
            }
            if (d > 127)
            {
                d = 127;
            }
            if (node_depths[cur.node] == 127)
            {
                node_depths[cur.node] = (int8_t) d;
            }
        }

        if (cur.depth >= SG_MAX_DEPTH)
        {
            continue;
        }

        const OV_NODE *cn = &m->nodes[cur.node];

        for (int k = m->adj_out_off[cur.node]; k < m->adj_out_off[cur.node + 1]; k++)
        {
            const OV_EDGE *e = &m->edges[m->adj_out[k]];

            if (cn->type == OV_NODE_STREAM)
            {
                if (!sg_edge_matches_mode_from_stream(e, mode))
                {
                    continue;
                }
            }
            else if (mode == SG_MODE_FPS)
            {
                /* FPS mode: only FPS<->stream edges */
                if (!sg_edge_matches_mode_from_stream(e, mode))
                {
                    continue;
                }
            }
            else
            {
                /* From FPS/PROC: follow edges to streams
                 * and FPS_RUNS_PROC edges (FPS->proc) */
                int tgt_type = m->nodes[e->tgt_node].type;
                if (tgt_type != OV_NODE_STREAM && e->type != OV_EDGE_FPS_RUNS_PROC)
                {
                    continue;
                }
            }

            int next = e->tgt_node;
            if (next < 0 || next >= m->nb_nodes)
            {
                continue;
            }
            if (sg_bget(visited, next))
            {
                continue;
            }

            sg_bset(visited, next);
            int d = cur.depth;
            if (m->nodes[next].type == start_type)
            {
                d++;
            }

            queue[qtail].node  = next;
            queue[qtail].depth = d;
            qtail++;
        }
    }

    /* Upstream */
    memset(visited, 0, sizeof(visited));
    sg_bset(visited, start_node);
    qhead              = 0;
    qtail              = 0;
    queue[qtail].node  = start_node;
    queue[qtail].depth = 0;
    qtail++;

    while (qhead < qtail)
    {
        sg_bfs_item_t cur = queue[qhead++];

        if (cur.node != start_node)
        {
            int d = cur.depth;
            if (m->nodes[cur.node].type != start_type)
            {
                d++;
            }
            if (d > 127)
            {
                d = 127;
            }
            if (node_depths[cur.node] == 127)
            {
                node_depths[cur.node] = (int8_t) (-d);
            }
        }

        if (cur.depth >= SG_MAX_DEPTH)
        {
            continue;
        }

        const OV_NODE *cn = &m->nodes[cur.node];

        for (int k = m->adj_in_off[cur.node]; k < m->adj_in_off[cur.node + 1]; k++)
        {
            const OV_EDGE *e = &m->edges[m->adj_in[k]];

            int next = e->src_node;
            if (next < 0 || next >= m->nb_nodes)
            {
                continue;
            }

            if (mode == SG_MODE_FPS)
            {
                /* FPS mode: restrict to FPS<->stream
                 * edges for both stream and non-stream
                 * nodes */
                if (!sg_edge_matches_mode_from_stream(e, mode))
                {
                    continue;
                }
            }
            else if (cn->type != OV_NODE_STREAM)
            {
                /* Going upstream from PROC/FPS:
                 * accept stream-related edges and
                 * FPS_RUNS_PROC (proc<-FPS).
                 * Stream nodes: no filter (accept
                 * all reverse edges). */
                if (!sg_edge_matches_mode_from_stream(e, mode) && e->type != OV_EDGE_FPS_RUNS_PROC)
                {
                    continue;
                }
            }

            if (sg_bget(visited, next))
            {
                continue;
            }

            sg_bset(visited, next);
            int d = cur.depth;
            if (m->nodes[next].type == start_type)
            {
                d++;
            }

            queue[qtail].node  = next;
            queue[qtail].depth = d;
            qtail++;
        }
    }
}
