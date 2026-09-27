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

        for (int ei = 0; ei < m->nb_edges; ei++)
        {
            const OV_EDGE *e = &m->edges[ei];
            if (e->src_node != cur.node)
            {
                continue;
            }

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

        for (int ei = 0; ei < m->nb_edges; ei++)
        {
            const OV_EDGE *e = &m->edges[ei];
            if (e->tgt_node != cur.node)
            {
                continue;
            }

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

/**
 * sg_compute_render_nodes - Build flat ordered array of reachable nodes for rendering
 * @m:          Pointer to data model
 * @start_node: Source node index
 * @mode:       Graph mode filter
 * @out_nodes:  Output array of render nodes
 *
 * Return: Number of reachable nodes placed in @out_nodes.
 */
int sg_compute_render_nodes(const OV_MODEL *m,
                            int             start_node,
                            sg_mode_t       mode,
                            SG_RENDER_NODE *out_nodes)
{
    if (start_node < 0 || start_node >= m->nb_nodes)
    {
        return 0;
    }

    int8_t depths[OV_MAX_NODES];
    for (int i = 0; i < OV_MAX_NODES; ++i)
    {
        depths[i] = 127;
    }

    sg_compute_node_depths(m, start_node, mode, depths);

    /* Collect all reachable nodes */
    SG_RENDER_NODE temp_nodes[OV_MAX_NODES];
    int            nb_nodes = 0;

    for (int i = 0; i < m->nb_nodes; ++i)
    {
        if (depths[i] != 127)
        {
            temp_nodes[nb_nodes].node_idx = i;
            temp_nodes[nb_nodes].depth    = depths[i];
            int type_order = m->nodes[i].type; // OV_NODE_STREAM=0, OV_NODE_FPS=1, OV_NODE_PROC=2
            int reverse_type_order    = 0;
            ov_node_type_t start_type = m->nodes[start_node].type;

            if (start_type == OV_NODE_STREAM && depths[i] > 0)
            {
                reverse_type_order = 1;
            }
            if (start_type != OV_NODE_STREAM && depths[i] < 0)
            {
                reverse_type_order = 1;
            }

            if (reverse_type_order)
            {
                type_order = 2 - type_order; // Invert: STREAM(2), FPS(1), PROC(0)
            }

            temp_nodes[nb_nodes].order = depths[i] * 10 + type_order;
            temp_nodes[nb_nodes].type  = m->nodes[i].type;

            /* Detect loops using BFS lineage if needed, but for simplicity,
             * we can just mark is_loop = 0. Real loop detection requires
             * lineage structures. We'll leave it 0 for now. */
            temp_nodes[nb_nodes].is_loop = 0;

            strncpy(temp_nodes[nb_nodes].name, m->nodes[i].name,
                    sizeof(temp_nodes[nb_nodes].name) - 1);
            temp_nodes[nb_nodes].name[sizeof(temp_nodes[nb_nodes].name) - 1] = '\0';

            nb_nodes++;
        }
    }

    /* Sort by order (which embeds depth and topological type ordering) */
    for (int i = 0; i < nb_nodes - 1; ++i)
    {
        for (int j = 0; j < nb_nodes - i - 1; ++j)
        {
            if (temp_nodes[j].order > temp_nodes[j + 1].order)
            {
                SG_RENDER_NODE tmp = temp_nodes[j];
                temp_nodes[j]      = temp_nodes[j + 1];
                temp_nodes[j + 1]  = tmp;
            }
        }
    }

    /* Copy to output */
    for (int i = 0; i < nb_nodes; ++i)
    {
        out_nodes[i] = temp_nodes[i];
    }

    return nb_nodes;
}
/**
 * sg_dfs_tree - Recursively format graph tree hierarchy with Unicode box-drawing branches
 * @m:               Pointer to data model
 * @current_stream:  Current stream index in traversal
 * @reader_node_idx: Node index of consuming process or FPS
 * @target_stream:   Root stream index
 * @target_proc:     Root process index
 * @S_words:         Reachable streams bitmask
 * @mode:            Graph mode filter
 * @prefix:          Prefix indentation string for tree branches
 * @is_last:         Flag indicating if this is the last sibling
 * @is_root:         Flag indicating if this is the root node
 * @depth:           Current tree depth
 * @path:            Array tracking ancestor nodes on path for cycle detection
 * @path_len:        Length of @path
 * @out_nodes:       Output array of formatted tree nodes
 * @nb_out_nodes:    Running count of tree nodes in output
 */
