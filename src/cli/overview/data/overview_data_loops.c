// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_loops_internal.h"
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* =========================================================
 * DFS Cycle Search
 * ========================================================= */

/**
 * dfs_search_cycles - Recursively search for cycles using depth-first search
 * @model:          Pointer to data model
 * @start_node:     Initial node index of traversal
 * @curr_node:      Current node being visited
 * @depth:          Current path depth
 * @path:           Array recording current traversal path
 * @in_path:        Bitmask/array marking nodes currently on the DFS path
 * @adj:            Adjacency matrix of outgoing edges
 * @adj_cnt:        Count of outgoing edges per node
 * @total_explored: Pointer to running counter of explored states
 */
static void dfs_search_cycles(
    OV_MODEL  *model,
    int        start_node,
    int        curr_node,
    int        depth,
    int       *path,
    uint8_t   *in_path,
    const int  adj[OV_MAX_NODES][64],
    const int  adj_cnt[OV_MAX_NODES],
    int       *total_explored)
{
    if (model->nb_loops >= OV_MAX_LOOPS || *total_explored > 4096)
    {
        return;
    }
    (*total_explored)++;

    if (depth >= OV_MAX_LOOP_NODES)
    {
        return;
    }

    path[depth]        = curr_node;
    in_path[curr_node] = 1;

    for (int i = 0; i < adj_cnt[curr_node]; i++)
    {
        int next = adj[curr_node][i];

        if (next == start_node && depth >= 1)
        {
            /* Cycle found */
            register_cycle(model, path, depth + 1);
            if (model->nb_loops >= OV_MAX_LOOPS)
            {
                in_path[curr_node] = 0;
                return;
            }
        }
        else if (!in_path[next])
        {
            /* Avoid visiting nodes with index < start_node to reduce redundant traversals */
            if (next >= start_node)
            {
                dfs_search_cycles(model, start_node, next, depth + 1, path, in_path, adj, adj_cnt,
                                  total_explored);
            }
        }
    }

    in_path[curr_node] = 0;
}

/* =========================================================
 * Loop Detection & Metric Computation Engine
 * ========================================================= */

/**
 * ov_detect_loops - Detect all elementary directed cycles in the graph.
 * @model: System model containing streams, processes, FPS, nodes, and edges
 * @mode:  Traversal mode (trigger, input, full)
 */
void ov_detect_loops(OV_MODEL *model, sg_mode_t mode)
{
    if (model == NULL || model->nb_nodes <= 0)
    {
        return;
    }

    if (!s_names_loaded)
    {
        ov_loop_names_load(model);
    }

    /* Reset loop counts on entities */
    for (int i = 0; i < model->nb_streams; i++)
    {
        model->streams[i].loop_mask       = 0;
        model->streams[i].nb_loops        = 0;
        model->streams[i].primary_loop_id = 0;
    }
    for (int i = 0; i < model->nb_procs; i++)
    {
        model->procs[i].loop_mask       = 0;
        model->procs[i].nb_loops        = 0;
        model->procs[i].primary_loop_id = 0;
    }
    for (int i = 0; i < model->nb_fps; i++)
    {
        model->fps[i].loop_mask       = 0;
        model->fps[i].nb_loops        = 0;
        model->fps[i].primary_loop_id = 0;
    }

    model->nb_loops = 0;

    /* Build adjacency lists */
    static int adj[OV_MAX_NODES][64];
    static int adj_cnt[OV_MAX_NODES];
    memset(adj_cnt, 0, sizeof(adj_cnt));

    for (int ei = 0; ei < model->nb_edges; ei++)
    {
        const OV_EDGE *e = &model->edges[ei];
        if (is_valid_loop_edge(model, e, mode))
        {
            int u = e->src_node;
            int v = e->tgt_node;
            if (adj_cnt[u] < 64)
            {
                int dup = 0;
                for (int k = 0; k < adj_cnt[u]; k++)
                {
                    if (adj[u][k] == v)
                    {
                        dup = 1;
                        break;
                    }
                }
                if (!dup)
                {
                    adj[u][adj_cnt[u]++] = v;
                }
            }
        }
    }

    /* Search for cycles starting from each stream node */
    int     path[OV_MAX_LOOP_NODES];
    uint8_t in_path[OV_MAX_NODES];
    memset(in_path, 0, sizeof(in_path));

    for (int ni = 0; ni < model->nb_nodes; ni++)
    {
        if (model->nodes[ni].type == OV_NODE_STREAM)
        {
            int total_explored = 0;
            dfs_search_cycles(model, ni, ni, 0, path, in_path, adj, adj_cnt, &total_explored);
            if (model->nb_loops >= OV_MAX_LOOPS)
            {
                break;
            }
        }
    }

    /* Tag member entities and aggregate health/rate metrics */
    for (int l = 0; l < model->nb_loops; l++)
    {
        OV_LOOP *lp   = &model->loops[l];
        uint32_t mask = (UINT32_C(1) << l);

        lp->is_running   = 1;
        lp->is_paused    = 0;
        lp->is_stale     = 0;
        lp->is_error     = 0;
        lp->min_hz       = 1e9;
        lp->max_hz       = 0.0;
        int active_rates = 0;

        /* Streams */
        for (int i = 0; i < lp->nb_streams; i++)
        {
            int si = lp->stream_indices[i];
            if (si >= 0 && si < model->nb_streams)
            {
                model->streams[si].loop_mask |= mask;
                model->streams[si].nb_loops++;
                if (model->streams[si].primary_loop_id == 0)
                {
                    model->streams[si].primary_loop_id = lp->loop_id;
                }
                if (model->streams[si].update_hz > 0.0)
                {
                    if (model->streams[si].update_hz < lp->min_hz)
                    {
                        lp->min_hz = model->streams[si].update_hz;
                    }
                    if (model->streams[si].update_hz > lp->max_hz)
                    {
                        lp->max_hz = model->streams[si].update_hz;
                    }
                    active_rates++;
                }
            }
        }

        /* Processes */
        for (int i = 0; i < lp->nb_procs; i++)
        {
            int pi = lp->proc_indices[i];
            if (pi >= 0 && pi < model->nb_procs)
            {
                model->procs[pi].loop_mask |= mask;
                model->procs[pi].nb_loops++;
                if (model->procs[pi].primary_loop_id == 0)
                {
                    model->procs[pi].primary_loop_id = lp->loop_id;
                }

                const OV_PROC *pr = &model->procs[pi];
                if (pr->loopstat != 1)
                {
                    lp->is_running = 0;
                }
                if (pr->loopstat == 2)
                {
                    lp->is_paused = 1;
                }
                if (pr->loopstat == 4)
                {
                    lp->is_error = 1;
                }
                if (pr->stale_count > 2)
                {
                    lp->is_stale = 1;
                }

                if (pr->loop_hz > 0.0)
                {
                    if (pr->loop_hz < lp->min_hz)
                    {
                        lp->min_hz = pr->loop_hz;
                    }
                    if (pr->loop_hz > lp->max_hz)
                    {
                        lp->max_hz = pr->loop_hz;
                    }
                    active_rates++;
                }
            }
        }

        /* FPS */
        for (int i = 0; i < lp->nb_fps; i++)
        {
            int fi = lp->fps_indices[i];
            if (fi >= 0 && fi < model->nb_fps)
            {
                model->fps[fi].loop_mask |= mask;
                model->fps[fi].nb_loops++;
                if (model->fps[fi].primary_loop_id == 0)
                {
                    model->fps[fi].primary_loop_id = lp->loop_id;
                }
            }
        }

        if (active_rates == 0)
        {
            lp->min_hz = 0.0;
        }
    }

    /* Compute pairwise overlaps and shared vs exclusive component counts */
    for (int i = 0; i < model->nb_loops; i++)
    {
        OV_LOOP *lp1         = &model->loops[i];
        lp1->overlap_mask    = 0;
        lp1->nb_shared_nodes = 0;

        for (int k = 0; k < lp1->nb_nodes; k++)
        {
            int            ni      = lp1->node_indices[k];
            const OV_NODE *n       = &model->nodes[ni];
            int            n_loops = 0;

            if (n->type == OV_NODE_STREAM && n->index >= 0 && n->index < model->nb_streams)
            {
                n_loops = model->streams[n->index].nb_loops;
            }
            else if (n->type == OV_NODE_PROC && n->index >= 0 && n->index < model->nb_procs)
            {
                n_loops = model->procs[n->index].nb_loops;
            }
            else if (n->type == OV_NODE_FPS && n->index >= 0 && n->index < model->nb_fps)
            {
                n_loops = model->fps[n->index].nb_loops;
            }

            if (n_loops > 1)
            {
                lp1->nb_shared_nodes++;
            }
        }

        lp1->nb_exclusive_nodes = lp1->nb_nodes - lp1->nb_shared_nodes;

        for (int j = 0; j < model->nb_loops; j++)
        {
            if (i == j)
            {
                continue;
            }
            const OV_LOOP *lp2    = &model->loops[j];
            int            shares = 0;

            for (int k1 = 0; k1 < lp1->nb_nodes; k1++)
            {
                for (int k2 = 0; k2 < lp2->nb_nodes; k2++)
                {
                    if (lp1->node_indices[k1] == lp2->node_indices[k2])
                    {
                        shares = 1;
                        break;
                    }
                }
                if (shares)
                {
                    break;
                }
            }

            if (shares)
            {
                lp1->overlap_mask |= (UINT32_C(1) << j);
            }
        }
    }
}
