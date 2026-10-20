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
static void dfs_search_cycles(OV_MODEL *model,
                              int       start_node,
                              int       curr_node,
                              int       depth,
                              int      *path,
                              uint8_t  *in_path,
                              const int adj[OV_MAX_NODES][64],
                              const int adj_cnt[OV_MAX_NODES],
                              int      *total_explored)
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

/** loop_sig_cmp - qsort order of loops by canonical signature, so loop IDs do not depend on scan order. */
static int loop_sig_cmp(const void *a, const void *b)
{
    return strcmp(((const OV_LOOP *) a)->signature, ((const OV_LOOP *) b)->signature);
}

/** loop_add_rate - fold one member rate into the loop min/max; 0 Hz counts (a stalled member is the bottleneck). */
static void loop_add_rate(OV_LOOP *lp, double hz, int *nrates)
{
    if (hz < lp->min_hz)
    {
        lp->min_hz = hz;
    }
    if (hz > lp->max_hz)
    {
        lp->max_hz = hz;
    }
    (*nrates)++;
}

/** loop_add_proc_health - fold one process state into the loop RUN/PAUS/ERR/STAL flags and rate. */
static void loop_add_proc_health(OV_LOOP *lp, const OV_PROC *pr, int *nrates)
{
    if (pr->loopstat != PROCESSINFO_LOOPSTAT_ACTIVE)
    {
        lp->is_running = 0;
    }
    else if (pr->stale_count >= OV_STALE_SCANS)
    {
        lp->is_stale = 1;
    }
    if (pr->loopstat == PROCESSINFO_LOOPSTAT_PAUSE)
    {
        lp->is_paused = 1;
    }
    if (pr->loopstat == PROCESSINFO_LOOPSTAT_ERROR || pr->loopstat == PROCESSINFO_LOOPSTAT_CRASHED)
    {
        lp->is_error = 1;
    }
    loop_add_rate(lp, pr->loop_hz, nrates);
}

/**
 * ov_detect_loops - Detect all elementary directed cycles in the graph.
 * @model: System model containing streams, processes, FPS, nodes, and edges
 * @mode:  Traversal mode (trigger, input, full)
 */
void ov_detect_loops(OV_MODEL *model, sg_mode_t mode)
{
    if (model == NULL)
    {
        return;
    }
    model->loops_truncated = 0;
    if (model->nb_nodes <= 0)
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
            if (total_explored > 4096 || model->nb_loops >= OV_MAX_LOOPS)
            {
                model->loops_truncated = 1;
            }
            if (model->nb_loops >= OV_MAX_LOOPS)
            {
                break;
            }
        }
    }

    /* Stable IDs: order by signature, then number and name */
    qsort(model->loops, (size_t) model->nb_loops, sizeof(model->loops[0]), loop_sig_cmp);
    for (int l = 0; l < model->nb_loops; l++)
    {
        model->loops[l].loop_id = l + 1;
        ov_loop_set_names(model, &model->loops[l]);
    }

    /* Tag member entities and aggregate health/rate metrics */
    for (int l = 0; l < model->nb_loops; l++)
    {
        OV_LOOP *lp   = &model->loops[l];
        uint32_t mask = (UINT32_C(1) << l);

        lp->is_running = 1;
        lp->is_paused  = 0;
        lp->is_stale   = 0;
        lp->is_error   = 0;
        lp->min_hz     = 1e9;
        lp->max_hz     = 0.0;
        int nrates     = 0;

        for (int i = 0; i < lp->nb_streams; i++)
        {
            int si = lp->stream_indices[i];
            if (si >= 0 && si < model->nb_streams)
            {
                OV_STREAM *st = &model->streams[si];
                st->loop_mask |= mask;
                st->nb_loops++;
                if (st->primary_loop_id == 0)
                {
                    st->primary_loop_id = lp->loop_id;
                }
                loop_add_rate(lp, st->update_hz, &nrates);
            }
        }

        for (int i = 0; i < lp->nb_procs; i++)
        {
            int pi = lp->proc_indices[i];
            if (pi >= 0 && pi < model->nb_procs)
            {
                OV_PROC *pr = &model->procs[pi];
                pr->loop_mask |= mask;
                pr->nb_loops++;
                if (pr->primary_loop_id == 0)
                {
                    pr->primary_loop_id = lp->loop_id;
                }
                loop_add_proc_health(lp, pr, &nrates);
            }
        }

        /* An FPS runs only through its run process: no RUN by vacuous truth */
        for (int i = 0; i < lp->nb_fps; i++)
        {
            int fi = lp->fps_indices[i];
            if (fi >= 0 && fi < model->nb_fps)
            {
                OV_FPS *f = &model->fps[fi];
                f->loop_mask |= mask;
                f->nb_loops++;
                if (f->primary_loop_id == 0)
                {
                    f->primary_loop_id = lp->loop_id;
                }
                int rp = f->run_alive ? ov_find_proc_by_pid(model, f->runpid) : -1;
                if (!f->run_alive)
                {
                    lp->is_running = 0;
                }
                else if (rp >= 0)
                {
                    loop_add_proc_health(lp, &model->procs[rp], &nrates);
                }
            }
        }

        if (nrates == 0)
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
