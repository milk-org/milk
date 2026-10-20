// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_nav_lineage.c
 * @brief Graph start node lookup, filtered count calculations, and DAG ancestry navigation
 */

#include "overview_input_internal.h"
#include <string.h>

/**
 * ov_input_get_filtered_count - number of rows the list panel draws (ov_rows).
 * @focus: Target focus panel (OV_FOCUS_STREAMS, OV_FOCUS_PROCS, OV_FOCUS_FPS).
 * @lay:   Pointer to layout structure.
 * @m:     Pointer to data model snapshot.
 *
 * Return: Number of items matching current filter.
 */
int ov_input_get_filtered_count(int focus, const OV_LAYOUT *lay, const OV_MODEL *m)
{
    int fidx[OV_MAX_NODES];
    return (focus < OV_FOCUS_GRAPH) ? ov_rows(lay, m, (ov_focus_t) focus, fidx, OV_MAX_NODES) : 0;
}

/**
 * find_relative_node_of_type - traverse BFS graph edges upstream or downstream.
 * @m:           Pointer to data model snapshot.
 * @start_node:  Source node index in graph.
 * @target_type: Desired node type (stream, proc, fps) or -1 for any.
 * @upstream:    1 for upstream parents, 0 for downstream children.
 *
 * Return: Target relative node index, or -1 if none found.
 */
static int find_relative_node_of_type(const OV_MODEL *m,
                                      int             start_node,
                                      int             target_type,
                                      int             upstream)
{
    int queue[OV_MAX_NODES];
    int visited[OV_MAX_NODES];
    memset(visited, 0, sizeof(visited));
    int head = 0;
    int tail = 0;

    queue[tail++]       = start_node;
    visited[start_node] = 1;

    while (head < tail)
    {
        int curr = queue[head++];

        for (int i = 0; i < m->nb_edges; i++)
        {
            int next = -1;
            if (upstream && m->edges[i].tgt_node == curr)
            {
                next = m->edges[i].src_node;
            }
            else if (!upstream && m->edges[i].src_node == curr)
            {
                next = m->edges[i].tgt_node;
            }

            if (next >= 0 && !visited[next])
            {
                if (target_type < 0 || m->nodes[next].type == target_type)
                {
                    return next;
                }
                visited[next] = 1;
                queue[tail++] = next;
            }
        }
    }
    return -1;
}

/**
 * ov_input__handle_ancestry_nav - navigate to immediate upstream parent or downstream child in DAG.
 * @key: Pressed key code (Shift+Up or Shift+Down).
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: 1 if ancestry navigation key was handled, 0 otherwise.
 */
int ov_input__handle_ancestry_nav(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (key != OV_KEY_SHIFT_UP && key != OV_KEY_SHIFT_DOWN)
    {
        return 0;
    }

    int current_node = -1;
    int target_type  = -1;
    int cur          = ov_sel_idx(lay, m, lay->focus);
    if (lay->focus == OV_FOCUS_STREAMS && cur >= 0)
    {
        current_node = m->streams[cur].node_idx;
        target_type  = OV_NODE_STREAM;
    }
    else if (lay->focus == OV_FOCUS_PROCS && cur >= 0)
    {
        current_node = m->procs[cur].node_idx;
        target_type  = OV_NODE_PROC;
    }
    else if (lay->focus == OV_FOCUS_FPS && cur >= 0)
    {
        current_node = m->fps[cur].node_idx;
        target_type  = OV_NODE_FPS;
    }
    else if (lay->focus == OV_FOCUS_GRAPH)
    {
        int                 n;
        const SG_TREE_NODE *rows = ov_graph_rows(lay, m, &n);
        if (lay->sub[OV_SUB_CONN].sel >= 0 && lay->sub[OV_SUB_CONN].sel < n)
        {
            current_node = m->streams[rows[lay->sub[OV_SUB_CONN].sel].stream_idx].node_idx;
        }
        target_type = OV_NODE_STREAM; /* CONNECTIONS rows are streams */
    }

    if (current_node < 0 || current_node >= m->nb_nodes)
    {
        return 1;
    }

    int target_node =
        find_relative_node_of_type(m, current_node, target_type, key == OV_KEY_SHIFT_UP);

    if (target_node >= 0 && target_node < m->nb_nodes)
    {
        const OV_NODE *tn = &m->nodes[target_node];
        if (lay->focus == OV_FOCUS_GRAPH)
        {
            int                 n;
            const SG_TREE_NODE *rows = ov_graph_rows(lay, m, &n);
            for (int i = 0; i < n; i++)
            {
                if (rows[i].stream_idx == tn->index && tn->type == OV_NODE_STREAM)
                {
                    lay->sub[OV_SUB_CONN].sel = i;
                    break;
                }
            }
        }
        else if ((lay->focus == OV_FOCUS_STREAMS && tn->type == OV_NODE_STREAM) ||
                 (lay->focus == OV_FOCUS_PROCS && tn->type == OV_NODE_PROC) ||
                 (lay->focus == OV_FOCUS_FPS && tn->type == OV_NODE_FPS))
        {
            ov_sel_model(lay, m, lay->focus, tn->index);
            if (lay->focus == OV_FOCUS_FPS)
            {
                lay->fps_param_focus = 0;
            }
        }
    }
    return 1;
}
