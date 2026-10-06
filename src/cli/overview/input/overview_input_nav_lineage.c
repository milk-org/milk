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
 * ov_input_get_graph_start_node - find graph node index corresponding to active selection.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: Node index in m->nodes, or -1 if no matching selection.
 */
int ov_input_get_graph_start_node(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    ov_focus_t eff_focus   = lay->freeze ? lay->freeze_focus : lay->focus;
    int        target_type = -1;
    int        target_idx  = -1;

    if (eff_focus == OV_FOCUS_STREAMS || eff_focus == OV_FOCUS_GRAPH)
    {
        target_type = OV_NODE_STREAM;
        target_idx  = lay->freeze ? lay->freeze_sel_stream : lay->sel_stream;
    }
    else if (eff_focus == OV_FOCUS_PROCS)
    {
        target_type = OV_NODE_PROC;
        target_idx  = lay->freeze ? lay->freeze_sel_proc : lay->sel_proc;
    }
    else if (eff_focus == OV_FOCUS_FPS)
    {
        target_type = OV_NODE_FPS;
        target_idx  = lay->freeze ? lay->freeze_sel_fps : lay->sel_fps;
    }

    if (target_type != -1 && target_idx != -1)
    {
        for (int i = 0; i < m->nb_nodes; i++)
        {
            if (m->nodes[i].type == target_type && m->nodes[i].index == target_idx)
            {
                return i;
            }
        }
    }
    return -1;
}

/**
 * ov_input_get_filtered_count - count visible items in a panel taking active filter into account.
 * @focus: Target focus panel (OV_FOCUS_STREAMS, OV_FOCUS_PROCS, OV_FOCUS_FPS).
 * @lay:   Pointer to layout structure.
 * @m:     Pointer to data model snapshot.
 *
 * Return: Number of items matching current filter.
 */
int ov_input_get_filtered_count(int focus, const OV_LAYOUT *lay, const OV_MODEL *m)
{
    int count = 0;
    if (focus == OV_FOCUS_STREAMS)
    {
        count            = m->nb_streams;
        const char *filt = ov_get_active_filter_for(lay, OV_FOCUS_STREAMS);
        if (filt[0] != '\0')
        {
            const char *names[OV_MAX_STREAMS];
            for (int i = 0; i < count; i++)
            {
                names[i] = m->streams[i].name;
            }
            int fidx[OV_MAX_STREAMS];
            count = ov_filter_build(filt, names, count, fidx, OV_MAX_STREAMS);
        }
    }
    else if (focus == OV_FOCUS_PROCS)
    {
        count            = m->nb_procs;
        const char *filt = ov_get_active_filter_for(lay, OV_FOCUS_PROCS);
        if (filt[0] != '\0')
        {
            const char *names[OV_MAX_PROCS];
            for (int i = 0; i < count; i++)
            {
                names[i] = m->procs[i].name;
            }
            int fidx[OV_MAX_PROCS];
            count = ov_filter_build(filt, names, count, fidx, OV_MAX_PROCS);
        }
    }
    else if (focus == OV_FOCUS_FPS)
    {
        count            = m->nb_fps;
        const char *filt = ov_get_active_filter_for(lay, OV_FOCUS_FPS);
        if (filt[0] != '\0')
        {
            const char *names[OV_MAX_FPS];
            for (int i = 0; i < count; i++)
            {
                names[i] = m->fps[i].name;
            }
            int fidx[OV_MAX_FPS];
            count = ov_filter_build(filt, names, count, fidx, OV_MAX_FPS);
        }
    }
    return count;
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
    if (lay->focus == OV_FOCUS_STREAMS && lay->sel_stream >= 0 && lay->sel_stream < m->nb_streams)
    {
        current_node = m->streams[lay->sel_stream].node_idx;
        target_type  = OV_NODE_STREAM;
    }
    else if (lay->focus == OV_FOCUS_PROCS && lay->sel_proc >= 0 && lay->sel_proc < m->nb_procs)
    {
        current_node = m->procs[lay->sel_proc].node_idx;
        target_type  = OV_NODE_PROC;
    }
    else if (lay->focus == OV_FOCUS_FPS && lay->sel_fps >= 0 && lay->sel_fps < m->nb_fps)
    {
        current_node = m->fps[lay->sel_fps].node_idx;
        target_type  = OV_NODE_FPS;
    }
    else if (lay->focus == OV_FOCUS_GRAPH)
    {
        int start_node = ov_input_get_graph_start_node(lay, m);
        if (start_node >= 0)
        {
            SG_RENDER_NODE rnodes[OV_MAX_NODES];
            int n_rnodes = sg_compute_render_nodes(m, start_node, lay->lineage_mode, rnodes);
            if (lay->sel_graph >= 0 && lay->sel_graph < n_rnodes)
            {
                current_node = rnodes[lay->sel_graph].node_idx;
            }
        }
        target_type = -1;
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
            int start_node = ov_input_get_graph_start_node(lay, m);
            if (start_node >= 0)
            {
                SG_RENDER_NODE rnodes[OV_MAX_NODES];
                int n_rnodes = sg_compute_render_nodes(m, start_node, lay->lineage_mode, rnodes);
                for (int i = 0; i < n_rnodes; i++)
                {
                    if (rnodes[i].node_idx == target_node)
                    {
                        lay->sel_graph = i;
                        break;
                    }
                }
            }
        }
        else if (lay->focus == OV_FOCUS_STREAMS && tn->type == OV_NODE_STREAM)
        {
            lay->sel_stream         = tn->index;
            lay->sel_name_stream[0] = '\0';
        }
        else if (lay->focus == OV_FOCUS_PROCS && tn->type == OV_NODE_PROC)
        {
            lay->sel_proc         = tn->index;
            lay->sel_name_proc[0] = '\0';
        }
        else if (lay->focus == OV_FOCUS_FPS && tn->type == OV_NODE_FPS)
        {
            lay->sel_fps         = tn->index;
            lay->sel_name_fps[0] = '\0';
            lay->fps_param_focus = 0;
        }
    }
    return 1;
}
