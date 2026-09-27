// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "stream_graph_internal.h"

static void sg_dfs_tree(
    const OV_MODEL *m,
    int             current_stream,
    int             reader_node_idx,
    int             target_stream,
    int             target_proc,
    const uint64_t *S_words,
    sg_mode_t       mode,
    const char     *prefix,
    int             is_last,
    int             is_root,
    int             depth,
    int            *path,
    int             path_len,
    SG_TREE_NODE   *out_nodes,
    int            *nb_out_nodes)
{
    if (*nb_out_nodes >= OV_MAX_NODES)
    {
        return;
    }

    /* Cycle detection */
    int is_cycle = 0;
    for (int i = 0; i < path_len; i++)
    {
        if (path[i] == current_stream)
        {
            is_cycle = 1;
            break;
        }
    }

    SG_TREE_NODE *node = &out_nodes[*nb_out_nodes];
    node->stream_idx   = current_stream;
    node->is_target    = (current_stream == target_stream);
    node->depth        = depth;
    node->is_loop      = is_cycle;
    strncpy(node->name, m->streams[current_stream].name, sizeof(node->name) - 1);
    node->name[sizeof(node->name) - 1] = '\0';

    /* Find reader proc */
    node->reader_name[0] = '\0';
    node->is_target_proc = 0;
    if (reader_node_idx >= 0)
    {
        const OV_NODE *rn = &m->nodes[reader_node_idx];
        strncpy(node->reader_name, rn->name, sizeof(node->reader_name) - 1);
        node->reader_name[sizeof(node->reader_name) - 1] = '\0';
        if (rn->type == OV_NODE_PROC && rn->index == target_proc)
        {
            node->is_target_proc = 1;
        }
    }

    /* Build prefix */
    if (is_root)
    {
        node->tree_prefix[0] = '\0';
    }
    else
    {
        snprintf(node->tree_prefix, sizeof(node->tree_prefix), "%s%s", prefix,
                 is_last ? "\xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 "
                         : "\xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 "); /* └──  and ├──  */
    }

    if (is_cycle)
    {
        (*nb_out_nodes)++;
        return;
    }

    (*nb_out_nodes)++;

    path[path_len] = current_stream;

    /* Find children nodes (stream, reader) */
    struct
    {
        int stream_idx;
        int reader_node_idx;
    } child_nodes[OV_MAX_NODES];
    int nb_child_nodes = 0;

    if (reader_node_idx >= 0)
    {
        /* Find streams written by the reader process/FPS */
        int child_streams[OV_MAX_STREAMS];
        int nb_child_streams = 0;

        for (int ei = 0; ei < m->nb_edges; ei++)
        {
            const OV_EDGE *e = &m->edges[ei];
            if (e->src_node == reader_node_idx &&
                (e->type == OV_EDGE_PROC_WRITES_STREAM || e->type == OV_EDGE_FPS_OUTPUT_STREAM))
            {
                int c_node = e->tgt_node;
                if (c_node >= 0 && c_node < m->nb_nodes && m->nodes[c_node].type == OV_NODE_STREAM)
                {
                    int c_stream = m->nodes[c_node].index;
                    if (sg_bget(S_words, c_stream))
                    {
                        int duplicate = 0;
                        for (int k = 0; k < nb_child_streams; k++)
                        {
                            if (child_streams[k] == c_stream)
                            {
                                duplicate = 1;
                                break;
                            }
                        }
                        if (!duplicate)
                        {
                            child_streams[nb_child_streams++] = c_stream;
                        }
                    }
                }
            }
        }

        /* For each child stream, find its reader processes/FPS in S */
        for (int i = 0; i < nb_child_streams; i++)
        {
            int c_stream = child_streams[i];
            int n_node   = m->streams[c_stream].node_idx;
            int readers[OV_MAX_PROCS];
            int nb_readers = 0;

            if (n_node >= 0)
            {
                for (int ei = 0; ei < m->nb_edges; ei++)
                {
                    const OV_EDGE *e1 = &m->edges[ei];
                    if (e1->src_node == n_node && sg_edge_matches_mode_from_stream(e1, mode))
                    {
                        int r_node    = e1->tgt_node;
                        int duplicate = 0;
                        for (int k = 0; k < nb_readers; k++)
                        {
                            if (readers[k] == r_node)
                            {
                                duplicate = 1;
                                break;
                            }
                        }
                        if (!duplicate)
                        {
                            readers[nb_readers++] = r_node;
                        }
                    }
                }
            }

            if (nb_readers == 0)
            {
                if (nb_child_nodes < OV_MAX_NODES)
                {
                    child_nodes[nb_child_nodes].stream_idx      = c_stream;
                    child_nodes[nb_child_nodes].reader_node_idx = -1;
                    nb_child_nodes++;
                }
            }
            else
            {
                for (int r = 0; r < nb_readers; r++)
                {
                    if (nb_child_nodes < OV_MAX_NODES)
                    {
                        child_nodes[nb_child_nodes].stream_idx      = c_stream;
                        child_nodes[nb_child_nodes].reader_node_idx = readers[r];
                        nb_child_nodes++;
                    }
                }
            }
        }
    }

    /* Recurse */
    char child_prefix[128];
    if (is_root)
    {
        child_prefix[0] = '\0';
    }
    else
    {
        snprintf(child_prefix, sizeof(child_prefix), "%s%s", prefix,
                 is_last ? "    " : "\xe2\x94\x82   "); /* "    " and "│   " */
    }

    for (int i = 0; i < nb_child_nodes; i++)
    {
        sg_dfs_tree(m, child_nodes[i].stream_idx, child_nodes[i].reader_node_idx, target_stream,
                    target_proc, S_words, mode, child_prefix, (i == nb_child_nodes - 1), 0,
                    depth + 1, path, path_len + 1, out_nodes, nb_out_nodes);
    }
}

/**
 * sg_compute_render_tree - Generate hierarchical tree layout with branch graphics
 * @m:          Pointer to data model
 * @start_node: Root node index for tree
 * @mode:       Graph mode filter
 * @out_nodes:  Output array of tree nodes with branch prefixes
 *
 * Return: Total number of tree nodes populated.
 */
int sg_compute_render_tree(
    const OV_MODEL *m,
    int             start_node,
    sg_mode_t       mode,
    SG_TREE_NODE   *out_nodes)
{
    int nb_out = 0;
    if (start_node < 0 || start_node >= m->nb_nodes)
    {
        return 0;
    }

    const OV_NODE *sn            = &m->nodes[start_node];
    int            target_stream = -1;
    int            target_proc   = -1;

    if (sn->type == OV_NODE_STREAM)
    {
        target_stream = sn->index;
    }
    else if (sn->type == OV_NODE_PROC)
    {
        target_proc = sn->index;
    }

    uint64_t S_words[SG_BSET_WORDS(OV_MAX_STREAMS)];
    memset(S_words, 0, sizeof(S_words));

    if (target_stream != -1)
    {
        SG_LINEAGE lin;
        memset(&lin, 0, sizeof(lin));
        sg_compute_lineage(m, target_stream, mode, &lin);

        sg_bset(S_words, target_stream);
        for (int i = 0; i < lin.nb_ancestors; i++)
        {
            sg_bset(S_words, lin.ancestors[i].stream_idx);
        }
        for (int i = 0; i < lin.nb_descendants; i++)
        {
            sg_bset(S_words, lin.descendants[i].stream_idx);
        }
    }
    else if (target_proc != -1)
    {
        int8_t depths[OV_MAX_NODES];
        memset(depths, 127, sizeof(depths));
        sg_compute_node_depths(m, start_node, mode, depths);
        depths[start_node] = 0;

        for (int i = 0; i < m->nb_nodes; i++)
        {
            if (depths[i] < 127 || depths[i] > -127)
            {
                if (m->nodes[i].type == OV_NODE_STREAM)
                {
                    sg_bset(S_words, m->nodes[i].index);
                }
            }
        }
    }
    else
    {
        return 0;
    }

    /* Find parents for everyone in S */
    int has_parent[OV_MAX_STREAMS];
    memset(has_parent, 0, sizeof(has_parent));

    for (int i = 0; i < m->nb_streams; i++)
    {
        if (!sg_bget(S_words, i))
        {
            continue;
        }

        int n_node = m->streams[i].node_idx;
        if (n_node < 0)
        {
            continue;
        }

        for (int ei = 0; ei < m->nb_edges; ei++)
        {
            const OV_EDGE *e1 = &m->edges[ei];
            if (e1->src_node == n_node && sg_edge_matches_mode_from_stream(e1, mode))
            {
                int p_node = e1->tgt_node;
                for (int ej = 0; ej < m->nb_edges; ej++)
                {
                    const OV_EDGE *e2 = &m->edges[ej];
                    if (e2->src_node == p_node)
                    {
                        int c_node = e2->tgt_node;
                        if (c_node >= 0 && c_node < m->nb_nodes &&
                            m->nodes[c_node].type == OV_NODE_STREAM)
                        {
                            int c_stream = m->nodes[c_node].index;
                            if (sg_bget(S_words, c_stream))
                            {
                                has_parent[c_stream] = 1;
                            }
                        }
                    }
                }
            }
        }
    }

    /* Find roots */
    int roots[OV_MAX_STREAMS];
    int nb_roots = 0;
    for (int i = 0; i < m->nb_streams; i++)
    {
        if (sg_bget(S_words, i) && !has_parent[i])
        {
            roots[nb_roots++] = i;
        }
    }

    if (nb_roots == 0)
    {
        /* Cycle graph with no absolute root. Use first available as root. */
        for (int i = 0; i < m->nb_streams; i++)
        {
            if (sg_bget(S_words, i))
            {
                roots[nb_roots++] = i;
                break;
            }
        }
    }

    int path[OV_MAX_STREAMS];
    /* For each root, find its reader processes/FPS and call sg_dfs_tree */
    struct
    {
        int stream_idx;
        int reader_node_idx;
    } root_nodes[OV_MAX_STREAMS * 4];
    int nb_root_nodes = 0;

    for (int i = 0; i < nb_roots; i++)
    {
        int r_stream = roots[i];
        int n_node   = m->streams[r_stream].node_idx;
        int readers[OV_MAX_PROCS];
        int nb_readers = 0;

        if (n_node >= 0)
        {
            for (int ei = 0; ei < m->nb_edges; ei++)
            {
                const OV_EDGE *e1 = &m->edges[ei];
                if (e1->src_node == n_node && sg_edge_matches_mode_from_stream(e1, mode))
                {
                    int r_node    = e1->tgt_node;
                    int duplicate = 0;
                    for (int k = 0; k < nb_readers; k++)
                    {
                        if (readers[k] == r_node)
                        {
                            duplicate = 1;
                            break;
                        }
                    }
                    if (!duplicate)
                    {
                        readers[nb_readers++] = r_node;
                    }
                }
            }
        }

        if (nb_readers == 0)
        {
            if (nb_root_nodes < OV_MAX_STREAMS * 4)
            {
                root_nodes[nb_root_nodes].stream_idx      = r_stream;
                root_nodes[nb_root_nodes].reader_node_idx = -1;
                nb_root_nodes++;
            }
        }
        else
        {
            for (int r = 0; r < nb_readers; r++)
            {
                if (nb_root_nodes < OV_MAX_STREAMS * 4)
                {
                    root_nodes[nb_root_nodes].stream_idx      = r_stream;
                    root_nodes[nb_root_nodes].reader_node_idx = readers[r];
                    nb_root_nodes++;
                }
            }
        }
    }

    for (int i = 0; i < nb_root_nodes; i++)
    {
        sg_dfs_tree(m, root_nodes[i].stream_idx, root_nodes[i].reader_node_idx, target_stream,
                    target_proc, S_words, mode, "", (i == nb_root_nodes - 1), 1, 0, path, 0,
                    out_nodes, &nb_out);
    }

    return nb_out;
}
