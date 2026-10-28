// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "stream_graph_internal.h"

#include <stdlib.h>

/* A child of a tree node: a stream and one of its readers (-1: none) */
typedef struct
{
    int stream_idx;
    int reader_node_idx;
} SG_CHILD;

/** sg_child_push - append (@stream_idx, @reader) to the growable array *@v. Return: 0, -1 out of memory. */
static int sg_child_push(SG_CHILD **v, int *n, int *cap, int stream_idx, int reader)
{
    if (*n == *cap)
    {
        int       nc = (*cap > 0) ? 2 * *cap : 8;
        SG_CHILD *p  = realloc(*v, (size_t) nc * sizeof(SG_CHILD));
        if (p == NULL)
        {
            return -1;
        }
        *v   = p;
        *cap = nc;
    }
    (*v)[(*n)++] = (SG_CHILD) { stream_idx, reader };
    return 0;
}

static void sg_dfs_tree(const OV_MODEL *m,
                        int             current_stream,
                        int             reader_node_idx,
                        int             target_stream,
                        int             target_node,
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
    if (*nb_out_nodes >= OV_MAX_NODES || path_len >= SG_TREE_MAX_DEPTH)
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
    node->reader_name[0]  = '\0';
    node->reader_node_idx = reader_node_idx;
    node->is_target_proc  = (reader_node_idx >= 0 && reader_node_idx == target_node);
    if (reader_node_idx >= 0)
    {
        const OV_NODE *rn = &m->nodes[reader_node_idx];
        strncpy(node->reader_name, rn->name, sizeof(node->reader_name) - 1);
        node->reader_name[sizeof(node->reader_name) - 1] = '\0';
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

    /* Children: (stream written by the reader, each of its readers in S), on the heap: this recurses */
    SG_CHILD *child          = NULL;
    int       nb_child_nodes = 0, cap_child = 0;

    if (reader_node_idx >= 0)
    {
        for (int ki = OV_ADJ_B(m, out, reader_node_idx); ki < OV_ADJ_E(m, out, reader_node_idx);
             ki++)
        {
            const OV_EDGE *e      = &m->edges[m->adj_out[ki]];
            int            c_node = e->tgt_node;
            if ((e->type != OV_EDGE_PROC_WRITES_STREAM && e->type != OV_EDGE_FPS_OUTPUT_STREAM) ||
                c_node < 0 || c_node >= m->nb_nodes || m->nodes[c_node].type != OV_NODE_STREAM ||
                !sg_bget(S_words, m->nodes[c_node].index))
            {
                continue;
            }
            int c_stream = m->nodes[c_node].index, seen = 0;
            for (int k = 0; k < nb_child_nodes && !seen; k++)
            {
                seen = (child[k].stream_idx == c_stream);
            }
            if (seen)
            {
                continue; /* two edges to one stream */
            }

            /* its readers in S, each once; none: the stream alone */
            int first = nb_child_nodes, n_node = m->streams[c_stream].node_idx;
            for (int kj = (n_node >= 0) ? OV_ADJ_B(m, out, n_node) : 0;
                 n_node >= 0 && kj < OV_ADJ_E(m, out, n_node); kj++)
            {
                const OV_EDGE *e1  = &m->edges[m->adj_out[kj]];
                int            dup = !sg_edge_matches_mode_from_stream(e1, mode);
                for (int k = first; k < nb_child_nodes && !dup; k++)
                {
                    dup = (child[k].reader_node_idx == e1->tgt_node);
                }
                if (!dup &&
                    sg_child_push(&child, &nb_child_nodes, &cap_child, c_stream, e1->tgt_node) != 0)
                {
                    break;
                }
            }
            if (nb_child_nodes == first)
            {
                sg_child_push(&child, &nb_child_nodes, &cap_child, c_stream, -1);
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
        sg_dfs_tree(m, child[i].stream_idx, child[i].reader_node_idx, target_stream, target_node,
                    S_words, mode, child_prefix, (i == nb_child_nodes - 1), 0, depth + 1, path,
                    path_len + 1, out_nodes, nb_out_nodes);
    }
    free(child);
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
int sg_compute_render_tree(const OV_MODEL *m,
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
    int            target_stream = (sn->type == OV_NODE_STREAM) ? sn->index : -1;
    int            target_node   = (sn->type == OV_NODE_STREAM) ? -1 : start_node;

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
    else if (sn->type == OV_NODE_PROC || sn->type == OV_NODE_FPS)
    {
        /* Process and FPS roots: the streams reachable from that node, upstream or downstream */
        int8_t depths[OV_MAX_NODES];
        memset(depths, 127, sizeof(depths));
        sg_compute_node_depths(m, start_node, mode, depths);
        depths[start_node] = 0;

        for (int i = 0; i < m->nb_nodes; i++)
        {
            if (depths[i] != 127)
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

        for (int ki = OV_ADJ_B(m, out, n_node); ki < OV_ADJ_E(m, out, n_node); ki++)
        {
            const OV_EDGE *e1 = &m->edges[m->adj_out[ki]];
            if (sg_edge_matches_mode_from_stream(e1, mode))
            {
                int p_node = e1->tgt_node;
                for (int kj = OV_ADJ_B(m, out, p_node); kj < OV_ADJ_E(m, out, p_node); kj++)
                {
                    int c_node = m->edges[m->adj_out[kj]].tgt_node;
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
            for (int ki = OV_ADJ_B(m, out, n_node); ki < OV_ADJ_E(m, out, n_node); ki++)
            {
                const OV_EDGE *e1 = &m->edges[m->adj_out[ki]];
                if (sg_edge_matches_mode_from_stream(e1, mode))
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
                    if (!duplicate && nb_readers < OV_MAX_PROCS) /* readers are processes or FPS */
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
                    target_node, S_words, mode, "", (i == nb_root_nodes - 1), 1, 0, path, 0,
                    out_nodes, &nb_out);
    }

    return nb_out;
}
