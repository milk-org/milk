/**
 * @file stream_graph.c
 * @brief BFS-based stream lineage traversal with loop detection
 *
 * Implements sg_compute_lineage() which performs breadth-first
 * traversal of the stream->proc/FPS->stream directed graph in
 * both directions.  Supports trigger-only, input-only, and full
 * (all FPS stream params) traversal modes.
 *
 * Loop detection: if the BFS downstream pass revisits the
 * starting stream node, the entry is marked is_loop=1 and
 * the cycle path is recorded in the result.
 */

#include <string.h>
#include <stdint.h>

#include "overview_data.h"
#include "stream_graph.h"

/* =========================================================
 * Internal bitset helpers (same pattern as overview_render)
 * ========================================================= */

#define SG_BITS_PER_WORD 64

#define SG_BSET_WORDS(n) \
    (((n) + SG_BITS_PER_WORD - 1) / SG_BITS_PER_WORD)

/**
 * @brief Set a bit in the stream graph adjacency matrix.
 */
static void sg_bset(uint64_t *words, int idx)
{
    words[idx / SG_BITS_PER_WORD] |=
        (UINT64_C(1) << (idx % SG_BITS_PER_WORD));
}

/**
 * @brief Get a bit from the stream graph adjacency matrix.
 */
static int sg_bget(const uint64_t *words, int idx)
{
    return (words[idx / SG_BITS_PER_WORD]
            >> (idx % SG_BITS_PER_WORD)) & 1;
}

/* =========================================================
 * BFS queue element
 * ========================================================= */

typedef struct
{
    int node;
    int depth;
} sg_bfs_item_t;

/* =========================================================
 * Edge filter predicate
 *
 * Returns 1 if the edge should be followed when
 * traversing FROM a stream node TO a proc/FPS node.
 * ========================================================= */

/**
 * sg_edge_matches_mode_from_stream - check if edge
 *     qualifies for downstream/upstream traversal
 *     from a stream node.
 * @e:    edge to test
 * @mode: traversal mode
 *
 * Return: 1 if edge should be followed.
 */
static int sg_edge_matches_mode_from_stream(
    const OV_EDGE *e,
    sg_mode_t     mode)
{
    switch (mode)
    {
    case SG_MODE_TRIGGER:
        return (e->type == OV_EDGE_STREAM_TRIGGERS_PROC
                || e->type
                   == OV_EDGE_PROC_TRIGGER_STREAM);

    case SG_MODE_INPUT:
        return (e->type == OV_EDGE_FPS_INPUT_STREAM
                || e->type
                   == OV_EDGE_STREAM_READ_BY_PROC);

    case SG_MODE_FULL:
        /* Follow all stream-related edges */
        return (e->type == OV_EDGE_STREAM_TRIGGERS_PROC
                || e->type
                   == OV_EDGE_PROC_TRIGGER_STREAM
                || e->type
                   == OV_EDGE_FPS_INPUT_STREAM
                || e->type
                   == OV_EDGE_FPS_OUTPUT_STREAM
                || e->type
                   == OV_EDGE_STREAM_READ_BY_PROC
                || e->type
                   == OV_EDGE_PROC_WRITES_STREAM);

    case SG_MODE_FPS:
        /* FPS-centric: only FPS<->stream edges */
        return (e->type == OV_EDGE_FPS_INPUT_STREAM
                || e->type
                   == OV_EDGE_FPS_OUTPUT_STREAM);
    } /* switch mode */

    return 0;
}


/* =========================================================
 * Downstream BFS (descendants)
 *
 * Follow forward edges: stream -> proc/FPS -> stream.
 * From a stream node, find procs/FPS that consume it.
 * From those, find streams they produce.
 * Each stream->proc->stream hop increments depth.
 * ========================================================= */

static void sg_bfs_downstream(
    const OV_MODEL *m,
    int            start_node,
    int            root_stream_idx,
    sg_mode_t      mode,
    SG_LINEAGE     *out)
{
    uint64_t visited[SG_BSET_WORDS(OV_MAX_NODES)];
    memset(visited, 0, sizeof(visited));
    sg_bset(visited, start_node);

    sg_bfs_item_t queue[OV_MAX_NODES];
    int qhead = 0;
    int qtail = 0;

    queue[qtail].node  = start_node;
    queue[qtail].depth = 0;
    qtail++;

    while (qhead < qtail)
    {
        sg_bfs_item_t cur = queue[qhead++];

        if (cur.depth >= SG_MAX_DEPTH)
        {
            continue;
        }

        const OV_NODE *cn = &m->nodes[cur.node];

        if (cn->type == OV_NODE_STREAM)
        {
            /* From stream, find proc/FPS consumers
             * via forward edges (src=this stream) */
            for (int ei = 0;
                 ei < m->nb_edges; ei++)
            {
                const OV_EDGE *e =
                    &m->edges[ei];
                if (e->src_node != cur.node)
                {
                    continue;
                }

                if (!sg_edge_matches_mode_from_stream(
                        e, mode))
                {
                    continue;
                }

                int next = e->tgt_node;
                if (next < 0
                    || next >= m->nb_nodes)
                {
                    continue;
                }
                if (sg_bget(visited, next))
                {
                    continue;
                }
                sg_bset(visited, next);
                queue[qtail].node  = next;
                queue[qtail].depth = cur.depth;
                qtail++;
            }
        }
        else /* PROC or FPS intermediary */
        {
            /* From proc/FPS, find output streams
             * via forward edges (src=this proc/FPS,
             * tgt=stream) */
            for (int ei = 0;
                 ei < m->nb_edges; ei++)
            {
                const OV_EDGE *e =
                    &m->edges[ei];
                if (e->src_node != cur.node)
                {
                    continue;
                }
                int next = e->tgt_node;
                if (next < 0
                    || next >= m->nb_nodes)
                {
                    continue;
                }

                const OV_NODE *nn =
                    &m->nodes[next];
                if (nn->type != OV_NODE_STREAM)
                {
                    continue;
                }

                /* Loop detection: if this stream
                 * is the root, record cycle */
                int is_loop = 0;
                if (nn->index == root_stream_idx
                    && cur.depth > 0)
                {
                    is_loop = 1;
                    out->has_loop = 1;
                }

                if (sg_bget(visited, next)
                    && !is_loop)
                {
                    continue;
                }
                if (!is_loop)
                {
                    sg_bset(visited, next);
                }

                int d = cur.depth + 1;
                if (out->nb_descendants
                    < SG_MAX_LINEAGE
                    && nn->index >= 0
                    && nn->index
                       < m->nb_streams)
                {
                    SG_LINEAGE_ENTRY *le =
                        &out->descendants[
                            out->nb_descendants];
                    le->stream_idx = nn->index;
                    le->depth      = d;
                    le->is_loop    = is_loop;
                    strncpy(le->via_name,
                            cn->name, 39);
                    le->via_name[39] = '\0';
                    out->nb_descendants++;
                }

                if (!is_loop)
                {
                    queue[qtail].node  = next;
                    queue[qtail].depth = d;
                    qtail++;
                }
            }
        }
    } /* downstream BFS */
}


/* =========================================================
 * Upstream BFS (ancestors)
 *
 * Follow edges backwards: stream <- proc/FPS <- stream.
 * From a stream node, find procs/FPS that write to it.
 * From those, find streams that feed them.
 * ========================================================= */

static void sg_bfs_upstream(
    const OV_MODEL *m,
    int            start_node,
    int            root_stream_idx,
    sg_mode_t      mode,
    SG_LINEAGE     *out)
{
    uint64_t visited[SG_BSET_WORDS(OV_MAX_NODES)];
    memset(visited, 0, sizeof(visited));
    sg_bset(visited, start_node);

    sg_bfs_item_t queue[OV_MAX_NODES];
    int qhead = 0;
    int qtail = 0;

    queue[qtail].node  = start_node;
    queue[qtail].depth = 0;
    qtail++;

    while (qhead < qtail)
    {
        sg_bfs_item_t cur = queue[qhead++];

        if (cur.depth >= SG_MAX_DEPTH)
        {
            continue;
        }

        const OV_NODE *cn = &m->nodes[cur.node];

        if (cn->type == OV_NODE_STREAM)
        {
            /* From stream, find proc/FPS that
             * produce it (edges where tgt=this) */
            for (int ei = 0;
                 ei < m->nb_edges; ei++)
            {
                const OV_EDGE *e =
                    &m->edges[ei];
                if (e->tgt_node != cur.node)
                {
                    continue;
                }
                int next = e->src_node;
                if (next < 0
                    || next >= m->nb_nodes)
                {
                    continue;
                }
                if (sg_bget(visited, next))
                {
                    continue;
                }
                sg_bset(visited, next);
                queue[qtail].node  = next;
                queue[qtail].depth = cur.depth;
                qtail++;
            }
        }
        else /* PROC or FPS intermediary */
        {
            /* From proc/FPS, find input streams
             * (edges where tgt=this proc/FPS,
             *  src=stream) */
            for (int ei = 0;
                 ei < m->nb_edges; ei++)
            {
                const OV_EDGE *e =
                    &m->edges[ei];
                if (e->tgt_node != cur.node)
                {
                    continue;
                }

                if (!sg_edge_matches_mode_from_stream(
                        e, mode))
                {
                    continue;
                }

                int next = e->src_node;
                if (next < 0
                    || next >= m->nb_nodes)
                {
                    continue;
                }

                const OV_NODE *nn =
                    &m->nodes[next];
                if (nn->type != OV_NODE_STREAM)
                {
                    continue;
                }

                /* Loop detection */
                int is_loop = 0;
                if (nn->index == root_stream_idx
                    && cur.depth > 0)
                {
                    is_loop = 1;
                    out->has_loop = 1;
                }

                if (sg_bget(visited, next)
                    && !is_loop)
                {
                    continue;
                }
                if (!is_loop)
                {
                    sg_bset(visited, next);
                }

                int d = cur.depth + 1;
                if (out->nb_ancestors
                    < SG_MAX_LINEAGE
                    && nn->index >= 0
                    && nn->index
                       < m->nb_streams)
                {
                    SG_LINEAGE_ENTRY *le =
                        &out->ancestors[
                            out->nb_ancestors];
                    le->stream_idx = nn->index;
                    le->depth      = d;
                    le->is_loop    = is_loop;
                    strncpy(le->via_name,
                            cn->name, 39);
                    le->via_name[39] = '\0';
                    out->nb_ancestors++;
                }

                if (!is_loop)
                {
                    queue[qtail].node  = next;
                    queue[qtail].depth = d;
                    qtail++;
                }
            }
        }
    } /* upstream BFS */
}


/* =========================================================
 * Public API
 * ========================================================= */

void sg_compute_lineage(
    const OV_MODEL *m,
    int            stream_idx,
    sg_mode_t      mode,
    SG_LINEAGE     *out)
{
    memset(out, 0, sizeof(*out));

    if (stream_idx < 0
        || stream_idx >= m->nb_streams)
    {
        return;
    }

    int start_node =
        m->streams[stream_idx].node_idx;
    if (start_node < 0)
    {
        return;
    }

    sg_bfs_downstream(
        m, start_node, stream_idx, mode, out);
    sg_bfs_upstream(
        m, start_node, stream_idx, mode, out);

    /* Build cycle path from descendant entries
     * that close a loop */
    if (out->has_loop)
    {
        out->cycle_len = 0;
        /* Start with root stream */
        if (out->cycle_len < SG_MAX_DEPTH)
        {
            out->cycle_path[out->cycle_len++] =
                stream_idx;
        }
        /* Walk descendants until we find the
         * loop-closing entry */
        for (int i = 0;
             i < out->nb_descendants; i++)
        {
            if (out->cycle_len < SG_MAX_DEPTH)
            {
                out->cycle_path[
                    out->cycle_len++] =
                    out->descendants[i]
                        .stream_idx;
            }
            if (out->descendants[i].is_loop)
            {
                break;
            }
        }
    }
}


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

void sg_compute_node_depths(
    const OV_MODEL *m,
    int            start_node,
    sg_mode_t      mode,
    int8_t         *node_depths)
{
    for (int i = 0; i < OV_MAX_NODES; i++) {
        node_depths[i] = 127;
    }
    if (start_node < 0 || start_node >= m->nb_nodes) return;
    node_depths[start_node] = 0;

    ov_node_type_t start_type = m->nodes[start_node].type;

    /* Downstream */
    uint64_t visited[SG_BSET_WORDS(OV_MAX_NODES)];
    memset(visited, 0, sizeof(visited));
    sg_bset(visited, start_node);

    sg_bfs_item_t queue[OV_MAX_NODES];
    int qhead = 0, qtail = 0;
    queue[qtail].node = start_node;
    queue[qtail].depth = 0;
    qtail++;

    while (qhead < qtail)
    {
        sg_bfs_item_t cur = queue[qhead++];

        if (cur.node != start_node)
        {
            int d = cur.depth;
            if (m->nodes[cur.node].type != start_type) d++;
            if (d > 127) d = 127;
            if (node_depths[cur.node] == 127)
                node_depths[cur.node] = (int8_t)d;
        }

        if (cur.depth >= SG_MAX_DEPTH) continue;

        const OV_NODE *cn = &m->nodes[cur.node];

        for (int ei = 0; ei < m->nb_edges; ei++)
        {
            const OV_EDGE *e = &m->edges[ei];
            if (e->src_node != cur.node) continue;

            if (cn->type == OV_NODE_STREAM) {
                if (!sg_edge_matches_mode_from_stream(
                        e, mode))
                    continue;
            } else if (mode == SG_MODE_FPS) {
                /* FPS mode: only FPS<->stream edges */
                if (!sg_edge_matches_mode_from_stream(
                        e, mode))
                    continue;
            } else {
                /* From FPS/PROC: follow edges to streams
                 * and FPS_RUNS_PROC edges (FPS->proc) */
                int tgt_type =
                    m->nodes[e->tgt_node].type;
                if (tgt_type != OV_NODE_STREAM
                    && e->type != OV_EDGE_FPS_RUNS_PROC)
                    continue;
            }

            int next = e->tgt_node;
            if (next < 0 || next >= m->nb_nodes) continue;
            if (sg_bget(visited, next)) continue;

            sg_bset(visited, next);
            int d = cur.depth;
            if (m->nodes[next].type == start_type)
                d++;

            queue[qtail].node = next;
            queue[qtail].depth = d;
            qtail++;
        }
    }

    /* Upstream */
    memset(visited, 0, sizeof(visited));
    sg_bset(visited, start_node);
    qhead = 0; qtail = 0;
    queue[qtail].node = start_node;
    queue[qtail].depth = 0;
    qtail++;

    while (qhead < qtail)
    {
        sg_bfs_item_t cur = queue[qhead++];

        if (cur.node != start_node)
        {
            int d = cur.depth;
            if (m->nodes[cur.node].type != start_type) d++;
            if (d > 127) d = 127;
            if (node_depths[cur.node] == 127)
                node_depths[cur.node] = (int8_t)(-d);
        }

        if (cur.depth >= SG_MAX_DEPTH) continue;

        const OV_NODE *cn = &m->nodes[cur.node];

        for (int ei = 0; ei < m->nb_edges; ei++)
        {
            const OV_EDGE *e = &m->edges[ei];
            if (e->tgt_node != cur.node) continue;

            int next = e->src_node;
            if (next < 0 || next >= m->nb_nodes) continue;
            
            if (mode == SG_MODE_FPS) {
                /* FPS mode: restrict to FPS<->stream
                 * edges for both stream and non-stream
                 * nodes */
                if (!sg_edge_matches_mode_from_stream(
                        e, mode))
                    continue;
            } else if (cn->type != OV_NODE_STREAM) {
                /* Going upstream from PROC/FPS:
                 * accept stream-related edges and
                 * FPS_RUNS_PROC (proc<-FPS).
                 * Stream nodes: no filter (accept
                 * all reverse edges). */
                if (!sg_edge_matches_mode_from_stream(
                        e, mode)
                    && e->type
                       != OV_EDGE_FPS_RUNS_PROC)
                    continue;
            }

            if (sg_bget(visited, next)) continue;

            sg_bset(visited, next);
            int d = cur.depth;
            if (m->nodes[next].type == start_type)
                d++;

            queue[qtail].node = next;
            queue[qtail].depth = d;
            qtail++;
        }
    }
}

int sg_compute_render_nodes(
    const OV_MODEL *m,
    int            start_node,
    sg_mode_t      mode,
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
    int nb_nodes = 0;

    for (int i = 0; i < m->nb_nodes; ++i)
    {
        if (depths[i] != 127)
        {
            temp_nodes[nb_nodes].node_idx = i;
            temp_nodes[nb_nodes].depth = depths[i];
            int type_order = m->nodes[i].type; // OV_NODE_STREAM=0, OV_NODE_FPS=1, OV_NODE_PROC=2
            int reverse_type_order = 0;
            ov_node_type_t start_type = m->nodes[start_node].type;
            
            if (start_type == OV_NODE_STREAM && depths[i] > 0) reverse_type_order = 1;
            if (start_type != OV_NODE_STREAM && depths[i] < 0) reverse_type_order = 1;
            
            if (reverse_type_order)
            {
                type_order = 2 - type_order; // Invert: STREAM(2), FPS(1), PROC(0)
            }
            
            temp_nodes[nb_nodes].order = depths[i] * 10 + type_order;
            temp_nodes[nb_nodes].type = m->nodes[i].type;
            
            /* Detect loops using BFS lineage if needed, but for simplicity,
             * we can just mark is_loop = 0. Real loop detection requires
             * lineage structures. We'll leave it 0 for now. */
            temp_nodes[nb_nodes].is_loop = 0;
            
            strncpy(temp_nodes[nb_nodes].name, m->nodes[i].name, sizeof(temp_nodes[nb_nodes].name) - 1);
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
                temp_nodes[j] = temp_nodes[j + 1];
                temp_nodes[j + 1] = tmp;
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
static void sg_dfs_tree(
    const OV_MODEL *m,
    int current_stream,
    int target_stream,
    int target_proc,
    const uint64_t *S_words,
    sg_mode_t mode,
    const char *prefix,
    int is_last,
    int is_root,
    int depth,
    int *path,
    int path_len,
    SG_TREE_NODE *out_nodes,
    int *nb_out_nodes)
{
    if (*nb_out_nodes >= OV_MAX_NODES) return;

    /* Cycle detection */
    int is_cycle = 0;
    for (int i = 0; i < path_len; i++) {
        if (path[i] == current_stream) {
            is_cycle = 1;
            break;
        }
    }

    SG_TREE_NODE *node = &out_nodes[*nb_out_nodes];
    node->stream_idx = current_stream;
    node->is_target = (current_stream == target_stream);
    node->depth = depth;
    strncpy(node->name, m->streams[current_stream].name, sizeof(node->name)-1);
    node->name[sizeof(node->name)-1] = '\0';
    
    /* Find writer proc */
    node->writer_name[0] = '\0';
    node->is_target_proc = 0;
    pid_t wpid = m->streams[current_stream].write_pid;
    if (wpid > 0) {
        int pidx = ov_find_proc_by_pid(m, wpid);
        if (pidx >= 0) {
            strncpy(node->writer_name, m->procs[pidx].name, sizeof(node->writer_name)-1);
            node->writer_name[sizeof(node->writer_name)-1] = '\0';
            if (pidx == target_proc) {
                node->is_target_proc = 1;
            }
        }
    }

    /* Build prefix */
    if (is_root) {
        node->tree_prefix[0] = '\0';
    } else {
        snprintf(node->tree_prefix, sizeof(node->tree_prefix), "%s%s", 
                 prefix, is_last ? "\xe2\x94\x94\xe2\x94\x80\xe2\x94\x80 " : "\xe2\x94\x9c\xe2\x94\x80\xe2\x94\x80 "); /* └──  and ├──  */
    }
    
    if (is_cycle) {
        /* Append cycle indicator to name */
        strncat(node->name, " (loop)", sizeof(node->name) - strlen(node->name) - 1);
        (*nb_out_nodes)++;
        return;
    }

    (*nb_out_nodes)++;

    path[path_len] = current_stream;

    /* Find children in S */
    int children[OV_MAX_STREAMS];
    int nb_children = 0;

    int n_node = m->streams[current_stream].node_idx;
    if (n_node >= 0) {
        for (int ei = 0; ei < m->nb_edges; ei++) {
            const OV_EDGE *e1 = &m->edges[ei];
            if (e1->src_node == n_node && sg_edge_matches_mode_from_stream(e1, mode)) {
                int p_node = e1->tgt_node;
                for (int ej = 0; ej < m->nb_edges; ej++) {
                    const OV_EDGE *e2 = &m->edges[ej];
                    if (e2->src_node == p_node) {
                        int c_node = e2->tgt_node;
                        if (c_node >= 0 && c_node < m->nb_nodes && m->nodes[c_node].type == OV_NODE_STREAM) {
                            int c_stream = m->nodes[c_node].index;
                            if (sg_bget(S_words, c_stream)) {
                                int duplicate = 0;
                                for (int k=0; k<nb_children; k++) if (children[k] == c_stream) duplicate = 1;
                                if (!duplicate) children[nb_children++] = c_stream;
                            }
                        }
                    }
                }
            }
        }
    }

    /* Recurse */
    char child_prefix[128];
    if (is_root) {
        child_prefix[0] = '\0';
    } else {
        snprintf(child_prefix, sizeof(child_prefix), "%s%s", 
                 prefix, is_last ? "    " : "\xe2\x94\x82   "); /* "    " and "│   " */
    }

    for (int i = 0; i < nb_children; i++) {
        sg_dfs_tree(m, children[i], target_stream, target_proc, S_words, mode, child_prefix, (i == nb_children - 1), 0, depth + 1, path, path_len + 1, out_nodes, nb_out_nodes);
    }
}

int sg_compute_render_tree(
    const OV_MODEL *m,
    int            start_node,
    sg_mode_t      mode,
    SG_TREE_NODE   *out_nodes)
{
    int nb_out = 0;
    if (start_node < 0 || start_node >= m->nb_nodes) return 0;
    
    const OV_NODE *sn = &m->nodes[start_node];
    int target_stream = -1;
    int target_proc = -1;
    
    if (sn->type == OV_NODE_STREAM) {
        target_stream = sn->index;
    } else if (sn->type == OV_NODE_PROC) {
        target_proc = sn->index;
    }

    uint64_t S_words[SG_BSET_WORDS(OV_MAX_STREAMS)];
    memset(S_words, 0, sizeof(S_words));

    if (target_stream != -1) {
        SG_LINEAGE lin;
        memset(&lin, 0, sizeof(lin));
        sg_compute_lineage(m, target_stream, mode, &lin);

        sg_bset(S_words, target_stream);
        for (int i=0; i<lin.nb_ancestors; i++) sg_bset(S_words, lin.ancestors[i].stream_idx);
        for (int i=0; i<lin.nb_descendants; i++) sg_bset(S_words, lin.descendants[i].stream_idx);
    } else if (target_proc != -1) {
        int8_t depths[OV_MAX_NODES];
        memset(depths, 127, sizeof(depths));
        sg_compute_node_depths(m, start_node, mode, depths);
        depths[start_node] = 0;
        
        for (int i=0; i<m->nb_nodes; i++) {
            if (depths[i] < 127 || depths[i] > -127) {
                if (m->nodes[i].type == OV_NODE_STREAM) {
                    sg_bset(S_words, m->nodes[i].index);
                }
            }
        }
    } else {
        return 0;
    }

    /* Find parents for everyone in S */
    int has_parent[OV_MAX_STREAMS];
    memset(has_parent, 0, sizeof(has_parent));

    for (int i=0; i<m->nb_streams; i++) {
        if (!sg_bget(S_words, i)) continue;
        
        int n_node = m->streams[i].node_idx;
        if (n_node < 0) continue;
        
        for (int ei = 0; ei < m->nb_edges; ei++) {
            const OV_EDGE *e1 = &m->edges[ei];
            if (e1->src_node == n_node && sg_edge_matches_mode_from_stream(e1, mode)) {
                int p_node = e1->tgt_node;
                for (int ej = 0; ej < m->nb_edges; ej++) {
                    const OV_EDGE *e2 = &m->edges[ej];
                    if (e2->src_node == p_node) {
                        int c_node = e2->tgt_node;
                        if (c_node >= 0 && c_node < m->nb_nodes && m->nodes[c_node].type == OV_NODE_STREAM) {
                            int c_stream = m->nodes[c_node].index;
                            if (sg_bget(S_words, c_stream)) {
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
    for (int i=0; i<m->nb_streams; i++) {
        if (sg_bget(S_words, i) && !has_parent[i]) {
            roots[nb_roots++] = i;
        }
    }

    if (nb_roots == 0) {
        /* Cycle graph with no absolute root. Use first available as root. */
        for (int i=0; i<m->nb_streams; i++) {
            if (sg_bget(S_words, i)) {
                roots[nb_roots++] = i;
                break;
            }
        }
    }

    int path[OV_MAX_STREAMS];
    for (int i=0; i<nb_roots; i++) {
        sg_dfs_tree(m, roots[i], target_stream, target_proc, S_words, mode, "", (i == nb_roots - 1), 1, 0, path, 0, out_nodes, &nb_out);
    }

    return nb_out;
}
