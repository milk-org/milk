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

static void sg_bset(uint64_t *words, int idx)
{
    words[idx / SG_BITS_PER_WORD] |=
        (UINT64_C(1) << (idx % SG_BITS_PER_WORD));
}

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
    sg_mode_t      mode)
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
    int             start_node,
    int             root_stream_idx,
    sg_mode_t       mode,
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
    int             start_node,
    int             root_stream_idx,
    sg_mode_t       mode,
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
    int             stream_idx,
    sg_mode_t       mode,
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
    int             start_node,
    sg_mode_t       mode,
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

        if (cur.node != start_node && cur.depth > 0)
        {
            int d = cur.depth;
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

        if (cur.node != start_node && cur.depth > 0)
        {
            int d = cur.depth;
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
