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
    }
    return "Unknown";
}
