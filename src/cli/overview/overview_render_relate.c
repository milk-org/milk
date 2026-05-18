#include "overview_render_internal.h"
int ov_filter_build(
    const char  *pattern,
    const char **names,
    int          count,
    int         *out,
    int          max_out)
{
    if (pattern[0] == '\0')
    {
        /* No filter — all items match */
        int n = count < max_out ? count : max_out;
        for (int i = 0; i < n; i++)
        {
            out[i] = i;
        }
        return n;
    }

    regex_t re;
    if (regcomp(&re, pattern,
                REG_EXTENDED | REG_NOSUB
                | REG_ICASE) != 0)
    {
        /* Invalid regex — show all */
        int n = count < max_out ? count : max_out;
        for (int i = 0; i < n; i++)
        {
            out[i] = i;
        }
        return n;
    }

    int n = 0;
    for (int i = 0; i < count && n < max_out; i++)
    {
        if (regexec(&re, names[i], 0, NULL, 0) == 0)
        {
            out[n++] = i;
        }
    }
    regfree(&re);
    return n;
}

/* =========================================================
 * Cross-panel relation highlight
 * ========================================================= */

/*
 * Bitset helpers and OV_RELATED defined in
 * overview_render_internal.h
 */

void bset(uint64_t *words, int idx)
{
    words[idx / BITS_PER_WORD] |= (UINT64_C(1) << (idx % BITS_PER_WORD));
}

int bget(const uint64_t *words, int idx)
{
    return (words[idx / BITS_PER_WORD] >> (idx % BITS_PER_WORD)) & 1;
}
void ov_compute_related(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m,
    OV_RELATED      *out)
{
    memset(out, 0, sizeof(*out));
    /* fps_param_mask initialised to 0 by memset — no matches yet */

    ov_focus_t focus;
    int sel_stream_idx;
    int sel_proc_idx;
    int sel_fps_idx;

    if (lay->mouse_hover && lay->hover_idx >= 0 && lay->hover_view != -1)
    {
        focus = lay->hover_view;
        sel_stream_idx = (focus == OV_FOCUS_STREAMS) ? lay->hover_idx : -1;
        sel_proc_idx   = (focus == OV_FOCUS_PROCS)   ? lay->hover_idx : -1;
        sel_fps_idx    = (focus == OV_FOCUS_FPS)     ? lay->hover_idx : -1;
    }
    else
    {
        focus = lay->freeze
                             ? lay->freeze_focus
                             : lay->focus;
        sel_stream_idx = lay->freeze
                             ? lay->freeze_sel_stream
                             : lay->sel_stream;
        sel_proc_idx   = lay->freeze
                             ? lay->freeze_sel_proc
                             : lay->sel_proc;
        sel_fps_idx    = lay->freeze
                             ? lay->freeze_sel_fps
                             : lay->sel_fps;
    }

    /* Determine the graph node index of the selected item */
    int sel_node = -1;
    if (focus == OV_FOCUS_STREAMS && sel_stream_idx >= 0)
    {
        int model_idx = sel_stream_idx;
        if (lay->filter_stream[0] != '\0')
        {
            const char *names[OV_MAX_STREAMS];
            for (int i = 0; i < m->nb_streams; i++) names[i] = m->streams[i].name;
            int fidx[OV_MAX_STREAMS];
            int n = ov_filter_build(lay->filter_stream, names, m->nb_streams, fidx, OV_MAX_STREAMS);
            model_idx = (sel_stream_idx < n) ? fidx[sel_stream_idx] : -1;
        }
        if (model_idx >= 0 && model_idx < m->nb_streams)
        {
            sel_node = m->streams[model_idx].node_idx;
        }
    }
    else if (focus == OV_FOCUS_FPS && sel_fps_idx >= 0)
    {
        int model_idx = sel_fps_idx;
        if (lay->filter_fps[0] != '\0')
        {
            const char *names[OV_MAX_FPS];
            for (int i = 0; i < m->nb_fps; i++) names[i] = m->fps[i].name;
            int fidx[OV_MAX_FPS];
            int n = ov_filter_build(lay->filter_fps, names, m->nb_fps, fidx, OV_MAX_FPS);
            model_idx = (sel_fps_idx < n) ? fidx[sel_fps_idx] : -1;
        }
        if (model_idx >= 0 && model_idx < m->nb_fps)
        {
            sel_node = m->fps[model_idx].node_idx;
        }
    }
    else if (focus == OV_FOCUS_PROCS && sel_proc_idx >= 0)
    {
        int model_idx = sel_proc_idx;
        if (lay->filter_proc[0] != '\0')
        {
            const char *names[OV_MAX_PROCS];
            for (int i = 0; i < m->nb_procs; i++) names[i] = m->procs[i].name;
            int fidx[OV_MAX_PROCS];
            int n = ov_filter_build(lay->filter_proc, names, m->nb_procs, fidx, OV_MAX_PROCS);
            model_idx = (sel_proc_idx < n) ? fidx[sel_proc_idx] : -1;
        }
        if (model_idx >= 0 && model_idx < m->nb_procs)
        {
            sel_node = m->procs[model_idx].node_idx;
            out->sel_pid = m->procs[model_idx].PID;
        }
    }

    if (sel_node < 0)
    {
        return;
    }

    /* Walk all edges; mark neighbours of sel_node */
    for (int ei = 0; ei < m->nb_edges; ei++)
    {
        const OV_EDGE *e = &m->edges[ei];
        int other    = -1;
        int is_write = 0; /* 1 = proc writes stream */
        /* For FPS edges: which end is the stream node? */
        int fps_is_src = 0;

        if (e->src_node == sel_node)
        {
            other      = e->tgt_node;
            is_write   = (e->type == OV_EDGE_PROC_WRITES_STREAM) || (e->type == OV_EDGE_FPS_OUTPUT_STREAM);
            fps_is_src = 0; /* FPS is tgt when stream→FPS */
        }
        else if (e->tgt_node == sel_node)
        {
            other      = e->src_node;
            is_write   = (e->type == OV_EDGE_PROC_WRITES_STREAM) || (e->type == OV_EDGE_FPS_OUTPUT_STREAM);
            fps_is_src = 1; /* FPS is src when FPS→stream */
        }

        if (other < 0 || other >= m->nb_nodes)
        {
            continue;
        }

        const OV_NODE *n = &m->nodes[other];
        if (n->type == OV_NODE_STREAM && n->index >= 0
            && n->index < m->nb_streams)
        {
            bset(out->streams, n->index);
            if (is_write)
            {
                bset(out->stream_written, n->index);
            }
        }
        else if (n->type == OV_NODE_FPS && n->index >= 0
                 && n->index < m->nb_fps)
        {
            int fi = n->index;
            bset(out->fps, fi);
            if (is_write)
            {
                bset(out->fps_writes, fi);
            }

            /* Find all stream params of this FPS that match sel_node.
             * Only meaningful when the selection is a stream.
             * OR all matching indices into the bitmask. */
            if (focus == OV_FOCUS_STREAMS
                && sel_stream_idx >= 0
                && sel_stream_idx < m->nb_streams)
            {
                const char *sname =
                    m->streams[sel_stream_idx].name;
                const OV_FPS *f = &m->fps[fi];
                for (int sp = 0; sp < f->nb_stream_params; sp++)
                {
                    if (strcmp(f->stream_param_value[sp],
                               sname) == 0)
                    {
                        out->fps_param_mask[fi] |=
                            (UINT32_C(1) << sp);
                    }
                } /* for sp */
            } /* if FOCUS_STREAMS */

            (void) fps_is_src; /* suppress unused-var warning */
        }
        else if (n->type == OV_NODE_PROC && n->index >= 0
                 && n->index < m->nb_procs)
        {
            bset(out->procs, n->index);
            if (is_write)
            {
                bset(out->proc_writes, n->index);
            }
        }
    } /* for ei */
}
