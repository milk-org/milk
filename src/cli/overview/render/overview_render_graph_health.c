// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_graph_health.c
 * @brief   Health of each CONNECTIONS row (stream rate, reader state, missed triggers, FPS, compute time)
 *          and where a pipeline stalls.
 */

#include "overview_render_internal.h"
#include "stream_graph.h"

/** reader_fps - the FPS running process @pi, or -1. */
static int reader_fps(const OV_MODEL *m, int pi)
{
    int pn = m->procs[pi].node_idx;
    for (int k = OV_ADJ_B(m, in, pn); k < OV_ADJ_E(m, in, pn); k++)
    {
        const OV_EDGE *e = &m->edges[m->adj_in[k]];
        if (e->type == OV_EDGE_FPS_RUNS_PROC && m->nodes[e->src_node].type == OV_NODE_FPS)
        {
            return m->nodes[e->src_node].index;
        }
    }
    return -1;
}

/** reader_exec_ns - compute time of process @pi on input stream @si: hop 0 of a stream it writes. */
static int64_t reader_exec_ns(const OV_MODEL *m, int pi, int si)
{
    int pn = m->procs[pi].node_idx;
    for (int k = OV_ADJ_B(m, out, pn); k < OV_ADJ_E(m, out, pn); k++)
    {
        const OV_EDGE *e = &m->edges[m->adj_out[k]];
        if (e->type != OV_EDGE_PROC_WRITES_STREAM || m->nodes[e->tgt_node].type != OV_NODE_STREAM)
        {
            continue;
        }
        const OV_STREAM *o = &m->streams[m->nodes[e->tgt_node].index];
        OV_HOP           h;
        if (o->nb_proctrace > 0 && ov_stream_hops(m, o)[0].pid == m->procs[pi].PID &&
            ov_stream_hops(m, o)[0].inode == m->streams[si].inode && ov_trace_hop(m, o, 0, &h) == 0)
        {
            return h.compute_ns;
        }
    }
    return -1;
}

/** ov_graph_row_health - what tree row @rn shows on its right: stream rate, its reader's state and more. */
void ov_graph_row_health(const OV_MODEL *m, const SG_TREE_NODE *rn, OV_ROW_HEALTH *h)
{
    memset(h, 0, sizeof(*h));
    h->state    = "-";
    h->state_fg = OV_FG_DIM;
    snprintf(h->miss, sizeof(h->miss), "-");
    snprintf(h->exec, sizeof(h->exec), "-");
    if (rn->stream_idx < 0 || rn->stream_idx >= m->nb_streams)
    {
        return;
    }
    const OV_STREAM *s  = &m->streams[rn->stream_idx];
    double           hz = (s->update_hz > 0.1) ? s->update_hz : 0.0;
    ov_fmt_si(h->hz, sizeof(h->hz), hz);

    int pi = sg_tree_reader_proc(m, rn);
    int ni = rn->reader_node_idx;
    if (pi < 0 && ni >= 0 && ni < m->nb_nodes && m->nodes[ni].type == OV_NODE_FPS)
    {
        const OV_FPS *f = &m->fps[m->nodes[ni].index]; /* an FPS reader: its run process, if any */
        snprintf(h->fps, sizeof(h->fps), "%s", f->name);
        pi          = f->run_alive ? ov_find_proc_by_pid(m, f->runpid) : -1;
        h->state    = f->run_alive ? h->state : "OFF";
        h->state_fg = OV_FG_DIM;
    }
    if (pi < 0)
    {
        return;
    }
    const OV_PROC *p = &m->procs[pi];
    h->state         = ov_proc_status(p, &h->state_fg);
    h->stalled       = (ov_proc_state(p) == OV_ST_STAL);
    h->mismatch      = (hz > 0.1 && p->loop_hz < 0.9 * hz && p->active);
    if (p->loopcnt + (int64_t) p->triggermissed_cumul > 0)
    {
        double pct = 100.0 * (double) p->triggermissed_cumul /
                     (double) ((uint64_t) p->loopcnt + p->triggermissed_cumul);
        snprintf(h->miss, sizeof(h->miss), (pct < 10.0 && pct > 0.0) ? "%.1f%%" : "%.0f%%", pct);
    }
    ov_fmt_dt_ns(h->exec, sizeof(h->exec), reader_exec_ns(m, pi, rn->stream_idx));
    int fi = reader_fps(m, pi);
    if (fi >= 0 && h->fps[0] == '\0')
    {
        snprintf(h->fps, sizeof(h->fps), "%s", m->fps[fi].name);
    }
}

/**
 * ov_graph_stalls - for each of the @n rows: 1 in @cause where a pipeline stops (a stalled reader with no
 * stalled row above it on its path), 1 in @dim for the rows below such a stop.
 */
void ov_graph_stalls(const OV_MODEL *m, const SG_TREE_NODE *rows, int n, int8_t *cause, int8_t *dim)
{
    int stall_depth = -1; /* depth of the stalled row whose subtree we are in, -1 = none */
    for (int i = 0; i < n; i++)
    {
        OV_ROW_HEALTH h;
        if (stall_depth >= 0 && rows[i].depth <= stall_depth)
        {
            stall_depth = -1; /* left that subtree */
        }
        ov_graph_row_health(m, &rows[i], &h);
        dim[i]   = (stall_depth >= 0);
        cause[i] = (!dim[i] && h.stalled);
        if (cause[i])
        {
            stall_depth = rows[i].depth;
        }
    }
}
