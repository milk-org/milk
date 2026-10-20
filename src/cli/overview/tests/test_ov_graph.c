// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkctrl_unit_graph: the typed link graph built from synthetic models (no shared memory). */

#include "overview_data_internal.h"

#include <stdlib.h>

static int failures = 0;

#define CHECK(c)                                                         \
    do                                                                   \
    {                                                                    \
        if (!(c))                                                        \
        {                                                                \
            fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #c); \
            failures++;                                                  \
        }                                                                \
    } while (0)

/** stream - add stream @name with inode @ino to @m. Return: its index. */
static int stream(OV_MODEL *m, const char *name, ino_t ino)
{
    OV_STREAM *s = &m->streams[m->nb_streams];
    snprintf(s->name, sizeof(s->name), "%s", name);
    s->valid = 1;
    s->inode = ino;
    return m->nb_streams++;
}

/** proc - add live process @name with @pid to @m. Return: its index. */
static int proc(OV_MODEL *m, const char *name, pid_t pid)
{
    OV_PROC *p = &m->procs[m->nb_procs];
    snprintf(p->name, sizeof(p->name), "%s", name);
    p->valid  = 1;
    p->active = 1;
    p->PID    = pid;
    return m->nb_procs++;
}

/** edge - the edge @type from node @src to node @tgt, or NULL. */
static const OV_EDGE *edge(const OV_MODEL *m, int src, int tgt, ov_edge_type_t type)
{
    for (int i = 0; i < m->nb_edges; i++)
    {
        const OV_EDGE *e = &m->edges[i];
        if (e->src_node == src && e->tgt_node == tgt && e->type == type)
        {
            return e;
        }
    }
    return NULL;
}

/* One edge per relation, carrying every way it was seen: proctrace, semaphores, processinfo trigger */
static void test_edge_evidence(void)
{
    static OV_MODEL m;
    memset(&m, 0, sizeof(m));
    int cam = stream(&m, "cam", 11);
    int wfs = stream(&m, "wfs", 12);
    int pw  = proc(&m, "pwfs", 500);
    proc(&m, "pdead", 500); /* a stale entry sharing the PID: edges go to the live one */
    m.procs[1].active     = 0;
    m.procs[1].createtime = 99.0;

    OV_STREAM *s                = &m.streams[wfs];
    s->nb_proctrace             = 1;
    s->proctrace_pid[0]         = 500; /* written by pwfs, triggered by cam */
    s->proctrace_inode[0]       = 11;
    s->nb_write_pids            = 1;
    s->write_pids[0]            = 500;
    m.streams[cam].nb_read_pids = 1;
    m.streams[cam].read_pids[0] = 500;
    snprintf(m.procs[pw].trigstreamname, sizeof(m.procs[pw].trigstreamname), "cam");

    ov_build_graph(&m);
    int ncam = m.streams[cam].node_idx, nwfs = m.streams[wfs].node_idx, npw = m.procs[pw].node_idx;
    const OV_EDGE *w = edge(&m, npw, nwfs, OV_EDGE_PROC_WRITES_STREAM);
    const OV_EDGE *t = edge(&m, ncam, npw, OV_EDGE_STREAM_TRIGGERS_PROC);
    const OV_EDGE *r = edge(&m, ncam, npw, OV_EDGE_STREAM_READ_BY_PROC);
    CHECK(w != NULL && w->evidence == (OV_EV_PROCTRACE | OV_EV_SEM_WRITE));
    CHECK(t != NULL && t->evidence == (OV_EV_PROCTRACE | OV_EV_PINFO_TRIGGER));
    CHECK(r != NULL && r->evidence == OV_EV_SEM_READ);
    CHECK(m.nb_edges == 3);
}

/* Adjacency lists: every edge once in its source's out-list and its target's in-list, in edge order */
static void test_adjacency(void)
{
    static OV_MODEL m;
    memset(&m, 0, sizeof(m));
    m.nb_nodes = 300;
    unsigned x = 12345;
    for (int i = 0; i < 2000; i++)
    {
        x     = x * 1103515245U + 12345U;
        int a = (int) ((x >> 8) % 300), b = (int) ((x >> 20) % 300);
        ov_add_edge(&m, a, b, (ov_edge_type_t) (i % OV_EDGE_TYPE_COUNT), OV_EV_PROCTRACE);
    }
    ov_model_adjacency(&m);
    int seen_out = 0, seen_in = 0;
    for (int v = 0; v < m.nb_nodes; v++)
    {
        for (int k = OV_ADJ_B(&m, out, v); k < OV_ADJ_E(&m, out, v); k++)
        {
            CHECK(m.edges[m.adj_out[k]].src_node == v);
            CHECK(k == OV_ADJ_B(&m, out, v) || m.adj_out[k] > m.adj_out[k - 1]);
            seen_out++;
        }
        for (int k = OV_ADJ_B(&m, in, v); k < OV_ADJ_E(&m, in, v); k++)
        {
            CHECK(m.edges[m.adj_in[k]].tgt_node == v);
            seen_in++;
        }
    }
    CHECK(seen_out == m.nb_edges && seen_in == m.nb_edges);
    CHECK(OV_ADJ_B(&m, out, -1) == OV_ADJ_E(&m, out, -1)); /* an invalid node has no edges */
}

/* Trigger-chain timing: compute only for processes measuring timing, hand-off, lag, end to end (links-13) */
static void test_trace_timing(void)
{
    static OV_MODEL m;
    memset(&m, 0, sizeof(m));
    int a                                        = stream(&m, "a", 21);
    int b                                        = stream(&m, "b", 22);
    m.streams[a].cnt0                            = 100;
    OV_STREAM *s                                 = &m.streams[b];
    s->nb_proctrace                              = 2;
    s->proctrace_pid[0]                          = 600; /* wrote b, triggered by a */
    s->proctrace_inode[0]                        = 21;
    s->proctrace_cnt0[0]                         = 97;
    s->proctrace_tstart_ns[0]                    = 1000000;
    s->proctrace_tupdate_ns[0]                   = 1250000;
    s->proctrace_pid[1]                          = 601; /* wrote a */
    s->proctrace_tstart_ns[1]                    = 500000;
    s->proctrace_tupdate_ns[1]                   = 900000;
    m.procs[proc(&m, "p600", 600)].MeasureTiming = 1;
    proc(&m, "p601", 601);
    ov_model_index(&m);

    OV_HOP h;
    CHECK(ov_trace_hop(&m, s, 0, &h) == 0);
    CHECK(h.compute_ns == 250000 && h.handoff_ns == 100000 && h.lag == 3);
    CHECK(ov_trace_hop(&m, s, 1, &h) == 0);
    CHECK(h.compute_ns == -1 && h.handoff_ns == -1 &&
          h.lag == -1); /* no timing, no upstream, no inode */
    CHECK(ov_trace_hop(&m, s, 2, &h) == -1);
    CHECK(ov_trace_end_to_end_ns(s) == 750000);
}

int main(void)
{
    test_edge_evidence();
    test_adjacency();
    test_trace_timing();
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}
