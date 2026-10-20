// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkctrl_unit_graph: the typed link graph built from synthetic models (no shared memory). */

#include "overview_data_internal.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include "overview_data_loops.h"

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

    OV_STREAM *s = &m.streams[wfs];
    *ov_stream_add_hop(&m, s) =
        (OV_TRACE_HOP) { .pid = 500, .inode = 11 }; /* written by pwfs, triggered by cam */
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
    int a                     = stream(&m, "a", 21);
    int b                     = stream(&m, "b", 22);
    m.streams[a].cnt0         = 100;
    OV_STREAM *s              = &m.streams[b];
    *ov_stream_add_hop(&m, s) = /* wrote b, triggered by a */
        (OV_TRACE_HOP) {
            .pid = 600, .inode = 21, .cnt0 = 97, .tstart_ns = 1000000, .tupdate_ns = 1250000
        };
    *ov_stream_add_hop(&m, s) =
        (OV_TRACE_HOP) { .pid = 601, .tstart_ns = 500000, .tupdate_ns = 900000 }; /* wrote a */
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
    CHECK(ov_trace_end_to_end_ns(&m, s) == 750000);
}

/* An FPS running a process of a loop is a loop member (F7 lists it), counted once with its process */
static void test_loop_fps_members(void)
{
    static OV_MODEL m;
    memset(&m, 0, sizeof(m));
    int a = stream(&m, "la", 31);
    int b = stream(&m, "lb", 32);
    proc(&m, "pa", 70); /* pa: la -> lb, pb: lb -> la */
    proc(&m, "pb", 71);
    *ov_stream_add_hop(&m, &m.streams[b]) = (OV_TRACE_HOP) { .pid = 70, .inode = 31 };
    *ov_stream_add_hop(&m, &m.streams[a]) = (OV_TRACE_HOP) { .pid = 71, .inode = 32 };
    snprintf(m.fps[0].name, sizeof(m.fps[0].name), "fpa");
    m.fps[0].valid     = 1;
    m.fps[0].run_alive = 1;
    m.fps[0].runpid    = 70;
    m.nb_fps           = 1;
    ov_build_graph(&m);
    ov_detect_loops(&m, SG_MODE_FULL);
    CHECK(m.nb_loops == 1);
    const OV_LOOP *lp = &m.loops[0];
    CHECK(lp->nb_procs == 2 && lp->nb_fps == 1 && lp->fps_indices[0] == 0 && lp->fps_via_run == 1U);
    CHECK(ov_loop_units(lp) == 2);
    CHECK(m.fps[0].nb_loops == 1 && m.fps[0].primary_loop_id == lp->loop_id);
}

/* Trace hops: one contiguous run per stream in the model's pool; a full pool refuses more */
static void test_hop_pool(void)
{
    static OV_MODEL m;
    memset(&m, 0, sizeof(m));
    OV_STREAM *a = &m.streams[0], *b = &m.streams[1];
    ov_stream_add_hop(&m, a)->pid = 1;
    ov_stream_add_hop(&m, a)->pid = 2;
    ov_stream_add_hop(&m, b)->pid = 3;
    CHECK(a->nb_proctrace == 2 && b->nb_proctrace == 1 && m.nb_trace == 3);
    CHECK(ov_stream_hops(&m, a)[1].pid == 2 && ov_stream_hops(&m, b)[0].pid == 3);
    m.nb_trace = OV_MAX_TRACE;
    CHECK(ov_stream_add_hop(&m, b) == NULL && b->nb_proctrace == 1);
    CHECK(sizeof(OV_STREAM) <= 512); /* the 10000-stream model stays small */
}

/* Directory entries: regular file, link (target inode), dangling link; failure reasons */
static void test_stream_entries(void)
{
    char dir[] = "/tmp/ov-sentry-XXXXXX", path[256], lpath[256], target[80];
    CHECK(mkdtemp(dir) != NULL);
    snprintf(path, sizeof(path), "%s/a.im.shm", dir);
    FILE *f = fopen(path, "w");
    CHECK(f != NULL);
    fclose(f);
    snprintf(lpath, sizeof(lpath), "%s/b.im.shm", dir);
    CHECK(symlink(path, lpath) == 0);
    snprintf(lpath, sizeof(lpath), "%s/c.im.shm", dir);
    CHECK(symlink("/nonexistent/z.im.shm", lpath) == 0);

    struct stat st;
    CHECK(stat(path, &st) == 0);
    DIR           *d = opendir(dir);
    struct dirent *ep;
    int            seen = 0;
    while (d != NULL && (ep = readdir(d)) != NULL)
    {
        ino_t ino;
        int   link, rc = ov_stream_dirent(dir, ep, &ino, &link, target, sizeof(target));
        if (strcmp(ep->d_name, "a.im.shm") == 0)
        {
            CHECK(rc == 0 && !link && ino == st.st_ino);
            seen++;
        }
        else if (strcmp(ep->d_name, "b.im.shm") == 0)
        {
            CHECK(rc == 0 && link && ino == st.st_ino && strcmp(target, "a") == 0);
            seen++;
        }
        else if (strcmp(ep->d_name, "c.im.shm") == 0)
        {
            CHECK(rc == OV_SERR_DANGLING && link);
            seen++;
        }
    }
    closedir(d);
    CHECK(seen == 3);
    CHECK(ov_stream_map_err(-1, 1) == OV_SERR_OPEN && ov_stream_map_err(-2, 1) == OV_SERR_FILE);
    CHECK(ov_stream_map_err(0, 0) == OV_SERR_VERSION && ov_stream_map_err(0, 1) == OV_SERR_DAMAGED);
    CHECK(strcmp(ov_stream_err_label(OV_SERR_DANGLING), "dangling link") == 0 &&
          ov_stream_err_label(0)[0] == 0);
    const char *rm[] = { "a.im.shm", "b.im.shm", "c.im.shm" };
    for (int i = 0; i < 3; i++)
    {
        snprintf(path, sizeof(path), "%s/%s", dir, rm[i]);
        CHECK(unlink(path) == 0);
    }
    CHECK(rmdir(dir) == 0);
}

int main(void)
{
    test_edge_evidence();
    test_stream_entries();
    test_hop_pool();
    test_adjacency();
    test_trace_timing();
    test_loop_fps_members();
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}
