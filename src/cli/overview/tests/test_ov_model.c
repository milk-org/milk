// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkctrl_unit_model: milk-CTRL data-layer checks that need no shared memory. */

#include "overview_data_internal.h"
#include "data/overview_data_loops_internal.h"
#include "stream_graph.h"

#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

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

/* A live process owned by another user (kill() -> EPERM) is alive, not dead (links-4) */
static void test_pid_liveness(void)
{
    CHECK(pid_get_status(getpid()) == OV_PID_ALIVE);
    if (geteuid() != 0)
    {
        CHECK(pid_get_status(1) != OV_PID_DEAD);
    }
}

/* /proc/<pid>/stat parsing survives a comm field with spaces (gap-11) */
static void test_stat_with_spaces(void)
{
    uint64_t ut = 0, st = 0;
    pid_t    pid = fork();
    if (pid == 0)
    {
        prctl(PR_SET_NAME, "a b c", 0, 0, 0);
        pause();
        _exit(0);
    }
    usleep(50000);
    CHECK(pid_get_cpu_ticks(pid, &ut, &st) == 0);
    CHECK(pid_check_zombie(pid) == 0);
    kill(pid, SIGKILL);
    int zombie = 0;
    for (int i = 0; i < 100 && !zombie; i++)
    {
        usleep(10000);
        zombie = pid_check_zombie(pid);
    }
    CHECK(zombie == 1);
    waitpid(pid, NULL, 0);
}

/* One /proc read per PID per scan: values for a live PID, none for an exited one (perf-3) */
static void test_pid_table(void)
{
    uint64_t ut = 0, st = 0;
    pid_cache_reset();
    CHECK(pid_get_start_ticks(getpid()) > 0);
    CHECK(pid_get_rss_kb(getpid()) > 0);
    CHECK(pid_get_cpu_ticks(getpid(), &ut, &st) == 0);
    pid_t pid = fork();
    if (pid == 0)
    {
        _exit(0);
    }
    waitpid(pid, NULL, 0);
    pid_cache_reset();
    CHECK(pid_get_status(pid) == OV_PID_DEAD);
    CHECK(pid_get_rss_kb(pid) == 0 && pid_get_start_ticks(pid) == 0);
    CHECK(pid_get_cpu_ticks(pid, &ut, &st) == -1);
    CHECK(!pid_same_process(pid, 1));
    pid_cache_reset(); /* remembered as dead: still dead without a new probe */
    CHECK(!pid_is_alive(pid));
}

/* A PID shared by dead registry entries and a live one resolves to the live, else newest (links-11) */
static void test_find_proc_by_pid(void)
{
    static OV_MODEL m;
    memset(&m, 0, sizeof(m));
    m.nb_procs = 3;
    for (int i = 0; i < 3; i++)
    {
        m.procs[i].valid = 1;
        m.procs[i].PID   = 4242;
    }
    m.procs[0].createtime = 10.0;
    m.procs[1].createtime = 30.0;
    m.procs[2].createtime = 20.0;
    CHECK(ov_find_proc_by_pid(&m, 4242) == 1);
    m.procs[2].active = 1;
    CHECK(ov_find_proc_by_pid(&m, 4242) == 2);
    CHECK(ov_find_proc_by_pid(&m, 7) == -1);
}


/** ix_is - match: the value equals *ctx */
static int ix_is(const void *ctx, int32_t v)
{
    return v == *(const int *) ctx;
}

/* Hash index: colliding keys, full table; model lookups agree with the array scan (perf-1) */
static void test_lookup_indexes(void)
{
    OV_INDEX_DEF(16) ix;
    ov_ix_clear(&ix);
    for (int v = 0; v < 8; v++)
    {
        CHECK(ov_ix_put(&ix, 42, v) == 0); /* all on one hash */
    }
    for (int v = 0; v < 8; v++)
    {
        CHECK(ov_ix_find(&ix, 42, ix_is, &v) == v);
    }
    int nine = 9;
    CHECK(ov_ix_find(&ix, 42, ix_is, &nine) == -1);
    for (int v = 8; v < 16; v++)
    {
        CHECK(ov_ix_put(&ix, 7, v) == 0);
    }
    CHECK(ov_ix_put(&ix, 7, 99) == -1); /* full */

    static OV_MODEL m;
    memset(&m, 0, sizeof(m));
    m.nb_streams = 1500;
    for (int i = 0; i < m.nb_streams; i++)
    {
        snprintf(m.streams[i].name, sizeof(m.streams[i].name), "st%d", i);
        m.streams[i].inode = (ino_t) (1000 + 3 * i);
        m.streams[i].valid = (i % 7 != 0);
    }
    m.nb_procs = 400;
    for (int i = 0; i < m.nb_procs; i++)
    {
        m.procs[i].PID        = 100 + i % 150; /* shared by up to 3 entries */
        m.procs[i].valid      = (i % 11 != 0);
        m.procs[i].active     = (i % 3 == 0);
        m.procs[i].createtime = (double) ((i * 37) % 101);
    }
    int lin_s[1600], lin_i[1600], lin_p[300];
    for (int q = 0; q < 1600; q++)
    {
        char name[32];
        snprintf(name, sizeof(name), "st%d", q);
        lin_s[q] = ov_find_stream_by_name(&m, name);
        lin_i[q] = ov_find_stream_by_inode(&m, (ino_t) (1000 + 3 * q));
    }
    for (int q = 0; q < 300; q++)
    {
        lin_p[q] = ov_find_proc_by_pid(&m, 50 + q);
    }
    ov_model_index(&m);
    CHECK(m.indexed == 1);
    for (int q = 0; q < 1600; q++)
    {
        char name[32];
        snprintf(name, sizeof(name), "st%d", q);
        CHECK(ov_find_stream_by_name(&m, name) == lin_s[q]);
        CHECK(ov_find_stream_by_inode(&m, (ino_t) (1000 + 3 * q)) == lin_i[q]);
    }
    for (int q = 0; q < 300; q++)
    {
        CHECK(ov_find_proc_by_pid(&m, 50 + q) == lin_p[q]);
    }

    /* Edge duplicates are dropped, whatever the arrival order */
    m.nb_edges = 0;
    for (int r = 0; r < 3; r++)
    {
        for (int i = 0; i < 1000; i++)
        {
            ov_add_edge(&m, i, i + 1, (ov_edge_type_t) (i % 3), OV_EV_PROCTRACE);
        }
    }
    CHECK(m.nb_edges == 1000);
    ov_add_edge(&m, 5, 6, (ov_edge_type_t) 1, OV_EV_PROCTRACE); /* new type for an existing pair */
    CHECK(m.nb_edges == 1001);
    m.nb_edges = 0; /* a reset array is re-indexed */
    ov_add_edge(&m, 1, 2, (ov_edge_type_t) 1, OV_EV_PROCTRACE);
    CHECK(m.nb_edges == 1);
}

/* Shared labels follow processinfo.h and processtools_trigger.h (gap-1, ux-8) */
static void test_labels(void)
{
    CHECK(strcmp(ov_loopstat_label(PROCESSINFO_LOOPSTAT_ACTIVE, 1), "RUNNING") == 0);
    CHECK(strcmp(ov_loopstat_label(PROCESSINFO_LOOPSTAT_PAUSE, 0), "PAUS") == 0);
    CHECK(strcmp(ov_loopstat_label(PROCESSINFO_LOOPSTAT_STOP, 1), "STOPPED") == 0);
    CHECK(strcmp(ov_loopstat_label(PROCESSINFO_LOOPSTAT_CRASHED, 0), "CRSH") == 0);
    CHECK(strcmp(ov_loopstat_label(42, 1), "UNKNOWN") == 0);
    CHECK(strcmp(ov_loopstat_label(PROCESSINFO_LOOPSTAT_ERROR, 0), "ERR") == 0);
    for (int st = 0; st < OV_ST_UNKNOWN; st++)
    {
        CHECK(strstr(OV_STATE_LEGEND, ov_state_name((ov_state_t) st, 0)) != NULL);
        CHECK(strlen(ov_state_name((ov_state_t) st, 0)) <= 4);
    }
    CHECK(strcmp(ov_ctrlval_label(PROCESSINFO_CTRLVAL_PAUSE), "PAUSE") == 0);
    CHECK(strcmp(ov_ctrlval_label(PROCESSINFO_CTRLVAL_INCR), "STEP") == 0);
    CHECK(strcmp(ov_trigmode_label(3, 1), "SEMAPHORE") == 0);
    CHECK(strcmp(ov_trigmode_label(1, 1), "CNT0") == 0);
    CHECK(strcmp(ov_trigmode_label(4, 0), "DLY") == 0);
}

/* A PID names the same process only while its start time matches (robust-3, control-5) */
static void test_pid_identity(void)
{
    uint64_t me = pid_get_start_ticks(getpid());
    CHECK(me != 0);
    CHECK(pid_same_process(getpid(), me) == 1);
    CHECK(pid_same_process(getpid(), me + 1) == 0);
    CHECK(pid_same_process(getpid(), 0) == 0);
}

/* Graph builders for the loop and lineage cases: no shared memory, no scan */
static int add_node(OV_MODEL *m, ov_node_type_t type, const char *name)
{
    int ni            = m->nb_nodes++;
    m->nodes[ni].type = type;
    snprintf(m->nodes[ni].name, sizeof(m->nodes[ni].name), "%s", name);
    if (type == OV_NODE_STREAM)
    {
        m->nodes[ni].index = m->nb_streams;
        snprintf(m->streams[m->nb_streams].name, sizeof(m->streams[0].name), "%s", name);
        m->streams[m->nb_streams++].node_idx = ni;
    }
    else if (type == OV_NODE_PROC)
    {
        m->nodes[ni].index = m->nb_procs;
        snprintf(m->procs[m->nb_procs].name, sizeof(m->procs[0].name), "%s", name);
        m->procs[m->nb_procs].valid    = 1;
        m->procs[m->nb_procs].loopstat = PROCESSINFO_LOOPSTAT_ACTIVE;
        m->procs[m->nb_procs].loop_hz  = 100.0;
        m->procs[m->nb_procs++].PID    = 1000 + ni;
    }
    else
    {
        m->nodes[ni].index = m->nb_fps;
        snprintf(m->fps[m->nb_fps].name, sizeof(m->fps[0].name), "%s", name);
        m->fps[m->nb_fps++].node_idx = ni;
    }
    ov_model_adjacency(m); /* the graph is set by hand: keep its adjacency current */
    return ni;
}

static void add_edge(OV_MODEL *m, int src, int tgt, ov_edge_type_t type)
{
    OV_EDGE *e  = &m->edges[m->nb_edges++];
    e->src_node = src;
    e->tgt_node = tgt;
    e->type     = type;
    e->evidence = OV_EV_PROCTRACE;
    ov_model_adjacency(m);
}

/* stream s -> unit u -> stream t -> unit v -> s, units being processes or FPS */
static void add_ring(OV_MODEL      *m,
                     const char    *s,
                     const char    *u,
                     const char    *t,
                     const char    *v,
                     ov_node_type_t ut)
{
    int            ns = add_node(m, OV_NODE_STREAM, s), nu = add_node(m, ut, u);
    int            nt = add_node(m, OV_NODE_STREAM, t), nv = add_node(m, ut, v);
    ov_edge_type_t in =
        (ut == OV_NODE_PROC) ? OV_EDGE_STREAM_TRIGGERS_PROC : OV_EDGE_FPS_INPUT_STREAM;
    ov_edge_type_t out =
        (ut == OV_NODE_PROC) ? OV_EDGE_PROC_WRITES_STREAM : OV_EDGE_FPS_OUTPUT_STREAM;
    add_edge(m, ns, nu, in);
    add_edge(m, nu, nt, out);
    add_edge(m, nt, nv, in);
    add_edge(m, nv, ns, out);
}

/* Loop IDs follow the loop signature, not the scan order (links-9) */
static void test_loop_ids_stable(void)
{
    static OV_MODEL a, b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    s_names_loaded = 1; /* no loop-name file */
    add_ring(&a, "a1", "pa", "a2", "qa", OV_NODE_PROC);
    add_ring(&a, "x1", "px", "x2", "qx", OV_NODE_PROC);
    add_ring(&b, "x1", "px", "x2", "qx", OV_NODE_PROC);
    add_ring(&b, "a1", "pa", "a2", "qa", OV_NODE_PROC);
    ov_detect_loops(&a, SG_MODE_FULL);
    ov_detect_loops(&b, SG_MODE_FULL);
    CHECK(a.nb_loops == 2 && b.nb_loops == 2);
    CHECK(strcmp(a.loops[0].signature, b.loops[0].signature) == 0);
    CHECK(a.loops[0].loop_id == 1 && b.loops[0].loop_id == 1);
    CHECK(strncmp(b.loops[0].name, "L01 [a1", 7) == 0);
    CHECK(a.loops_truncated == 0);
}

/* Health counts every member: 0 Hz, crashed, stalled; FPS loops need a live run process (links-3/10) */
static void test_loop_health(void)
{
    static OV_MODEL m;
    memset(&m, 0, sizeof(m));
    s_names_loaded = 1;
    add_ring(&m, "a1", "pa", "a2", "qa", OV_NODE_PROC);
    m.streams[0].update_hz = 0.0;
    m.streams[1].update_hz = 100.0;
    ov_detect_loops(&m, SG_MODE_FULL);
    CHECK(m.nb_loops == 1 && m.loops[0].min_hz == 0.0 && m.loops[0].is_running == 1);
    m.procs[0].stale_count = OV_STALE_SCANS;
    ov_detect_loops(&m, SG_MODE_FULL);
    CHECK(m.loops[0].is_stale == 1 && m.loops[0].is_error == 0);
    m.procs[1].loopstat = PROCESSINFO_LOOPSTAT_CRASHED;
    ov_detect_loops(&m, SG_MODE_FULL);
    CHECK(m.loops[0].is_error == 1 && m.loops[0].is_running == 0);

    static OV_MODEL f;
    memset(&f, 0, sizeof(f));
    add_ring(&f, "c1", "fa", "c2", "fb", OV_NODE_FPS);
    ov_detect_loops(&f, SG_MODE_FULL);
    CHECK(f.nb_loops == 1 && f.loops[0].is_running == 0); /* declared only: IDLE, not RUN */
    int rp = add_node(&f, OV_NODE_PROC, "runner");
    for (int i = 0; i < f.nb_fps; i++)
    {
        f.fps[i].run_alive = 1;
        f.fps[i].runpid    = f.procs[f.nodes[rp].index].PID;
    }
    ov_detect_loops(&f, SG_MODE_FULL);
    CHECK(f.loops[0].is_running == 1);
}

/* More cycles than OV_MAX_LOOPS are flagged, not silently dropped (links-10) */
static void test_loop_truncation(void)
{
    static OV_MODEL m;
    memset(&m, 0, sizeof(m));
    s_names_loaded = 1;
    int hub        = add_node(&m, OV_NODE_STREAM, "hub");
    for (int i = 0; i <= OV_MAX_LOOPS; i++)
    {
        char name[16];
        snprintf(name, sizeof(name), "p%02d", i);
        int p = add_node(&m, OV_NODE_PROC, name);
        add_edge(&m, hub, p, OV_EDGE_STREAM_TRIGGERS_PROC);
        add_edge(&m, p, hub, OV_EDGE_PROC_WRITES_STREAM);
    }
    ov_detect_loops(&m, SG_MODE_FULL);
    CHECK(m.nb_loops == OV_MAX_LOOPS && m.loops_truncated == 1);
}

/* A process root shows only its own streams; an FPS root has a tree (links-2, robust-18) */
static void test_lineage_roots(void)
{
    static OV_MODEL     m;
    static SG_TREE_NODE rows[OV_MAX_NODES];
    memset(&m, 0, sizeof(m));
    int s_in  = add_node(&m, OV_NODE_STREAM, "in");
    int p     = add_node(&m, OV_NODE_PROC, "unit");
    int s_out = add_node(&m, OV_NODE_STREAM, "out");
    add_node(&m, OV_NODE_STREAM, "zz");
    int s_fi = add_node(&m, OV_NODE_STREAM, "fin");
    int f    = add_node(&m, OV_NODE_FPS, "fpsu");
    int s_fo = add_node(&m, OV_NODE_STREAM, "fout");
    add_edge(&m, s_in, p, OV_EDGE_STREAM_TRIGGERS_PROC);
    add_edge(&m, p, s_out, OV_EDGE_PROC_WRITES_STREAM);
    add_edge(&m, s_fi, f, OV_EDGE_FPS_INPUT_STREAM);
    add_edge(&m, f, s_fo, OV_EDGE_FPS_OUTPUT_STREAM);

    int n = sg_compute_render_tree(&m, p, SG_MODE_FULL, rows);
    CHECK(n >= 2);
    for (int i = 0; i < n; i++)
    {
        CHECK(strcmp(rows[i].name, "zz") != 0 && strcmp(rows[i].name, "fin") != 0);
    }
    CHECK(n > 0 && rows[0].reader_node_idx == p && rows[0].is_target_proc == 1);
    CHECK(sg_tree_reader_proc(&m, &rows[0]) == m.nodes[p].index);

    n = sg_compute_render_tree(&m, f, SG_MODE_FULL, rows);
    CHECK(n >= 2);
    CHECK(n > 0 && rows[0].reader_node_idx == f && sg_tree_reader_proc(&m, &rows[0]) == -1);
}

/* 1 if the graph has an edge @type between stream @sname and FPS @fname, in that direction */
static int has_fps_edge(const OV_MODEL *m,
                        const char     *sname,
                        const char     *fname,
                        ov_edge_type_t  type)
{
    for (int e = 0; e < m->nb_edges; e++)
    {
        const OV_NODE *a = &m->nodes[m->edges[e].src_node], *b = &m->nodes[m->edges[e].tgt_node];
        const OV_NODE *st = (type == OV_EDGE_FPS_OUTPUT_STREAM) ? b : a;
        const OV_NODE *fp = (type == OV_EDGE_FPS_OUTPUT_STREAM) ? a : b;
        if (m->edges[e].type == type && strcmp(st->name, sname) == 0 &&
            strcmp(fp->name, fname) == 0)
        {
            return 1;
        }
    }
    return 0;
}

/* FPS stream direction: the run process's I/O first, then trigger/run-required flags, then the name (links-8) */
static void test_fps_stream_direction(void)
{
    static OV_MODEL m;
    memset(&m, 0, sizeof(m));
    const char *sn[2] = { "imname", "out_x" };
    for (int i = 0; i < 2; i++)
    {
        snprintf(m.streams[i].name, sizeof(m.streams[i].name), "%s", sn[i]);
        m.streams[i].valid = 1;
    }
    m.nb_streams           = 2;
    m.streams[0].write_pid = 777;
    OV_FPS *f              = &m.fps[0];
    snprintf(f->name, sizeof(f->name), "f1");
    f->valid            = 1;
    f->run_alive        = 1;
    f->runpid           = 777;
    f->nb_stream_params = 2;
    for (int i = 0; i < 2; i++)
    {
        snprintf(f->stream_param_name[i], sizeof(f->stream_param_name[i]), "%s", sn[i]);
        snprintf(f->stream_param_value[i], sizeof(f->stream_param_value[i]), "%s", sn[i]);
    }
    f->stream_param_flags[1] = FPFLAG_TRIGGER_STREAM;
    m.nb_fps                 = 1;
    ov_build_graph(&m);
    CHECK(has_fps_edge(&m, "imname", "f1", OV_EDGE_FPS_OUTPUT_STREAM));
    CHECK(has_fps_edge(&m, "out_x", "f1", OV_EDGE_FPS_INPUT_STREAM));
}

/* The filter regex cache: an invalid pattern compiles once, keeps the substring fallback and evicts nothing */
static void test_filter_cache(void)
{
    const char *names[] = { "a(b", "xyz", "ab" };
    int         out[3];
    CHECK(ov_filter_build("x|a", names, 3, out, 3) == 3); /* compiled and cached */
    unsigned long c0 = ov_filter_compile_count();
    for (int i = 0; i < 5; i++)
    {
        CHECK(ov_filter_build("a(", names, 3, out, 3) == 1 && out[0] == 0); /* substring fallback */
    }
    CHECK(ov_filter_compile_count() == c0 + 1);
    CHECK(ov_filter_build("x|a", names, 3, out, 3) == 3);
    CHECK(ov_filter_compile_count() == c0 + 1); /* "x|a" was not evicted */
}

int main(void)
{
    test_pid_liveness();
    test_stat_with_spaces();
    test_find_proc_by_pid();
    test_lookup_indexes();
    test_pid_table();
    test_labels();
    test_pid_identity();
    test_loop_ids_stable();
    test_loop_health();
    test_loop_truncation();
    test_lineage_roots();
    test_fps_stream_direction();
    test_filter_cache();
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}
