// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkctrl_unit_view: display order of the list panels on synthetic models (no shared memory). */

#include "overview_data_internal.h"
#include "overview_render_internal.h"

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

/* A model with streams @names (update_hz = position) and sequence @seq; heap, the model is large */
static OV_MODEL *mk_streams(const char *const *names, int n, uint64_t seq)
{
    OV_MODEL *m = calloc(1, sizeof(*m));
    for (int i = 0; i < n; i++)
    {
        snprintf(m->streams[i].name, sizeof(m->streams[i].name), "%s", names[i]);
        m->streams[i].update_hz = (double) i;
        m->streams[i].node_idx  = -1;
    }
    m->nb_streams = n;
    m->seq        = seq;
    return m;
}

/* 1 if the display order of panel @p in @m spells @want */
static int order_is(const OV_MODEL *m, int p, const char *const *want, int n)
{
    int        no;
    const int *o = ov_order_get(m, p, &no);
    if (no != n)
    {
        return 0;
    }
    for (int r = 0; r < n; r++)
    {
        const char *name = (p == OV_ORDER_PROCS) ? m->procs[o[r]].name : m->streams[o[r]].name;
        if (strcmp(name, want[r]) != 0)
        {
            return 0;
        }
    }
    return 1;
}

/* Before any sort: scan order. Sorting orders the index array and leaves the model untouched */
static void test_sort_does_not_touch_model(void)
{
    static const char *const names[] = { "cc", "aa", "bb" };
    OV_MODEL                *m       = mk_streams(names, 3, 1);
    OV_MODEL                *copy    = malloc(sizeof(*m));
    memcpy(copy, m, sizeof(*m));
    ov_order_reset();
    CHECK(order_is(m, OV_ORDER_STREAMS, names, 3));
    ov_order_sort(m, OV_ORDER_STREAMS, 0, 0);
    static const char *const asc[] = { "aa", "bb", "cc" };
    CHECK(order_is(m, OV_ORDER_STREAMS, asc, 3));
    ov_order_sort(m, OV_ORDER_STREAMS, 0, 1);
    static const char *const desc[] = { "cc", "bb", "aa" };
    CHECK(order_is(m, OV_ORDER_STREAMS, desc, 3));
    ov_order_sort(m, OV_ORDER_STREAMS, 3, 1); /* Hz descending: update_hz = scan position */
    static const char *const hz[] = { "bb", "aa", "cc" };
    CHECK(order_is(m, OV_ORDER_STREAMS, hz, 3));
    CHECK(memcmp(copy, m, sizeof(*m)) == 0);
    free(copy);
    free(m);
}

/* The order of the last sort survives new models: known entries keep their rank, new ones go last by
 * name, vanished ones drop out; the new model's own scan order does not matter */
static void test_order_survives_new_models(void)
{
    static const char *const n1[] = { "s3", "s1", "s2" };
    OV_MODEL                *m1   = mk_streams(n1, 3, 10);
    ov_order_reset();
    ov_order_sort(m1, OV_ORDER_STREAMS, 0, 1);                          /* s3 s2 s1 */
    static const char *const n2[]   = { "s1", "new_b", "s3", "new_a" }; /* s2 gone, two new */
    OV_MODEL                *m2     = mk_streams(n2, 4, 11);
    static const char *const want[] = { "s3", "s1", "new_a", "new_b" };
    CHECK(order_is(m2, OV_ORDER_STREAMS, want, 4));
    m2->seq = 12; /* same slot address, new publication: rebuilt, same result */
    CHECK(order_is(m2, OV_ORDER_STREAMS, want, 4));
    free(m1);
    free(m2);
}

/* Processes with the same name order by PID, and keep their own ranks across models */
static void test_same_name_procs(void)
{
    OV_MODEL *m       = calloc(1, sizeof(*m));
    int       pids[3] = { 300, 100, 200 };
    for (int i = 0; i < 3; i++)
    {
        snprintf(m->procs[i].name, sizeof(m->procs[i].name), "loop");
        m->procs[i].PID      = pids[i];
        m->procs[i].node_idx = -1;
    }
    m->nb_procs = 3;
    m->seq      = 20;
    ov_order_reset();
    ov_order_sort(m, OV_ORDER_PROCS, 0, 0);
    int        n;
    const int *o = ov_order_get(m, OV_ORDER_PROCS, &n);
    CHECK(n == 3 && m->procs[o[0]].PID == 100 && m->procs[o[1]].PID == 200 &&
          m->procs[o[2]].PID == 300);
    /* reshuffled next scan: the PID order holds */
    OV_PROC t   = m->procs[0];
    m->procs[0] = m->procs[2];
    m->procs[2] = t;
    m->seq      = 21;
    o           = ov_order_get(m, OV_ORDER_PROCS, &n);
    CHECK(n == 3 && m->procs[o[0]].PID == 100 && m->procs[o[1]].PID == 200 &&
          m->procs[o[2]].PID == 300);
    free(m);
}

/* The row cache: built once per input change, missed on a new publication, row_of inverse of rows */
static void test_view_cache(void)
{
    static const char *const names[] = { "v3", "v1", "v2", "x9" };
    OV_MODEL                *m       = mk_streams(names, 4, 30);
    static OV_LAYOUT         lay;
    memset(&lay, 0, sizeof(lay));
    lay.sub[OV_SUB_LOOPS].sel = -1;
    ov_order_reset();
    ov_order_sort(m, OV_ORDER_STREAMS, 0, 0);

    int           n;
    unsigned long b0   = ov_view_build_count();
    const int    *rows = ov_view_rows(&lay, m, OV_FOCUS_STREAMS, &n);
    CHECK(n == 4 && strcmp(m->streams[rows[0]].name, "v1") == 0);
    CHECK(ov_view_build_count() == b0 + 1);
    ov_view_rows(&lay, m, OV_FOCUS_STREAMS, &n); /* nothing changed: cached */
    CHECK(ov_view_build_count() == b0 + 1);
    for (int r = 0; r < n; r++)
    {
        CHECK(ov_view_row_of(&lay, m, OV_FOCUS_STREAMS, rows[r]) == r);
    }

    snprintf(lay.panel[OV_FOCUS_STREAMS].filter, sizeof(lay.panel[0].filter), "^v");
    lay.panel[OV_FOCUS_STREAMS].filter_on = 1;
    rows = ov_view_rows(&lay, m, OV_FOCUS_STREAMS, &n); /* filter changed: rebuilt */
    CHECK(n == 3 && ov_view_build_count() == b0 + 2);
    CHECK(ov_view_row_of(&lay, m, OV_FOCUS_STREAMS, 3) == -1); /* x9 filtered out */

    m->seq = 31; /* new publication in the same slot */
    ov_view_rows(&lay, m, OV_FOCUS_STREAMS, &n);
    CHECK(ov_view_build_count() == b0 + 3);

    /* identity lookups through the name index agree with a linear scan */
    OV_TARGET t;
    for (int i = 0; i < 4; i++)
    {
        ov_tgt_set(&t, m, OV_FOCUS_STREAMS, i);
        CHECK(ov_tgt_find(m, OV_FOCUS_STREAMS, &t) == i);
    }
    snprintf(t.name, sizeof(t.name), "absent");
    CHECK(ov_tgt_find(m, OV_FOCUS_STREAMS, &t) == -1);
    free(m);
}

/* Column tables: unique ids, one column per sort key, <= OV_TABLE_MAX_COLS, @hidden columns hidden in compact
 * mode, and the >/< cycle over visible sortable columns only */
static void check_table(const OV_TABLE *t, OV_LAYOUT *lay, int hidden)
{
    CHECK(t->n <= OV_TABLE_MAX_COLS);
    for (int a = 0; a < t->n; a++)
    {
        for (int b = a + 1; b < t->n; b++)
        {
            CHECK(strcmp(t->cols[a].id, t->cols[b].id) != 0);
            CHECK(t->cols[a].sort_key < 0 || t->cols[a].sort_key != t->cols[b].sort_key);
        }
    }
    lay->compact_mode = 1;
    int vis[OV_TABLE_MAX_COLS];
    CHECK(ov_table_visible(t, lay, vis) == t->n - hidden);
    int k = 0;
    for (int i = 0; i < 30; i++)
    {
        k     = ov_table_next_sort(t, lay, k, +1);
        int c = ov_table_find_key(t, k);
        CHECK(c >= 0 && !(t->cols[c].flags & OV_COLF_COMPACT_HIDE));
    }
}

/* FPS table: the >/< cycle wraps, and a few cells */
static void test_fps_table(void)
{
    const OV_TABLE  *t = &ov_fps_table;
    static OV_LAYOUT lay;
    memset(&lay, 0, sizeof(lay));
    check_table(t, &lay, 2); /* description and sparkline */
    int mem = t->cols[ov_table_find_id(t, "mem")].sort_key;
    int anc = t->cols[ov_table_find_id(t, "anc")].sort_key;
    CHECK(ov_table_next_sort(t, &lay, mem, +1) == anc); /* wraps from the last sortable column */
    CHECK(ov_table_next_sort(t, &lay, anc, -1) == mem);
    CHECK(strcmp(ov_table_status_label(t, anc), "ancestry") == 0);

    OV_MODEL *m = calloc(1, sizeof(*m));
    snprintf(m->fps[0].name, sizeof(m->fps[0].name), "fpsx");
    m->fps[0].confpid = 0;
    m->nb_fps         = 1;
    OV_ROW_CTX rc;
    memset(&rc, 0, sizeof(rc));
    rc.lay = &lay;
    rc.m   = m;
    OV_COLCELL c;
    memset(&c, 0, sizeof(c));
    t->cols[ov_table_find_id(t, "cpid")].cell(&c, &rc);
    CHECK(strcmp(c.text, "-") == 0);
    memset(&c, 0, sizeof(c));
    t->cols[ov_table_find_id(t, "tmx")].cell(&c, &rc);
    CHECK(strcmp(c.text, "---") == 0);
    free(m);
}

/* PROC table: hidden columns, every sort key has a column, and a few cells */
static void test_procs_table(void)
{
    const OV_TABLE  *t = &ov_procs_table;
    static OV_LAYOUT lay;
    memset(&lay, 0, sizeof(lay));
    check_table(t, &lay, 5); /* TRG, trig-strm, exec, DUTY, MISSED */
    for (int k = 0; k < 11; k++)
    {
        CHECK(ov_table_find_key(t, k) >= 0);
    }
    CHECK(strcmp(ov_table_status_label(t, 2), "stat") == 0);
    CHECK(strcmp(ov_table_status_label(t, 8), "CPU%") == 0);

    OV_MODEL *m = calloc(1, sizeof(*m));
    snprintf(m->procs[0].name, sizeof(m->procs[0].name), "pa");
    m->procs[0].rt_priority = -1;
    m->procs[0].cpu_used    = 12.5;
    m->nb_procs             = 1;
    OV_ROW_CTX rc;
    memset(&rc, 0, sizeof(rc));
    rc.lay             = &lay;
    rc.m               = m;
    const char *ids[]  = { "prio", "cpu", "duty", "exec" };
    const char *want[] = { "-", "12.5%", "-", "-" };
    for (int i = 0; i < 4; i++)
    {
        OV_COLCELL c;
        memset(&c, 0, sizeof(c));
        t->cols[ov_table_find_id(t, ids[i])].cell(&c, &rc);
        CHECK(strcmp(c.text, want[i]) == 0);
    }
    free(m);
}

/* STREAMS table: per-semaphore cells, RPID overflow marker, NAME highlight */
static void test_streams_table(void)
{
    const OV_TABLE  *t = &ov_streams_table;
    static OV_LAYOUT lay;
    memset(&lay, 0, sizeof(lay));
    check_table(t, &lay, 3); /* INODE, COUNT, SEMS */

    OV_MODEL  *m = calloc(1, sizeof(*m));
    OV_STREAM *s = &m->streams[0];
    snprintf(s->name, sizeof(s->name), "wfsim");
    s->nb_sem       = 3;
    s->semval[0]    = 0;
    s->semval[1]    = 12;
    s->semval[2]    = -1;
    s->nb_read_pids = 5;
    for (int i = 0; i < 5; i++)
    {
        s->read_pids[i] = 1000001 + i;
    }
    m->nb_streams = 1;
    OV_ROW_CTX rc;
    memset(&rc, 0, sizeof(rc));
    rc.lay = &lay;
    rc.m   = m;
    OV_COLCELL c;
    memset(&c, 0, sizeof(c));
    t->cols[ov_table_find_id(t, "sems")].cell(&c, &rc);
    CHECK(strcmp(c.text, "0+-.......") == 0 && c.nspan == 10);
    memset(&c, 0, sizeof(c));
    t->cols[ov_table_find_id(t, "rpid")].cell(&c, &rc);
    CHECK(strcmp(c.text, "1000001:1000002+3") == 0); /* a third PID and "+2" would not fit in 23 */
    CHECK((int) strlen(c.text) <= t->cols[ov_table_find_id(t, "rpid")].width);

    regex_t re;
    CHECK(regcomp(&re, "fs", REG_EXTENDED) == 0);
    rc.re = &re;
    memset(&c, 0, sizeof(c));
    t->cols[ov_table_find_id(t, "name")].cell(&c, &rc);
    CHECK(strcmp(c.text, "w\x01"
                         "fs\x02"
                         "im") == 0);
    regfree(&re);
    free(m);
}

/* Graph walks are cached per model, root and mode; a stream also relates to its writer's FPS (two hops) */
static void test_derived_and_related(void)
{
    static OV_MODEL m, copy;
    memset(&m, 0, sizeof(m));
    for (int i = 0; i < 2; i++)
    {
        snprintf(m.streams[i].name, sizeof(m.streams[i].name), i ? "sout" : "sin");
        m.streams[i].valid = 1;
        m.streams[i].inode = (ino_t) (5 + i);
    }
    m.nb_streams                    = 2;
    m.streams[1].nb_proctrace       = 1; /* pw writes sout, triggered by sin */
    m.streams[1].proctrace_pid[0]   = 77;
    m.streams[1].proctrace_inode[0] = 5;
    snprintf(m.procs[0].name, sizeof(m.procs[0].name), "pw");
    m.procs[0].valid  = 1;
    m.procs[0].active = 1;
    m.procs[0].PID    = 77;
    m.nb_procs        = 1;
    snprintf(m.fps[0].name, sizeof(m.fps[0].name), "fw"); /* fw runs pw */
    m.fps[0].valid     = 1;
    m.fps[0].run_alive = 1;
    m.fps[0].runpid    = 77;
    m.nb_fps           = 1;
    ov_build_graph(&m);

    m.seq       = 9;
    unsigned b0 = ov_derived_build_count();
    int      sn = m.streams[0].node_idx;
    CHECK(ov_node_depths(&m, sn, SG_MODE_FULL) == ov_node_depths(&m, sn, SG_MODE_FULL));
    CHECK(ov_derived_build_count() == b0 + 1); /* same model, root and mode: one walk */
    m.seq = 10;
    ov_node_depths(&m, sn, SG_MODE_FULL);
    CHECK(ov_derived_build_count() == b0 + 2); /* a new model */
    m.seq = 0;
    ov_node_depths(&m, sn, SG_MODE_FULL);
    ov_node_depths(&m, sn, SG_MODE_FULL);
    CHECK(ov_derived_build_count() == b0 + 4); /* never published: never cached */
    memcpy(&copy, &m, sizeof(m));
    ov_lineage(&m, 0, SG_MODE_FULL);
    CHECK(memcmp(&copy, &m, sizeof(m)) == 0);

    static OV_LAYOUT  lay;
    static OV_RELATED rel;
    memset(&lay, 0, sizeof(lay));
    lay.focus = OV_FOCUS_STREAMS;
    m.seq     = 11;
    ov_sel_model(&lay, &m, OV_FOCUS_STREAMS, 1);
    ov_compute_related(&lay, &m, &rel);
    CHECK(bget(rel.procs, 0) && bget(rel.proc_writes, 0));
    CHECK(bget(rel.fps, 0) && bget(rel.fps_writes, 0)); /* sout -> pw -> fw */
}

/* CONNECTIONS health: reader state, FPS, rate mismatch; a stall marks its row and dims the rows below */
static void test_graph_health(void)
{
    static OV_MODEL m;
    memset(&m, 0, sizeof(m));
    snprintf(m.streams[0].name, sizeof(m.streams[0].name), "cam");
    m.streams[0].valid     = 1;
    m.streams[0].update_hz = 1000.0;
    m.nb_streams           = 1;
    snprintf(m.procs[0].name, sizeof(m.procs[0].name), "wfs");
    m.procs[0].valid       = 1;
    m.procs[0].active      = 1;
    m.procs[0].PID         = 50;
    m.procs[0].loopstat    = PROCESSINFO_LOOPSTAT_ACTIVE;
    m.procs[0].stale_count = OV_STALE_SCANS; /* stalled */
    m.procs[0].loop_hz     = 10.0;
    m.nb_procs             = 1;
    snprintf(m.fps[0].name, sizeof(m.fps[0].name), "wfsfps");
    m.fps[0].valid     = 1;
    m.fps[0].run_alive = 1;
    m.fps[0].runpid    = 50;
    m.nb_fps           = 1;
    ov_build_graph(&m);

    SG_TREE_NODE rows[3];
    memset(rows, 0, sizeof(rows));
    rows[0] =
        (SG_TREE_NODE) { .stream_idx = 0, .reader_node_idx = m.procs[0].node_idx, .depth = 0 };
    rows[1] = (SG_TREE_NODE) { .stream_idx = 0, .reader_node_idx = -1, .depth = 1 };
    rows[2] = (SG_TREE_NODE) { .stream_idx = 0, .reader_node_idx = -1, .depth = 0 };
    OV_ROW_HEALTH h;
    ov_graph_row_health(&m, &rows[0], &h);
    CHECK(strcmp(h.state, "STAL") == 0 && h.stalled && h.mismatch);
    CHECK(strcmp(h.hz, "1.00k") == 0 && strcmp(h.fps, "wfsfps") == 0);
    int8_t cause[3], dim[3];
    ov_graph_stalls(&m, rows, 3, cause, dim);
    CHECK(cause[0] && !dim[0] && dim[1] && !cause[1] &&
          !dim[2]); /* the sibling subtree is not dimmed */
    char b[16];
    CHECK(strcmp(ov_fmt_si(b, sizeof(b), 12345.0), "12.3k") == 0);
    CHECK(strcmp(ov_fmt_si(b, sizeof(b), 999.9), "999.9") == 0);
}

int main(void)
{
    test_sort_does_not_touch_model();
    test_order_survives_new_models();
    test_same_name_procs();
    test_view_cache();
    test_fps_table();
    test_procs_table();
    test_streams_table();
    test_derived_and_related();
    test_graph_health();
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}
