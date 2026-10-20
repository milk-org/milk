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
    lay.sel_loop = -1;
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

/* Column tables: unique ids, one column per sort key, <= OV_TABLE_MAX_COLS, the >/< cycle over visible
 * sortable columns only; and a few FPS cells */
static void test_fps_table(void)
{
    const OV_TABLE *t = &ov_fps_table;
    CHECK(t->n <= OV_TABLE_MAX_COLS);
    for (int a = 0; a < t->n; a++)
    {
        for (int b = a + 1; b < t->n; b++)
        {
            CHECK(strcmp(t->cols[a].id, t->cols[b].id) != 0);
            CHECK(t->cols[a].sort_key < 0 || t->cols[a].sort_key != t->cols[b].sort_key);
        }
    }
    static OV_LAYOUT lay;
    memset(&lay, 0, sizeof(lay));
    lay.compact_mode = 1;
    int vis[OV_TABLE_MAX_COLS];
    int nv = ov_table_visible(t, &lay, vis);
    CHECK(nv == t->n - 2); /* description and sparkline hidden */
    int k = 0;
    for (int i = 0; i < 20; i++) /* the cycle only lands on visible sortable columns */
    {
        k     = ov_table_next_sort(t, &lay, k, +1);
        int c = ov_table_find_key(t, k);
        CHECK(c >= 0 && !(t->cols[c].flags & OV_COLF_COMPACT_HIDE));
    }
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

int main(void)
{
    test_sort_does_not_touch_model();
    test_order_survives_new_models();
    test_same_name_procs();
    test_view_cache();
    test_fps_table();
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}
