// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkctrl_unit_view: display order of the list panels on synthetic models (no shared memory). */

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

int main(void)
{
    test_sort_does_not_touch_model();
    test_order_survives_new_models();
    test_same_name_procs();
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}
