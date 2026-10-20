// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_data_order.c
 * @brief   Display order of the list panels, as model index arrays; the model is never permuted.
 *
 * A sort request computes the order and snapshots each entry's rank by identity (name; processes
 * also PID). Later models are shown in that snapshot order, in O(n), until the next sort request;
 * entries the snapshot does not know go last, by name. Before the first sort, order = scan order.
 */

#include "overview_data_internal.h"
#include <stdint.h>
#include <string.h>

#define OV_RANK_SLOTS 4096 /* power of two, > 2 * OV_MAX_STREAMS: short probes */
#define OV_RANK_NONE INT32_MAX

typedef struct
{
    uint64_t key; /* 0 = empty */
    int32_t  rank;
} ov_rank_slot_t;

typedef struct
{
    int             idx[OV_MAX_NODES]; /* model indices in display order */
    int             n;
    const OV_MODEL *m; /* model the order was built for */
    uint64_t        seq;
    int             sorted; /* a rank snapshot exists */
    uint32_t        gen;    /* bumped whenever idx[] changes */
    ov_rank_slot_t  slot[OV_RANK_SLOTS];
} ov_order_t;

static ov_order_t s_ord[OV_ORDER_NB];

/** ov_name_hash - FNV-1a of @s mixed with @extra; never 0. */
static uint64_t ov_name_hash(const char *s, uint64_t extra)
{
    uint64_t h = UINT64_C(14695981039346656037);
    for (; *s != '\0'; s++)
    {
        h = (h ^ (uint8_t) *s) * UINT64_C(1099511628211);
    }
    h ^= extra * UINT64_C(0x9E3779B97F4A7C15);
    return (h == 0) ? 1 : h;
}

/** ov_entry_count - number of entries of @panel in @m. */
static int ov_entry_count(const OV_MODEL *m, int panel)
{
    return (panel == OV_ORDER_STREAMS) ? m->nb_streams
           : (panel == OV_ORDER_PROCS) ? m->nb_procs
                                       : m->nb_fps;
}

/** ov_entry_name - name of entry @i of @panel. */
static const char *ov_entry_name(const OV_MODEL *m, int panel, int i)
{
    return (panel == OV_ORDER_STREAMS) ? m->streams[i].name
           : (panel == OV_ORDER_PROCS) ? m->procs[i].name
                                       : m->fps[i].name;
}

/** ov_entry_key - identity hash of entry @i of @panel (processes: name and PID). */
static uint64_t ov_entry_key(const OV_MODEL *m, int panel, int i)
{
    return ov_name_hash(ov_entry_name(m, panel, i),
                        (panel == OV_ORDER_PROCS) ? (uint64_t) m->procs[i].PID : 0);
}

/** ov_rank_lookup - snapshot rank of @key in @o, or OV_RANK_NONE. */
static int32_t ov_rank_lookup(const ov_order_t *o, uint64_t key)
{
    for (uint32_t h = (uint32_t) key & (OV_RANK_SLOTS - 1);; h = (h + 1) & (OV_RANK_SLOTS - 1))
    {
        if (o->slot[h].key == key)
        {
            return o->slot[h].rank;
        }
        if (o->slot[h].key == 0)
        {
            return OV_RANK_NONE;
        }
    }
}

/** ov_rank_insert - record @rank for @key (first rank wins for duplicate identities). */
static void ov_rank_insert(ov_order_t *o, uint64_t key, int32_t rank)
{
    for (uint32_t h = (uint32_t) key & (OV_RANK_SLOTS - 1);; h = (h + 1) & (OV_RANK_SLOTS - 1))
    {
        if (o->slot[h].key == key)
        {
            return;
        }
        if (o->slot[h].key == 0)
        {
            o->slot[h].key  = key;
            o->slot[h].rank = rank;
            return;
        }
    }
}

/* Rebuild context: ranks of the current model's entries */
static const OV_MODEL *s_rm;
static int             s_rpanel;
static int32_t         s_rank[OV_MAX_NODES];

/** cmp_rank - snapshot rank, then name (unknown entries), then index. */
static int cmp_rank(const void *a, const void *b)
{
    int ia = *(const int *) a, ib = *(const int *) b;
    if (s_rank[ia] != s_rank[ib])
    {
        return (s_rank[ia] > s_rank[ib]) - (s_rank[ia] < s_rank[ib]);
    }
    int r = strcmp(ov_entry_name(s_rm, s_rpanel, ia), ov_entry_name(s_rm, s_rpanel, ib));
    return (r != 0) ? r : (ia > ib) - (ia < ib);
}

/** ov_order_rebuild - order of @m's @panel entries from the rank snapshot (scan order without one). */
static void ov_order_rebuild(ov_order_t *o, const OV_MODEL *m, int panel)
{
    o->n = ov_entry_count(m, panel);
    for (int i = 0; i < o->n; i++)
    {
        o->idx[i] = i;
    }
    if (o->sorted && o->n > 1)
    {
        for (int i = 0; i < o->n; i++)
        {
            s_rank[i] = ov_rank_lookup(o, ov_entry_key(m, panel, i));
        }
        s_rm     = m;
        s_rpanel = panel;
        qsort(o->idx, (size_t) o->n, sizeof(int), cmp_rank);
    }
    o->m   = m;
    o->seq = m->seq;
    o->gen++;
}

/**
 * ov_order_sort - sort @panel of @m by @key/@dir (see ov_sort_streams) and keep that order for later
 * models until the next call. Ancestry keys use the depths set by ov_sort_set_depths().
 */
void ov_order_sort(const OV_MODEL *m, int panel, int key, int dir)
{
    ov_order_t *o = &s_ord[panel];
    if (panel == OV_ORDER_STREAMS)
    {
        ov_sort_streams(m, key, dir, o->idx);
    }
    else if (panel == OV_ORDER_PROCS)
    {
        ov_sort_procs(m, key, dir, o->idx);
    }
    else
    {
        ov_sort_fps(m, key, dir, o->idx);
    }
    o->n = ov_entry_count(m, panel);
    memset(o->slot, 0, sizeof(o->slot));
    for (int r = 0; r < o->n; r++)
    {
        ov_rank_insert(o, ov_entry_key(m, panel, o->idx[r]), r);
    }
    o->sorted = 1;
    o->m      = m;
    o->seq    = m->seq;
    o->gen++;
}

/**
 * ov_order_get - model indices of @panel in display order for @m (rebuilt when @m is a new model).
 * Return: the order (valid until the next ov_order_* call), *@n entries.
 */
const int *ov_order_get(const OV_MODEL *m, int panel, int *n)
{
    ov_order_t *o = &s_ord[panel];
    if (o->m != m || o->seq != m->seq || o->n != ov_entry_count(m, panel))
    {
        ov_order_rebuild(o, m, panel);
    }
    *n = o->n;
    return o->idx;
}

/** ov_order_gen - generation of @panel's order: changes whenever its order array is rebuilt. */
uint32_t ov_order_gen(int panel)
{
    return s_ord[panel].gen;
}

/** ov_order_reset - forget every order and rank snapshot (scan order again). */
void ov_order_reset(void)
{
    for (int p = 0; p < OV_ORDER_NB; p++)
    {
        uint32_t gen = s_ord[p].gen;
        memset(&s_ord[p], 0, sizeof(s_ord[p]));
        s_ord[p].gen = gen + 1; /* never reuse a generation a cache may hold */
    }
}
