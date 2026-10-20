// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_derived.c
 * @brief   Results of graph walks (node depths, stream lineage), cached per model, root and mode.
 *
 * Render and input ask for the same walks many times per frame and on every key; they are recomputed
 * only when the model (its publication number), the root or the mode changes. A model never published
 * (seq 0, e.g. built by a test) is not cached.
 */

#include "overview_render_internal.h"
#include "stream_graph.h"

#define OV_DERIVED_SLOTS 4

typedef struct
{
    const OV_MODEL *m;
    uint64_t        seq;
    int             root;
    int             mode;
} OV_DERIVED_KEY;

static unsigned s_builds;

/** derived_slot - slot of @keys holding (m, root, mode), or the one to overwrite (*hit 0). */
static int derived_slot(OV_DERIVED_KEY *keys,
                        int            *next,
                        const OV_MODEL *m,
                        int             root,
                        int             mode,
                        int            *hit)
{
    for (int i = 0; i < OV_DERIVED_SLOTS; i++)
    {
        if (m->seq != 0 && keys[i].m == m && keys[i].seq == m->seq && keys[i].root == root &&
            keys[i].mode == mode)
        {
            *hit = 1;
            return i;
        }
    }
    int i   = *next;
    *next   = (*next + 1) % OV_DERIVED_SLOTS;
    keys[i] = (OV_DERIVED_KEY) { m, m->seq, root, mode };
    *hit    = 0;
    s_builds++;
    return i;
}

/** ov_node_depths - signed lineage distance of every node from node @root (127 = unrelated). */
const int8_t *ov_node_depths(const OV_MODEL *m, int root, sg_mode_t mode)
{
    static OV_DERIVED_KEY keys[OV_DERIVED_SLOTS];
    static int8_t         depths[OV_DERIVED_SLOTS][OV_MAX_NODES];
    static int            next;
    int                   hit;
    int                   i = derived_slot(keys, &next, m, root, (int) mode, &hit);
    if (!hit)
    {
        sg_compute_node_depths(m, root, mode, depths[i]); /* fills every node, 127 when unrelated */
    }
    return depths[i];
}

/** ov_lineage - ancestors and descendants of stream @si. */
const SG_LINEAGE *ov_lineage(const OV_MODEL *m, int si, sg_mode_t mode)
{
    static OV_DERIVED_KEY keys[OV_DERIVED_SLOTS];
    static SG_LINEAGE     lin[OV_DERIVED_SLOTS];
    static int            next;
    int                   hit;
    int                   i = derived_slot(keys, &next, m, si, (int) mode, &hit);
    if (!hit)
    {
        sg_compute_lineage(m, si, mode, &lin[i]);
    }
    return &lin[i];
}

/** ov_derived_build_count - walks computed so far (cache misses), for tests. */
unsigned ov_derived_build_count(void)
{
    return s_builds;
}
