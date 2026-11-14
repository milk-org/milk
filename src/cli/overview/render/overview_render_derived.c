// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_derived.c
 * @brief   Results of graph walks (node depths, stream lineage), cached per model, root and mode.
 *
 * Purpose:     the list panels and stream DETAILS ask for the same graph walks at every frame,
 *              and a frame is drawn at every key; a walk is recomputed only when the model (its
 *              publication number), the root or the mode changes.
 * Does:        ov_node_depths() (signed lineage distance of every node from a root, 127 when
 *              unrelated) and ov_lineage() (the ancestors and descendants of a stream) each keep
 *              their last 4 results, replaced in turn. A model never published (seq 0, e.g. built
 *              by a test) is not cached.
 * Used by:     the ancestry of the processes and FPS lists (render/overview_render_procs.c,
 *              overview_render_fps.c: ov_node_depths); the streams list and stream DETAILS
 *              (render/overview_render_streams.c, overview_render_detail_stream.c: ov_lineage).
 *              Declared in render/overview_render_internal.h.
 * Uses:        stream_graph.h (sg_compute_node_depths, sg_compute_lineage).
 * Constraints: UI thread only (static caches); a later call can overwrite a returned result.
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
