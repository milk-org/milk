// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_relate.c
 * @brief   Cross-panel relationship computation and filtered indices for milk-CTRL.
 */

#include "overview_render_internal.h"
#include <string.h>

/**
 * ov_filter_streams - filter stream indices by regex, freeze relations, and loop isolation.
 * @lay:      Pointer to layout structure.
 * @m:        Current model snapshot.
 * @rel:      Related items bitset (optional, used in freeze mode).
 * @fidx:     Output array for filtered model stream indices.
 * @max_fidx: Maximum capacity of fidx.
 *
 * Return: Number of matching stream indices written to fidx.
 */
int ov_filter_streams(const OV_LAYOUT  *lay,
                      const OV_MODEL   *m,
                      const OV_RELATED *rel,
                      int              *fidx,
                      int               max_fidx)
{
    if (lay == NULL || m == NULL || m->nb_streams <= 0 || fidx == NULL || max_fidx <= 0)
    {
        return 0;
    }

    const char *names[OV_MAX_STREAMS];
    int         total = (m->nb_streams < max_fidx) ? m->nb_streams : max_fidx;
    for (int i = 0; i < m->nb_streams; i++)
    {
        names[i] = m->streams[i].name;
    }

    const char *filt   = ov_get_active_filter_for(lay, OV_FOCUS_STREAMS);
    int         filt_n = 0;
    if (filt != NULL && filt[0] != '\0')
    {
        filt_n = ov_filter_build(filt, names, m->nb_streams, fidx, max_fidx);
    }
    else
    {
        filt_n = total;
        for (int i = 0; i < filt_n; i++)
        {
            fidx[i] = i;
        }
    }

    if (lay->freeze && lay->freeze_focus != OV_FOCUS_STREAMS && rel != NULL)
    {
        int new_filt_n = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (bget(rel->streams, fidx[i]))
            {
                fidx[new_filt_n++] = fidx[i];
            }
        }
        filt_n = new_filt_n;
    }

    if (lay->loop_filter_active && lay->sel_loop >= 0 && lay->sel_loop < m->nb_loops)
    {
        uint32_t active_mask = (UINT32_C(1) << lay->sel_loop);
        int      new_filt_n  = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (m->streams[fidx[i]].loop_mask & active_mask)
            {
                fidx[new_filt_n++] = fidx[i];
            }
        }
        filt_n = new_filt_n;
    }

    return filt_n;
}

/**
 * ov_filter_procs - filter process indices by regex, freeze relations, and loop isolation.
 * @lay:      Pointer to layout structure.
 * @m:        Current model snapshot.
 * @rel:      Related items bitset (optional, used in freeze mode).
 * @fidx:     Output array for filtered model process indices.
 * @max_fidx: Maximum capacity of fidx.
 *
 * Return: Number of matching process indices written to fidx.
 */
int ov_filter_procs(const OV_LAYOUT  *lay,
                    const OV_MODEL   *m,
                    const OV_RELATED *rel,
                    int              *fidx,
                    int               max_fidx)
{
    if (lay == NULL || m == NULL || m->nb_procs <= 0 || fidx == NULL || max_fidx <= 0)
    {
        return 0;
    }

    const char *names[OV_MAX_PROCS];
    int         total = (m->nb_procs < max_fidx) ? m->nb_procs : max_fidx;
    for (int i = 0; i < m->nb_procs; i++)
    {
        names[i] = m->procs[i].name;
    }

    const char *filt   = ov_get_active_filter_for(lay, OV_FOCUS_PROCS);
    int         filt_n = 0;
    if (filt != NULL && filt[0] != '\0')
    {
        filt_n = ov_filter_build(filt, names, m->nb_procs, fidx, max_fidx);
    }
    else
    {
        filt_n = total;
        for (int i = 0; i < filt_n; i++)
        {
            fidx[i] = i;
        }
    }

    if (lay->freeze && lay->freeze_focus != OV_FOCUS_PROCS && rel != NULL)
    {
        int new_filt_n = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (bget(rel->procs, fidx[i]))
            {
                fidx[new_filt_n++] = fidx[i];
            }
        }
        filt_n = new_filt_n;
    }

    if (lay->loop_filter_active && lay->sel_loop >= 0 && lay->sel_loop < m->nb_loops)
    {
        uint32_t active_mask = (UINT32_C(1) << lay->sel_loop);
        int      new_filt_n  = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (m->procs[fidx[i]].loop_mask & active_mask)
            {
                fidx[new_filt_n++] = fidx[i];
            }
        }
        filt_n = new_filt_n;
    }

    return filt_n;
}

/**
 * ov_filter_fps - filter FPS indices by regex, freeze relations, and loop isolation.
 * @lay:      Pointer to layout structure.
 * @m:        Current model snapshot.
 * @rel:      Related items bitset (optional, used in freeze mode).
 * @fidx:     Output array for filtered model FPS indices.
 * @max_fidx: Maximum capacity of fidx.
 *
 * Return: Number of matching FPS indices written to fidx.
 */
int ov_filter_fps(const OV_LAYOUT  *lay,
                  const OV_MODEL   *m,
                  const OV_RELATED *rel,
                  int              *fidx,
                  int               max_fidx)
{
    if (lay == NULL || m == NULL || m->nb_fps <= 0 || fidx == NULL || max_fidx <= 0)
    {
        return 0;
    }

    const char *names[OV_MAX_FPS];
    int         total = (m->nb_fps < max_fidx) ? m->nb_fps : max_fidx;
    for (int i = 0; i < m->nb_fps; i++)
    {
        names[i] = m->fps[i].name;
    }

    const char *filt   = ov_get_active_filter_for(lay, OV_FOCUS_FPS);
    int         filt_n = 0;
    if (filt != NULL && filt[0] != '\0')
    {
        filt_n = ov_filter_build(filt, names, m->nb_fps, fidx, max_fidx);
    }
    else
    {
        filt_n = total;
        for (int i = 0; i < filt_n; i++)
        {
            fidx[i] = i;
        }
    }

    if (lay->freeze && lay->freeze_focus != OV_FOCUS_FPS && rel != NULL)
    {
        int new_filt_n = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (bget(rel->fps, fidx[i]))
            {
                fidx[new_filt_n++] = fidx[i];
            }
        }
        filt_n = new_filt_n;
    }

    if (lay->loop_filter_active && lay->sel_loop >= 0 && lay->sel_loop < m->nb_loops)
    {
        uint32_t active_mask = (UINT32_C(1) << lay->sel_loop);
        int      new_filt_n  = 0;
        for (int i = 0; i < filt_n; i++)
        {
            if (m->fps[fidx[i]].loop_mask & active_mask)
            {
                fidx[new_filt_n++] = fidx[i];
            }
        }
        filt_n = new_filt_n;
    }

    return filt_n;
}

/**
 * ov_get_selected_stream_idx - resolve selected stream row to index in OV_MODEL.
 * @lay: Pointer to layout structure.
 * @m:   Current model snapshot.
 *
 * Return: Model stream index in 0..nb_streams-1, or -1 if none.
 */
int ov_get_selected_stream_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (lay == NULL || m == NULL || m->nb_streams <= 0)
    {
        return -1;
    }
    int ssel = lay->freeze ? lay->freeze_sel_stream : lay->sel_stream;
    if (ssel < 0)
    {
        return -1;
    }
    int fidx[OV_MAX_STREAMS];
    int n = ov_filter_streams(lay, m, NULL, fidx, OV_MAX_STREAMS);
    if (ssel < n)
    {
        return fidx[ssel];
    }
    return -1;
}

/**
 * ov_get_selected_proc_idx - resolve selected process row to index in OV_MODEL.
 * @lay: Pointer to layout structure.
 * @m:   Current model snapshot.
 *
 * Return: Model process index in 0..nb_procs-1, or -1 if none.
 */
int ov_get_selected_proc_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (lay == NULL || m == NULL || m->nb_procs <= 0)
    {
        return -1;
    }
    int psel = lay->freeze ? lay->freeze_sel_proc : lay->sel_proc;
    if (psel < 0)
    {
        return -1;
    }
    int fidx[OV_MAX_PROCS];
    int n = ov_filter_procs(lay, m, NULL, fidx, OV_MAX_PROCS);
    if (psel < n)
    {
        return fidx[psel];
    }
    return -1;
}

/**
 * ov_get_selected_fps_idx - resolve selected FPS row to index in OV_MODEL.
 * @lay: Pointer to layout structure.
 * @m:   Current model snapshot.
 *
 * Return: Model FPS index in 0..nb_fps-1, or -1 if none.
 */
int ov_get_selected_fps_idx(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (lay == NULL || m == NULL || m->nb_fps <= 0)
    {
        return -1;
    }
    int fsel = lay->freeze ? lay->freeze_sel_fps : lay->sel_fps;
    if (fsel < 0)
    {
        return -1;
    }
    int fidx[OV_MAX_FPS];
    int n = ov_filter_fps(lay, m, NULL, fidx, OV_MAX_FPS);
    if (fsel < n)
    {
        return fidx[fsel];
    }
    return -1;
}

/**
 * bset - set bit at index in 64-bit word bitset array.
 * @words: Pointer to 64-bit integer bitset words.
 * @idx:   0-based bit index to set.
 */
void bset(uint64_t *words, int idx)
{
    words[idx / BITS_PER_WORD] |= (UINT64_C(1) << (idx % BITS_PER_WORD));
}

/**
 * bget - test bit at index in 64-bit word bitset array.
 * @words: Pointer to 64-bit integer bitset words.
 * @idx:   0-based bit index to test.
 *
 * Return: 1 if bit is set, 0 otherwise.
 */
int bget(const uint64_t *words, int idx)
{
    return (words[idx / BITS_PER_WORD] >> (idx % BITS_PER_WORD)) & 1;
}

/**
 * ov_compute_related - identify connected items across panels for active selection.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 * @out: Output structure populated with highlight bitsets and selected IDs.
 */
void ov_compute_related(const OV_LAYOUT *lay, const OV_MODEL *m, OV_RELATED *out)
{
    memset(out, 0, sizeof(*out));

    ov_focus_t focus = lay->freeze ? lay->freeze_focus : lay->focus;
    if (lay->mouse_hover && lay->hover_idx >= 0 && lay->hover_view != -1)
    {
        focus = lay->hover_view;
    }

    int sel_stream_idx = (lay->mouse_hover && lay->hover_global_stream >= 0)
                             ? lay->hover_global_stream
                             : ov_get_selected_stream_idx(lay, m);

    /* Determine the graph node index of the selected item */
    int sel_node = -1;
    if (focus == OV_FOCUS_STREAMS)
    {
        if (sel_stream_idx >= 0 && sel_stream_idx < m->nb_streams)
        {
            sel_node = m->streams[sel_stream_idx].node_idx;
        }
    }
    else if (focus == OV_FOCUS_FPS)
    {
        int model_idx = (lay->mouse_hover && lay->hover_global_fps >= 0)
                            ? lay->hover_global_fps
                            : ov_get_selected_fps_idx(lay, m);
        if (model_idx >= 0 && model_idx < m->nb_fps)
        {
            sel_node = m->fps[model_idx].node_idx;
        }
    }
    else if (focus == OV_FOCUS_PROCS)
    {
        int model_idx = (lay->mouse_hover && lay->hover_global_proc >= 0)
                            ? lay->hover_global_proc
                            : ov_get_selected_proc_idx(lay, m);
        if (model_idx >= 0 && model_idx < m->nb_procs)
        {
            sel_node     = m->procs[model_idx].node_idx;
            out->sel_pid = m->procs[model_idx].PID;
        }
    }

    if (sel_node < 0)
    {
        return;
    }

    /* Walk all edges; mark neighbours of sel_node */
    for (int ei = 0; ei < m->nb_edges; ei++)
    {
        const OV_EDGE *e          = &m->edges[ei];
        int            other      = -1;
        int            is_write   = 0; /* 1 = proc writes stream */
        int            fps_is_src = 0;

        if (e->src_node == sel_node)
        {
            other = e->tgt_node;
            is_write =
                (e->type == OV_EDGE_PROC_WRITES_STREAM) || (e->type == OV_EDGE_FPS_OUTPUT_STREAM);
            fps_is_src = 0;
        }
        else if (e->tgt_node == sel_node)
        {
            other = e->src_node;
            is_write =
                (e->type == OV_EDGE_PROC_WRITES_STREAM) || (e->type == OV_EDGE_FPS_OUTPUT_STREAM);
            fps_is_src = 1;
        }

        if (other < 0 || other >= m->nb_nodes)
        {
            continue;
        }

        const OV_NODE *n = &m->nodes[other];
        if (n->type == OV_NODE_STREAM && n->index >= 0 && n->index < m->nb_streams)
        {
            bset(out->streams, n->index);
            if (is_write)
            {
                bset(out->stream_written, n->index);
            }
        }
        else if (n->type == OV_NODE_FPS && n->index >= 0 && n->index < m->nb_fps)
        {
            int fi = n->index;
            bset(out->fps, fi);
            if (is_write)
            {
                bset(out->fps_writes, fi);
            }

            /* Find all stream params of this FPS that match sel_node */
            if (focus == OV_FOCUS_STREAMS && sel_stream_idx >= 0 && sel_stream_idx < m->nb_streams)
            {
                const char   *sname = m->streams[sel_stream_idx].name;
                const OV_FPS *f     = &m->fps[fi];
                for (int sp = 0; sp < f->nb_stream_params; sp++)
                {
                    if (strcmp(f->stream_param_value[sp], sname) == 0)
                    {
                        out->fps_param_mask[fi] |= (UINT32_C(1) << sp);
                    }
                }
            }

            (void) fps_is_src;
        }
        else if (n->type == OV_NODE_PROC && n->index >= 0 && n->index < m->nb_procs)
        {
            bset(out->procs, n->index);
            if (is_write)
            {
                bset(out->proc_writes, n->index);
            }
        }
    }
}
