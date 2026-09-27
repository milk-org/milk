// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_sync.c
 * @brief   Selection bounds clamping and node synchronization for milk-CTRL.
 */

#include "overview_render_internal.h"
#include "overview_render_fps_params.h"
#include "overview_render_loops.h"
#include "overview_data_internal.h"
#include <stdio.h>
#include <string.h>

static const OV_MODEL *g_last_model = NULL;

/**
 * ov_render__sync_selection - clamp selections, handle sort freezing, and sync node indices.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 */
void ov_render__sync_selection(OV_LAYOUT *lay, const OV_MODEL *m)
{
    /* Ensure there exists a valid selected parameter when in the PARAMS panel on F5 view */
    int cur_fidx = ov_get_selected_fps_idx(lay, m);
    if (lay->view == OV_VIEW_FPS && cur_fidx >= 0 && cur_fidx < m->nb_fps)
    {
        const OV_FPS   *fps = &m->fps[cur_fidx];
        fps_tree_item_t items[1024];
        int             nitems = ov_get_fps_tree_items(fps, lay->fps_param_path, items, 1024);

        if (nitems > 0)
        {
            if (lay->fps_param_sel < 0)
            {
                lay->fps_param_sel = 0;
            }
            else if (lay->fps_param_sel >= nitems)
            {
                lay->fps_param_sel = nitems - 1;
            }
        }
    }

    /* One-shot sort: only runs once when the user
     * presses S or s.  Order stays frozen until
     * the user explicitly presses S/s again. */
    if (lay->sort_pending)
    {
        OV_MODEL *mm = (OV_MODEL *) (uintptr_t) m;

        /* Calculate ancestry depths before sorting */
        int8_t depths[OV_MAX_NODES];
        for (int i = 0; i < OV_MAX_NODES; i++)
        {
            depths[i] = 127;
        }

        int        sel_node       = -1;
        ov_focus_t focus          = lay->freeze ? lay->freeze_focus : lay->focus;
        int        sel_stream_idx = lay->freeze ? lay->freeze_sel_stream : lay->sel_stream;
        int        sel_proc_idx   = lay->freeze ? lay->freeze_sel_proc : lay->sel_proc;
        int        sel_fps_idx    = lay->freeze ? lay->freeze_sel_fps : lay->sel_fps;

        char saved_sel_stream[80] = { 0 };
        char saved_sel_proc[80]   = { 0 };
        char saved_sel_fps[80]    = { 0 };

        {
            const char *names[OV_MAX_NODES];
            int         fidx[OV_MAX_NODES];
            const char *f_str = ov_get_active_filter_for(lay, OV_FOCUS_STREAMS);
            const char *f_prc = ov_get_active_filter_for(lay, OV_FOCUS_PROCS);
            const char *f_fps = ov_get_active_filter_for(lay, OV_FOCUS_FPS);

            /* Streams */
            for (int i = 0; i < mm->nb_streams; i++)
            {
                names[i] = mm->streams[i].name;
            }
            int fn = ov_filter_build(f_str, names, mm->nb_streams, fidx, OV_MAX_NODES);
            if (lay->sel_stream >= 0 && lay->sel_stream < fn)
            {
                strncpy(saved_sel_stream, mm->streams[fidx[lay->sel_stream]].name, 79);
            }
            if (focus == OV_FOCUS_STREAMS && sel_stream_idx >= 0 && sel_stream_idx < fn)
            {
                sel_node = mm->streams[fidx[sel_stream_idx]].node_idx;
            }

            /* Procs */
            for (int i = 0; i < mm->nb_procs; i++)
            {
                names[i] = mm->procs[i].name;
            }
            fn = ov_filter_build(f_prc, names, mm->nb_procs, fidx, OV_MAX_NODES);
            if (lay->sel_proc >= 0 && lay->sel_proc < fn)
            {
                strncpy(saved_sel_proc, mm->procs[fidx[lay->sel_proc]].name, 79);
            }
            if (focus == OV_FOCUS_PROCS && sel_proc_idx >= 0 && sel_proc_idx < fn)
            {
                sel_node = mm->procs[fidx[sel_proc_idx]].node_idx;
            }

            /* FPS */
            for (int i = 0; i < mm->nb_fps; i++)
            {
                names[i] = mm->fps[i].name;
            }
            fn = ov_filter_build(f_fps, names, mm->nb_fps, fidx, OV_MAX_NODES);
            if (lay->sel_fps >= 0 && lay->sel_fps < fn)
            {
                strncpy(saved_sel_fps, mm->fps[fidx[lay->sel_fps]].name, 79);
            }
            if (focus == OV_FOCUS_FPS && sel_fps_idx >= 0 && sel_fps_idx < fn)
            {
                sel_node = mm->fps[fidx[sel_fps_idx]].node_idx;
            }
        }

        if (sel_node >= 0)
        {
            sg_mode_t smode = (focus == OV_FOCUS_FPS) ? SG_MODE_FPS : SG_MODE_FULL;
            sg_compute_node_depths(mm, sel_node, smode, depths);
        }
        ov_sort_set_depths(depths);

        ov_sort_streams(mm, lay->sort_key_stream, lay->sort_dir_stream);
        ov_sort_procs(mm, lay->sort_key_proc, lay->sort_dir_proc);
        ov_sort_fps(mm, lay->sort_key_fps, lay->sort_dir_fps);

        ov_sort_freeze_snapshot(mm);

        {
            const char *names[OV_MAX_NODES];
            int         fidx[OV_MAX_NODES];
            const char *f_str = ov_get_active_filter_for(lay, OV_FOCUS_STREAMS);
            const char *f_prc = ov_get_active_filter_for(lay, OV_FOCUS_PROCS);
            const char *f_fps = ov_get_active_filter_for(lay, OV_FOCUS_FPS);

            if (saved_sel_stream[0] != '\0')
            {
                for (int i = 0; i < mm->nb_streams; i++)
                {
                    names[i] = mm->streams[i].name;
                }
                int fn = ov_filter_build(f_str, names, mm->nb_streams, fidx, OV_MAX_NODES);
                for (int i = 0; i < fn; i++)
                {
                    if (strcmp(saved_sel_stream, mm->streams[fidx[i]].name) == 0)
                    {
                        lay->sel_stream = i;
                        int page_h      = lay->r_streams.height - 3;
                        if (page_h > 0)
                        {
                            if (lay->sel_stream < lay->scroll_stream)
                            {
                                lay->scroll_stream = lay->sel_stream;
                            }
                            if (lay->sel_stream >= lay->scroll_stream + page_h)
                            {
                                lay->scroll_stream = lay->sel_stream - page_h + 1;
                            }
                        }
                        break;
                    }
                }
            }

            if (saved_sel_proc[0] != '\0')
            {
                for (int i = 0; i < mm->nb_procs; i++)
                {
                    names[i] = mm->procs[i].name;
                }
                int fn = ov_filter_build(f_prc, names, mm->nb_procs, fidx, OV_MAX_NODES);
                for (int i = 0; i < fn; i++)
                {
                    if (strcmp(saved_sel_proc, mm->procs[fidx[i]].name) == 0)
                    {
                        lay->sel_proc = i;
                        int page_h    = lay->r_procs.height - 3;
                        if (page_h > 0)
                        {
                            if (lay->sel_proc < lay->scroll_proc)
                            {
                                lay->scroll_proc = lay->sel_proc;
                            }
                            if (lay->sel_proc >= lay->scroll_proc + page_h)
                            {
                                lay->scroll_proc = lay->sel_proc - page_h + 1;
                            }
                        }
                        break;
                    }
                }
            }

            if (saved_sel_fps[0] != '\0')
            {
                for (int i = 0; i < mm->nb_fps; i++)
                {
                    names[i] = mm->fps[i].name;
                }
                int fn = ov_filter_build(f_fps, names, mm->nb_fps, fidx, OV_MAX_NODES);
                for (int i = 0; i < fn; i++)
                {
                    if (strcmp(saved_sel_fps, mm->fps[fidx[i]].name) == 0)
                    {
                        lay->sel_fps = i;
                        int page_h   = lay->r_fps.height - 3;
                        if (page_h > 0)
                        {
                            if (lay->sel_fps < lay->scroll_fps)
                            {
                                lay->scroll_fps = lay->sel_fps;
                            }
                            if (lay->sel_fps >= lay->scroll_fps + page_h)
                            {
                                lay->scroll_fps = lay->sel_fps - page_h + 1;
                            }
                        }
                        break;
                    }
                }
            }
        }

        lay->sort_pending = 0;
        g_last_model      = m;
    }
    else
    {
        /* A new scan model arrived. Re-apply the saved order so items don't shuffle. */
        OV_MODEL *mm = (OV_MODEL *) (uintptr_t) m;
        ov_sort_apply_ranks(mm);
        g_last_model = m;
    }

    /* Enforce active selection tracking:
     * If the selected item no longer exists in the filtered list
     * (e.g. removed by an external process), reset the selection to 0.
     * Otherwise, clamp bounds and update the tracked name. */
    {
        const char *names[OV_MAX_NODES];
        int         fidx[OV_MAX_NODES];
        const char *f_str = ov_get_active_filter_for(lay, OV_FOCUS_STREAMS);
        const char *f_prc = ov_get_active_filter_for(lay, OV_FOCUS_PROCS);
        const char *f_fps = ov_get_active_filter_for(lay, OV_FOCUS_FPS);

        /* Streams */
        for (int i = 0; i < m->nb_streams; i++)
        {
            names[i] = m->streams[i].name;
        }
        int fn = ov_filter_build(f_str, names, m->nb_streams, fidx, OV_MAX_NODES);
        if (fn > 0)
        {
            if (lay->sel_name_stream[0] != '\0')
            {
                int still_exists = 0;
                for (int i = 0; i < fn; i++)
                {
                    if (strcmp(m->streams[fidx[i]].name, lay->sel_name_stream) == 0)
                    {
                        still_exists    = 1;
                        lay->sel_stream = i;
                        int page_h      = lay->r_streams.height - 3;
                        if (page_h > 0)
                        {
                            if (lay->sel_stream < lay->scroll_stream)
                            {
                                lay->scroll_stream = lay->sel_stream;
                            }
                            if (lay->sel_stream >= lay->scroll_stream + page_h)
                            {
                                lay->scroll_stream = lay->sel_stream - page_h + 1;
                            }
                        }
                        break;
                    }
                }
                if (!still_exists)
                {
                    lay->sel_stream = 0;
                }
            }
            if (lay->sel_stream >= fn)
            {
                lay->sel_stream = fn - 1;
            }
            if (lay->sel_stream < 0)
            {
                lay->sel_stream = 0;
            }
            strncpy(lay->sel_name_stream, m->streams[fidx[lay->sel_stream]].name, 79);
        }
        else
        {
            lay->sel_stream         = 0;
            lay->sel_name_stream[0] = '\0';
        }

        /* Procs */
        for (int i = 0; i < m->nb_procs; i++)
        {
            names[i] = m->procs[i].name;
        }
        fn = ov_filter_build(f_prc, names, m->nb_procs, fidx, OV_MAX_NODES);
        if (fn > 0)
        {
            if (lay->sel_name_proc[0] != '\0')
            {
                int still_exists = 0;
                for (int i = 0; i < fn; i++)
                {
                    if (strcmp(m->procs[fidx[i]].name, lay->sel_name_proc) == 0 &&
                        m->procs[fidx[i]].PID == lay->sel_pid_proc)
                    {
                        still_exists  = 1;
                        lay->sel_proc = i;
                        int page_h    = lay->r_procs.height - 3;
                        if (page_h > 0)
                        {
                            if (lay->sel_proc < lay->scroll_proc)
                            {
                                lay->scroll_proc = lay->sel_proc;
                            }
                            if (lay->sel_proc >= lay->scroll_proc + page_h)
                            {
                                lay->scroll_proc = lay->sel_proc - page_h + 1;
                            }
                        }
                        break;
                    }
                }
                if (!still_exists)
                {
                    lay->sel_proc = 0;
                }
            }
            if (lay->sel_proc >= fn)
            {
                lay->sel_proc = fn - 1;
            }
            if (lay->sel_proc < 0)
            {
                lay->sel_proc = 0;
            }
            strncpy(lay->sel_name_proc, m->procs[fidx[lay->sel_proc]].name, 79);
            lay->sel_pid_proc = m->procs[fidx[lay->sel_proc]].PID;
        }
        else
        {
            lay->sel_proc         = 0;
            lay->sel_name_proc[0] = '\0';
            lay->sel_pid_proc     = 0;
        }

        /* FPS */
        for (int i = 0; i < m->nb_fps; i++)
        {
            names[i] = m->fps[i].name;
        }
        fn = ov_filter_build(f_fps, names, m->nb_fps, fidx, OV_MAX_NODES);
        if (fn > 0)
        {
            if (lay->sel_name_fps[0] != '\0')
            {
                int still_exists = 0;
                for (int i = 0; i < fn; i++)
                {
                    if (strcmp(m->fps[fidx[i]].name, lay->sel_name_fps) == 0)
                    {
                        still_exists = 1;
                        lay->sel_fps = i;
                        int page_h   = lay->r_fps.height - 3;
                        if (page_h > 0)
                        {
                            if (lay->sel_fps < lay->scroll_fps)
                            {
                                lay->scroll_fps = lay->sel_fps;
                            }
                            if (lay->sel_fps >= lay->scroll_fps + page_h)
                            {
                                lay->scroll_fps = lay->sel_fps - page_h + 1;
                            }
                        }
                        break;
                    }
                }
                if (!still_exists)
                {
                    lay->sel_fps = 0;
                }
            }
            if (lay->sel_fps >= fn)
            {
                lay->sel_fps = fn - 1;
            }
            if (lay->sel_fps < 0)
            {
                lay->sel_fps = 0;
            }
            strncpy(lay->sel_name_fps, m->fps[fidx[lay->sel_fps]].name, 79);
        }
        else
        {
            lay->sel_fps         = 0;
            lay->sel_name_fps[0] = '\0';
        }
    }

    /* Always patch graph node .index fields to
     * reflect current array positions — needed
     * whether we just sorted or scan rebuilt the
     * model.  Each item's node_idx still points
     * to its graph node; update the reverse link. */
    {
        OV_MODEL *mm = (OV_MODEL *) (uintptr_t) m;
        for (int i = 0; i < mm->nb_streams; i++)
        {
            int ni = mm->streams[i].node_idx;
            if (ni >= 0 && ni < mm->nb_nodes)
            {
                mm->nodes[ni].index = i;
            }
        }
        for (int i = 0; i < mm->nb_fps; i++)
        {
            int ni = mm->fps[i].node_idx;
            if (ni >= 0 && ni < mm->nb_nodes)
            {
                mm->nodes[ni].index = i;
            }
        }
        for (int i = 0; i < mm->nb_procs; i++)
        {
            int ni = mm->procs[i].node_idx;
            if (ni >= 0 && ni < mm->nb_nodes)
            {
                mm->nodes[ni].index = i;
            }
        }
    }
}
