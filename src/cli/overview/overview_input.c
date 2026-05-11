/**
 * @file overview_input.c
 * @brief Keyboard input handler for milkCTRL
 */

#include <stdlib.h>
#include <string.h>

#include "overview_defs.h"
#include "overview_ansi.h"
#include "overview_data.h"
#include "overview_layout.h"
#include "overview_ctrl.h"
#include "stream_graph.h"

static int get_graph_start_node(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    ov_focus_t eff_focus = lay->freeze ? lay->freeze_focus : lay->focus;
    int target_type = -1;
    int target_idx = -1;
    
    if (eff_focus == OV_FOCUS_STREAMS || eff_focus == OV_FOCUS_GRAPH) {
        target_type = OV_NODE_STREAM;
        target_idx = lay->freeze ? lay->freeze_sel_stream : lay->sel_stream;
    } else if (eff_focus == OV_FOCUS_PROCS) {
        target_type = OV_NODE_PROC;
        target_idx = lay->freeze ? lay->freeze_sel_proc : lay->sel_proc;
    } else if (eff_focus == OV_FOCUS_FPS) {
        target_type = OV_NODE_FPS;
        target_idx = lay->freeze ? lay->freeze_sel_fps : lay->sel_fps;
    }

    if (target_type != -1 && target_idx != -1) {
        for (int i = 0; i < m->nb_nodes; i++) {
            if (m->nodes[i].type == target_type && m->nodes[i].index == target_idx) {
                return i;
            }
        }
    }
    return -1;
}


/* scan API */
extern float ov_scan_get_interval(void);
extern void  ov_scan_set_interval(float s);

/* help panel utilities (overview_render_help.c) */
extern int ov_help_visible_count(const OV_LAYOUT *lay);
extern int ov_help_nb_sections(void);
extern int ov_help_toggle_at(
    OV_LAYOUT *lay, int vis_row);

/**
 * ov_handle_key - process one key event.
 * @key: keycode from ov_get_key()
 * @lay: mutable layout state
 * @m:   current model (read-only)
 *
 * Return: 0 to continue, 1 to quit.
 */


static int ov_input__handle_filter_mode(int key, OV_LAYOUT *lay)
{
    char *active_filter = NULL;
    switch (lay->focus)
    {
    case OV_FOCUS_STREAMS: active_filter = lay->filter_stream; break;
    case OV_FOCUS_PROCS:   active_filter = lay->filter_proc;   break;
    case OV_FOCUS_FPS:     active_filter = lay->filter_fps;    break;
    default: break;
    }

    if (lay->filter_editing)
    {
        if (active_filter == NULL)
        {
            lay->filter_editing = 0;
            return 1;
        }

        /* ESC — cancel filter edit, restore empty */
        if (key == 27)
        {
            active_filter[0] = '\0';
            lay->filter_cursor = 0;
            lay->filter_editing = 0;
            lay->sel_stream = 0;
            lay->scroll_stream = 0;
            lay->sel_proc = 0;
            lay->scroll_proc = 0;
            lay->sel_fps = 0;
            lay->scroll_fps = 0;
            return 1;
        }

        /* ENTER — accept filter */
        if (key == '\n' || key == '\r')
        {
            lay->filter_editing = 0;
            switch (lay->focus)
            {
            case OV_FOCUS_STREAMS: lay->sel_stream = 0; lay->scroll_stream = 0; break;
            case OV_FOCUS_PROCS:   lay->sel_proc = 0;   lay->scroll_proc = 0;   break;
            case OV_FOCUS_FPS:     lay->sel_fps = 0;    lay->scroll_fps = 0;    break;
            default: break;
            }
            return 1;
        }

        /* Backspace — delete last char */
        if (key == 127 || key == 8)
        {
            if (lay->filter_cursor > 0)
            {
                lay->filter_cursor--;
                active_filter[lay->filter_cursor] = '\0';
            }
            return 1;
        }

        /* Printable ASCII — append to filter */
        if (key >= 32 && key < 127 && lay->filter_cursor < 62)
        {
            active_filter[lay->filter_cursor] = (char) key;
            lay->filter_cursor++;
            active_filter[lay->filter_cursor] = '\0';
            return 1;
        }

        return 1; /* Ignore other keys during editing */
    }

    /* '/' — enter filter editing mode */
    if (key == '/' && active_filter != NULL)
    {
        lay->filter_editing = 1;
        active_filter[0] = '\0';
        lay->filter_cursor = 0;
        return 1;
    }

    /* ESC — clear active filter on focused panel */
    if (key == 27 && active_filter != NULL && active_filter[0] != '\0')
    {
        active_filter[0] = '\0';
        lay->sel_stream = 0;
        lay->scroll_stream = 0;
        lay->sel_proc = 0;
        lay->scroll_proc = 0;
        lay->sel_fps = 0;
        lay->scroll_fps = 0;
        return 1;
    }

    return 0;
}

#define INSIDE(R, MR, MC) \
    ((MR) >= (R).row && (MR) < (R).row + (R).height && \
     (MC) >= (R).col && (MC) < (R).col + (R).width)

static int ov_input__handle_mouse(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (key == OV_KEY_MOUSE_CLICK)
    {
        int mr = ov_mouse_row;
        int mc = ov_mouse_col;

        static struct timespec last_click_ts = {0,0};
        static int last_click_r = -1;
        static int last_click_c = -1;
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        int is_dbl = 0;
        if (mr == last_click_r && mc == last_click_c)
        {
            double dt = (now.tv_sec - last_click_ts.tv_sec) +
                        (now.tv_nsec - last_click_ts.tv_nsec) / 1e9;
            if (dt < 0.3)
            {
                is_dbl = 1;
                last_click_r = -1;
            }
        }
        if (!is_dbl)
        {
            last_click_ts = now;
            last_click_r = mr;
            last_click_c = mc;
        }

        /* Check for header tab clicks */
        if (mr == lay->r_header.row && mc >= 1)
        {
            int tab_widths[] = {9, 10, 9, 9, 8};
            int tabs_total_width = 0;
            for (int v = 0; v < OV_VIEW_COUNT; v++) tabs_total_width += tab_widths[v];
            
            int x = lay->term_cols - tabs_total_width + 1;
            if (mc >= x)
            {
                for (int v = 0; v < OV_VIEW_COUNT; v++)
                {
                    if (mc >= x && mc < x + tab_widths[v])
                    {
                        lay->view = (ov_view_t)v;
                        break;
                    }
                    x += tab_widths[v];
                }
            }
        }

        if (INSIDE(lay->r_streams, mr, mc))
        {
            lay->focus = OV_FOCUS_STREAMS;
            int body_row = mr - lay->r_streams.row - 2;
            if (body_row >= 0)
            {
                int idx = lay->scroll_stream + body_row;
                if (idx < m->nb_streams)
                {
                    lay->sel_stream = idx;
                    if (is_dbl) lay->detail_mode = 1;
                }
            }
        }
        else if (INSIDE(lay->r_procs, mr, mc))
        {
            lay->focus = OV_FOCUS_PROCS;
            int body_row = mr - lay->r_procs.row - 2;
            if (body_row >= 0)
            {
                int idx = lay->scroll_proc + body_row;
                if (idx < m->nb_procs)
                {
                    lay->sel_proc = idx;
                    if (is_dbl) lay->detail_mode = 1;
                }
            }
        }
        else if (INSIDE(lay->r_fps, mr, mc))
        {
            lay->focus = OV_FOCUS_FPS;
            int body_row = mr - lay->r_fps.row - 2;
            if (body_row >= 0)
            {
                int idx = lay->scroll_fps + body_row;
                if (idx < m->nb_fps)
                {
                    lay->sel_fps = idx;
                    if (is_dbl) lay->detail_mode = 1;
                }
            }
        }
        else if (INSIDE(lay->r_graph, mr, mc))
        {
            lay->focus = OV_FOCUS_GRAPH;
            int body_row = mr - lay->r_graph.row - 2;
            if (body_row >= 0)
            {
                int idx = lay->scroll_graph + body_row;
                if (idx < m->nb_edges)
                {
                    lay->sel_graph = idx;
                    if (is_dbl)
                    {
                        const OV_EDGE *edge = &m->edges[idx];
                        const OV_NODE *node = &m->nodes[edge->src_node];
                        if (node->type == OV_NODE_STREAM) { lay->focus = OV_FOCUS_STREAMS; lay->sel_stream = node->index; }
                        else if (node->type == OV_NODE_PROC) { lay->focus = OV_FOCUS_PROCS; lay->sel_proc = node->index; }
                        else if (node->type == OV_NODE_FPS) { lay->focus = OV_FOCUS_FPS; lay->sel_fps = node->index; }
                        lay->view = OV_VIEW_DASHBOARD;
                    }
                }
            }
        }
        return 1;
    }

    if (key == OV_KEY_MOUSE_UP || key == OV_KEY_MOUSE_DOWN)
    {
        int mr = ov_mouse_row;
        int mc = ov_mouse_col;
        int dir = (key == OV_KEY_MOUSE_UP) ? -3 : 3;

        int *sel    = NULL;
        int *scroll = NULL;
        int  count  = 0;
        int  page_h = 10;

        if (INSIDE(lay->r_streams, mr, mc))
        {
            sel    = &lay->sel_stream;
            scroll = &lay->scroll_stream;
            count  = m->nb_streams;
            page_h = lay->r_streams.height - 3;
        }
        else if (INSIDE(lay->r_procs, mr, mc))
        {
            sel    = &lay->sel_proc;
            scroll = &lay->scroll_proc;
            count  = m->nb_procs;
            page_h = lay->r_procs.height - 3;
        }
        else if (INSIDE(lay->r_fps, mr, mc))
        {
            sel    = &lay->sel_fps;
            scroll = &lay->scroll_fps;
            count  = m->nb_fps;
            page_h = lay->r_fps.height - 3;
        }
        else if (INSIDE(lay->r_graph, mr, mc))
        {
            sel    = &lay->sel_graph;
            scroll = &lay->scroll_graph;
            count  = m->nb_edges;
            page_h = lay->r_graph.height - 3;
        }

        if (sel != NULL && scroll != NULL)
        {
            *sel += dir;
            if (*sel < 0) *sel = 0;
            if (*sel >= count) *sel = count - 1;
            if (*sel < 0) *sel = 0;
            if (*sel < *scroll) *scroll = *sel;
            if (page_h > 0 && *sel >= *scroll + page_h) *scroll = *sel - page_h + 1;
        }
        return 1;
    }

    return 0;
}
#undef INSIDE

static int ov_input__handle_view_switch(int key, OV_LAYOUT *lay)
{
    if (key >= OV_KEY_F2 && key <= OV_KEY_F6)
    {
        int vi = key - OV_KEY_F2;
        if (vi < OV_VIEW_COUNT)
        {
            lay->view = (ov_view_t) vi;
            if (vi == OV_VIEW_STREAMS) lay->focus = OV_FOCUS_STREAMS;
            else if (vi == OV_VIEW_PROCS) lay->focus = OV_FOCUS_PROCS;
            else if (vi == OV_VIEW_FPS) lay->focus = OV_FOCUS_FPS;
            else if (vi == OV_VIEW_GRAPH) lay->focus = OV_FOCUS_GRAPH;
        }
        return 1;
    }

    if (key == OV_KEY_CTRL_LEFT)
    {
        int v = (int) lay->view;
        v = (v - 1 + OV_VIEW_COUNT) % OV_VIEW_COUNT;
        lay->view = (ov_view_t) v;
        if (v == OV_VIEW_STREAMS) lay->focus = OV_FOCUS_STREAMS;
        else if (v == OV_VIEW_PROCS) lay->focus = OV_FOCUS_PROCS;
        else if (v == OV_VIEW_FPS) lay->focus = OV_FOCUS_FPS;
        else if (v == OV_VIEW_GRAPH) lay->focus = OV_FOCUS_GRAPH;
        return 1;
    }
    if (key == OV_KEY_CTRL_RIGHT)
    {
        int v = (int) lay->view;
        v = (v + 1) % OV_VIEW_COUNT;
        lay->view = (ov_view_t) v;
        if (v == OV_VIEW_STREAMS) lay->focus = OV_FOCUS_STREAMS;
        else if (v == OV_VIEW_PROCS) lay->focus = OV_FOCUS_PROCS;
        else if (v == OV_VIEW_FPS) lay->focus = OV_FOCUS_FPS;
        else if (v == OV_VIEW_GRAPH) lay->focus = OV_FOCUS_GRAPH;
        return 1;
    }

    if (key == OV_KEY_TAB)
    {
        lay->focus = (ov_focus_t)(((int) lay->focus + 1) % OV_FOCUS_COUNT);
        return 1;
    }

    return 0;
}

static int ov_input__handle_misc_toggles(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (key == '+' || key == '=')
    {
        float cur = ov_scan_get_interval();
        ov_scan_set_interval(cur * 0.7f);
        return 1;
    }
    if (key == '-')
    {
        float cur = ov_scan_get_interval();
        ov_scan_set_interval(cur * 1.4f);
        return 1;
    }
    if (key == 'D')
    {
        lay->detail_mode = !lay->detail_mode;
        return 1;
    }
    if (key == 'L')
    {
        lay->lineage_mode = (lay->lineage_mode + 1) % 3;
        return 1;
    }
    if (key == 'p' || key == 'P')
    {
        lay->paused = !lay->paused;
        ov_cmdlog_push(&lay->cmdlog,
                       OV_CMDLOG_INFO,
                       "Display %s",
                       lay->paused
                       ? "paused" : "resumed");
        return 1;
    }
    if (key == 'W')
    {
        ov_model_export_snapshot(m);
        ov_cmdlog_push(&lay->cmdlog,
                       OV_CMDLOG_OK,
                       "Snapshot exported");
        return 1;
    }

    /* Graph jump on Enter (must be before detail mode toggle) */
    if ((key == '\n' || key == '\r') && lay->focus == OV_FOCUS_GRAPH)
    {
        int start_node = get_graph_start_node(lay, m);
        if (start_node >= 0)
        {
            SG_RENDER_NODE rnodes[OV_MAX_NODES];
            int n_rnodes = sg_compute_render_nodes(m, start_node, lay->lineage_mode, rnodes);
            if (lay->sel_graph < n_rnodes)
            {
                const SG_RENDER_NODE *rn = &rnodes[lay->sel_graph];
                const OV_NODE *node = &m->nodes[rn->node_idx];
                if (node->type == OV_NODE_STREAM) { lay->focus = OV_FOCUS_STREAMS; lay->sel_stream = node->index; }
                else if (node->type == OV_NODE_PROC) { lay->focus = OV_FOCUS_PROCS; lay->sel_proc = node->index; }
                else if (node->type == OV_NODE_FPS) { lay->focus = OV_FOCUS_FPS; lay->sel_fps = node->index; }
                lay->view = OV_VIEW_DASHBOARD;
                lay->freeze = 0;
            }
        }
        return 1;
    }

    /* ENTER — toggle detail mode */
    if ((key == 10 || key == 13) && m != NULL)
    {
        lay->detail_mode = !lay->detail_mode;
        return 1;
    }

    if (key == ' ')
    {
        if (lay->freeze)
        {
            lay->freeze = 0;
        }
        else
        {
            lay->freeze = 1;
            lay->freeze_focus = lay->focus;
            lay->freeze_sel_stream = lay->sel_stream;
            lay->freeze_sel_proc   = lay->sel_proc;
            lay->freeze_sel_fps    = lay->sel_fps;
        }
        return 1;
    }

    if (key == OV_KEY_LEFT || key == OV_KEY_RIGHT)
    {
        if (key == OV_KEY_LEFT)
        {
            if (lay->focus == OV_FOCUS_STREAMS) lay->focus = OV_FOCUS_GRAPH;
            else if (lay->focus == OV_FOCUS_PROCS) lay->focus = OV_FOCUS_STREAMS;
            else if (lay->focus == OV_FOCUS_FPS) lay->focus = OV_FOCUS_PROCS;
            else if (lay->focus == OV_FOCUS_GRAPH)
            {
                if (lay->view == OV_VIEW_DASHBOARD && !lay->detail_mode) lay->focus = OV_FOCUS_FPS;
                else lay->focus = OV_FOCUS_STREAMS;
            }
        }
        else
        {
            if (lay->focus == OV_FOCUS_STREAMS) lay->focus = OV_FOCUS_PROCS;
            else if (lay->focus == OV_FOCUS_PROCS) lay->focus = OV_FOCUS_FPS;
            else if (lay->focus == OV_FOCUS_FPS) lay->focus = OV_FOCUS_GRAPH;
            else if (lay->focus == OV_FOCUS_GRAPH) lay->focus = OV_FOCUS_STREAMS;
        }
        return 1;
    }

    return 0;
}

static int ov_input__handle_sorting(int key, OV_LAYOUT *lay)
{
    if (key == 'S')
    {
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS: lay->sort_key_stream = 3; break;
        case OV_FOCUS_PROCS:   lay->sort_key_proc = 3;   break;
        case OV_FOCUS_FPS:     lay->sort_key_fps = 1;    break;
        default: break;
        }
        lay->sort_pending = 1;
        return 1;
    }

    if (key == 's' && !(lay->ctrl_mode && lay->focus == OV_FOCUS_FPS))
    {
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS: lay->sort_key_stream = 0; break;
        case OV_FOCUS_PROCS:   lay->sort_key_proc = 0;   break;
        case OV_FOCUS_FPS:     lay->sort_key_fps = 0;    break;
        default: break;
        }
        lay->sort_pending = 1;
        return 1;
    }

    if (key == '>' || key == ']')
    {
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS: lay->sort_key_stream = (lay->sort_key_stream + 1) % 8; break;
        case OV_FOCUS_PROCS:   lay->sort_key_proc = (lay->sort_key_proc + 1) % 6;   break;
        case OV_FOCUS_FPS:     lay->sort_key_fps = (lay->sort_key_fps + 1) % 4;     break;
        default: break;
        }
        lay->sort_pending = 1;
        return 1;
    }

    if (key == '<')
    {
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS: lay->sort_key_stream = (lay->sort_key_stream + 7) % 8; break;
        case OV_FOCUS_PROCS:   lay->sort_key_proc = (lay->sort_key_proc + 5) % 6;   break;
        case OV_FOCUS_FPS:     lay->sort_key_fps = (lay->sort_key_fps + 3) % 4;     break;
        default: break;
        }
        lay->sort_pending = 1;
        return 1;
    }

    if (key == '[')
    {
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS: lay->sort_dir_stream = !lay->sort_dir_stream; break;
        case OV_FOCUS_PROCS:   lay->sort_dir_proc = !lay->sort_dir_proc; break;
        case OV_FOCUS_FPS:     lay->sort_dir_fps = !lay->sort_dir_fps; break;
        default: break;
        }
        lay->sort_pending = 1;
        return 1;
    }

    return 0;
}

static int ov_input__handle_actions(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_CMDLOG *log = &lay->cmdlog;

    if (key == ctrl('e') && lay->ctrl_mode)
    {
        if (lay->focus == OV_FOCUS_FPS
            && lay->sel_fps >= 0
            && lay->sel_fps < m->nb_fps)
        {
            ov_ctrl_fps_remove(&m->fps[lay->sel_fps], log);
            return 1;
        }
        else if (lay->focus == OV_FOCUS_PROCS
                 && lay->sel_proc >= 0
                 && lay->sel_proc < m->nb_procs)
        {
            ov_ctrl_proc_remove(&m->procs[lay->sel_proc], log);
            return 1;
        }
        else if (lay->focus == OV_FOCUS_STREAMS
                 && lay->sel_stream >= 0
                 && lay->sel_stream < m->nb_streams)
        {
            ov_ctrl_stream_delete(&m->streams[lay->sel_stream], log);
            return 1;
        }
    }

    if (key == 'k' || key == 'K')
    {
        if (lay->focus == OV_FOCUS_PROCS
            && lay->sel_proc >= 0
            && lay->sel_proc < m->nb_procs)
        {
            const OV_PROC *p =
                &m->procs[lay->sel_proc];
            if (key == 'k')
            {
                ov_ctrl_proc_kill(p, log);
            }
            else if (key == 'K')
            {
                ov_ctrl_proc_sigkill(p, log);
            }
            return 1;
        }
        else if (lay->focus == OV_FOCUS_FPS
                 && lay->sel_fps >= 0
                 && lay->sel_fps < m->nb_fps)
        {
            const OV_FPS *f =
                &m->fps[lay->sel_fps];
            if (key == 'k')
            {
                ov_ctrl_fps_signal_pid(
                    f, SIGTERM, log);
            }
            else if (key == 'K')
            {
                ov_ctrl_fps_signal_pid(
                    f, SIGKILL, log);
            }
            else if (key == 'x')
            {
                ov_ctrl_fps_pause_toggle(f, log);
            }
            return 1;
        }
    }

    if (key == 'C')
    {
        if (lay->focus == OV_FOCUS_PROCS)
        {
            ov_ctrl_procs_cleanup(log);
            return 1;
        }
    }

    if (key == 'i')
    {
        if (lay->focus == OV_FOCUS_STREAMS
            && lay->sel_stream >= 0
            && lay->sel_stream < m->nb_streams)
        {
            ov_ctrl_inspect_item(OV_FOCUS_STREAMS, &m->streams[lay->sel_stream]);
            return 1;
        }
        else if (lay->focus == OV_FOCUS_PROCS
                 && lay->sel_proc >= 0
                 && lay->sel_proc < m->nb_procs)
        {
            ov_ctrl_inspect_item(OV_FOCUS_PROCS, &m->procs[lay->sel_proc]);
            return 1;
        }
        else if (lay->focus == OV_FOCUS_FPS
                 && lay->sel_fps >= 0
                 && lay->sel_fps < m->nb_fps)
        {
            ov_ctrl_inspect_item(OV_FOCUS_FPS, &m->fps[lay->sel_fps]);
            return 1;
        }
    }

    if (lay->ctrl_mode)
    {
        if (lay->focus == OV_FOCUS_FPS
            && lay->sel_fps >= 0
            && lay->sel_fps < m->nb_fps)
        {
            const OV_FPS *f =
                &m->fps[lay->sel_fps];
            if (key == 'r')
            {
                ov_ctrl_fps_run_toggle(f, log);
                return 1;
            }
            if (key == 's')
            {
                ov_ctrl_fps_conf_toggle(f, log);
                return 1;
            }
        }



        if (lay->focus == OV_FOCUS_STREAMS
            && lay->sel_stream >= 0
            && lay->sel_stream < m->nb_streams)
        {
            const OV_STREAM *s =
                &m->streams[lay->sel_stream];
            if (key == OV_KEY_DEL)
            {
                ov_ctrl_stream_delete(s, log);
                return 1;
            }
        }
    }

    return 0;
}

static int ov_input__handle_navigation(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    int *sel    = NULL;
    int *scroll = NULL;
    int  count  = 0;
    int  page_h = 10;

    switch (lay->focus)
    {
    case OV_FOCUS_STREAMS:
        sel    = &lay->sel_stream;
        scroll = &lay->scroll_stream;
        count  = m->nb_streams;
        if (lay->filter_stream[0] != '\0') {
            const char *names[OV_MAX_STREAMS];
            for (int i = 0; i < count; i++) names[i] = m->streams[i].name;
            int fidx[OV_MAX_STREAMS];
            count = ov_filter_build(lay->filter_stream, names, count, fidx, OV_MAX_STREAMS);
        }
        page_h = lay->r_streams.height - 3;
        break;
    case OV_FOCUS_PROCS:
        sel    = &lay->sel_proc;
        scroll = &lay->scroll_proc;
        count  = m->nb_procs;
        if (lay->filter_proc[0] != '\0') {
            const char *names[OV_MAX_PROCS];
            for (int i = 0; i < count; i++) names[i] = m->procs[i].name;
            int fidx[OV_MAX_PROCS];
            count = ov_filter_build(lay->filter_proc, names, count, fidx, OV_MAX_PROCS);
        }
        page_h = lay->r_procs.height - 3;
        break;
    case OV_FOCUS_FPS:
        sel    = &lay->sel_fps;
        scroll = &lay->scroll_fps;
        count  = m->nb_fps;
        if (lay->filter_fps[0] != '\0') {
            const char *names[OV_MAX_FPS];
            for (int i = 0; i < count; i++) names[i] = m->fps[i].name;
            int fidx[OV_MAX_FPS];
            count = ov_filter_build(lay->filter_fps, names, count, fidx, OV_MAX_FPS);
        }
        page_h = lay->r_fps.height - 3;
        break;
    case OV_FOCUS_GRAPH:
        sel    = &lay->sel_graph;
        scroll = &lay->scroll_graph;
        {
            int start_node = get_graph_start_node(lay, m);
            if (start_node >= 0) {
                SG_RENDER_NODE rnodes[OV_MAX_NODES];
                count = sg_compute_render_nodes(m, start_node, lay->lineage_mode, rnodes);
            } else {
                count = 0;
            }
        }
        page_h = lay->r_graph.height - 3;
        break;
    default:
        break;
    }

    if (sel != NULL)
    {
        int old_sel = *sel;
        int navigated = 0;
        int is_nav_key = 0;

        if (key == OV_KEY_UP)
        {
            is_nav_key = 1;
            if (*sel > 0) { (*sel)--; navigated = 1; }
        }
        else if (key == OV_KEY_DOWN)
        {
            is_nav_key = 1;
            if (*sel < count - 1) { (*sel)++; navigated = 1; }
        }
        else if (key == OV_KEY_PGUP)
        {
            is_nav_key = 1;
            *sel -= page_h;
            if (*sel < 0) *sel = 0;
            navigated = 1;
        }
        else if (key == OV_KEY_PGDN)
        {
            is_nav_key = 1;
            *sel += page_h;
            if (*sel >= count) *sel = count - 1;
            if (*sel < 0) *sel = 0;
            navigated = 1;
        }
        else if (key == OV_KEY_HOME)
        {
            is_nav_key = 1;
            *sel = 0;
            navigated = 1;
        }
        else if (key == OV_KEY_END)
        {
            is_nav_key = 1;
            *sel = count - 1;
            if (*sel < 0) *sel = 0;
            navigated = 1;
        }

        if (is_nav_key)
        {
            if (navigated)
            {
                if (lay->focus == OV_FOCUS_GRAPH && *sel != old_sel)
                {
                    int start_node = get_graph_start_node(lay, m);
                    if (start_node >= 0)
                    {
                        SG_RENDER_NODE rnodes[OV_MAX_NODES];
                        int n_rnodes = sg_compute_render_nodes(m, start_node, lay->lineage_mode, rnodes);
                        if (*sel < n_rnodes)
                        {
                            const SG_RENDER_NODE *rn = &rnodes[*sel];
                            const OV_NODE *node = &m->nodes[rn->node_idx];
                            
                            if (node->type == OV_NODE_STREAM) {
                                lay->sel_stream = node->index;
                                if (lay->sel_stream < lay->scroll_stream) lay->scroll_stream = lay->sel_stream;
                                if (lay->r_streams.height > 3 && lay->sel_stream >= lay->scroll_stream + lay->r_streams.height - 3) {
                                    lay->scroll_stream = lay->sel_stream - (lay->r_streams.height - 3) + 1;
                                }
                            } else if (node->type == OV_NODE_PROC) {
                                lay->sel_proc = node->index;
                                if (lay->sel_proc < lay->scroll_proc) lay->scroll_proc = lay->sel_proc;
                                if (lay->r_procs.height > 3 && lay->sel_proc >= lay->scroll_proc + lay->r_procs.height - 3) {
                                    lay->scroll_proc = lay->sel_proc - (lay->r_procs.height - 3) + 1;
                                }
                            } else if (node->type == OV_NODE_FPS) {
                                lay->sel_fps = node->index;
                                if (lay->sel_fps < lay->scroll_fps) lay->scroll_fps = lay->sel_fps;
                                if (lay->r_fps.height > 3 && lay->sel_fps >= lay->scroll_fps + lay->r_fps.height - 3) {
                                    lay->scroll_fps = lay->sel_fps - (lay->r_fps.height - 3) + 1;
                                }
                            }
                        }
                    }
                }
            }
            return 1;
        }
    }

    /* Auto-scroll to keep selection visible happens in the main function or here if handled, 
       but wait, if UP/DOWN/etc are not pressed we shouldn't do anything. 
       We only return 1 if one of those keys was pressed.
       But the auto-scroll needs to happen if sel changes. We can do it before returning 1. */
    return 0; /* Not a navigation key */
}

int ov_handle_key(
    int              key,
    OV_LAYOUT       *lay,
    const OV_MODEL  *m)
{
    if (key == OV_KEY_NONE)
    {
        return 0;
    }

    /* Quit */
    if (key == 'q' || key == 'x')
    {
        return 1;
    }

    /* Command log panel toggle — 'G' */
    if (key == 'G')
    {
        if (lay->cmdlog_rows > 0)
        {
            lay->cmdlog_rows = 0;
        }
        else
        {
            lay->cmdlog_rows = 4;
        }
        ov_cmdlog_push(&lay->cmdlog,
                       OV_CMDLOG_INFO,
                       "Command log %s",
                       lay->cmdlog_rows > 0
                       ? "shown" : "hidden");
        return 0;
    }

    /* CONTROL mode toggle — 'c' */
    if (key == 'c')
    {
        lay->ctrl_mode = !lay->ctrl_mode;
        ov_cmdlog_push(&lay->cmdlog,
                       OV_CMDLOG_INFO,
                       "Control mode %s",
                       lay->ctrl_mode
                       ? "✅ ON" : "❌ OFF");
        return 0;
    }

    /* Help toggle */
    if (key == 'h')
    {
        lay->show_help = !lay->show_help;
        if (lay->show_help)
        {
            lay->help_sel = 0;
        }
        return 0;
    }

    /* Interactive help navigation when overlay is open */
    if (lay->show_help)
    {
        int nvis = ov_help_visible_count(lay);
        if (nvis < 1)
        {
            nvis = 1;
        }

        switch (key)
        {
        case OV_KEY_UP:
        case 'k':
            if (lay->help_sel > 0)
            {
                lay->help_sel--;
            }
            break;

        case OV_KEY_DOWN:
        case 'j':
            if (lay->help_sel < nvis - 1)
            {
                lay->help_sel++;
            }
            break;

        case OV_KEY_HOME:
            lay->help_sel = 0;
            break;

        case OV_KEY_END:
            lay->help_sel = nvis - 1;
            break;

        case OV_KEY_PGUP:
            lay->help_sel -= 8;
            if (lay->help_sel < 0)
            {
                lay->help_sel = 0;
            }
            break;

        case OV_KEY_PGDN:
            lay->help_sel += 8;
            if (lay->help_sel >= nvis)
            {
                lay->help_sel = nvis - 1;
            }
            break;

        case OV_KEY_ENTER:
        case '\r':
        {
            ov_help_toggle_at(lay, lay->help_sel);
            /* Clamp cursor after expand change */
            int new_nvis =
                ov_help_visible_count(lay);
            if (lay->help_sel >= new_nvis)
            {
                lay->help_sel = new_nvis - 1;
            }
            break;
        }

        case 'q':
        case 27: /* ESC */
            lay->show_help = 0;
            break;

        default:
            /* Unknown key while help showing —
             * ignore silently */
            break;
        }
        return 0;
    }

    if (ov_input__handle_filter_mode(key, lay)) return 0;
    if (ov_input__handle_mouse(key, lay, m)) return 0;
    if (ov_input__handle_view_switch(key, lay)) return 0;
    if (ov_input__handle_misc_toggles(key, lay, m)) return 0;
    if (ov_input__handle_sorting(key, lay)) return 0;
    if (ov_input__handle_actions(key, lay, m)) return 0;
    
    if (ov_input__handle_navigation(key, lay, m))
    {
        /* Perform auto-scrolling if a navigation key was pressed */
        int *sel = NULL;
        int *scroll = NULL;
        int page_h = 10;
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS:
            sel = &lay->sel_stream; scroll = &lay->scroll_stream; page_h = lay->r_streams.height - 3; break;
        case OV_FOCUS_PROCS:
            sel = &lay->sel_proc; scroll = &lay->scroll_proc; page_h = lay->r_procs.height - 3; break;
        case OV_FOCUS_FPS:
            sel = &lay->sel_fps; scroll = &lay->scroll_fps; page_h = lay->r_fps.height - 3; break;
        case OV_FOCUS_GRAPH:
            sel = &lay->sel_graph; scroll = &lay->scroll_graph; page_h = lay->r_graph.height - 3; break;
        default: break;
        }

        if (sel != NULL && scroll != NULL && page_h > 0)
        {
            if (*sel < *scroll) *scroll = *sel;
            if (*sel >= *scroll + page_h) *scroll = *sel - page_h + 1;
        }
        return 0;
    }

    if (key >= 32 && key < 127) {
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN, "Unmapped key: '%c' (code %d)", key, key);
    } else {
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN, "Unmapped key code: %d", key);
    }
    return 0;
}
