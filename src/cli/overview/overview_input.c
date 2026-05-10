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

/* scan API */
extern float ov_scan_get_interval(void);
extern void  ov_scan_set_interval(float s);

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
        return 1;
    }
    if (key == 'W')
    {
        ov_model_export_snapshot(m);
        return 1;
    }

    /* Graph jump on Enter (must be before detail mode toggle) */
    if ((key == '\n' || key == '\r') && lay->focus == OV_FOCUS_GRAPH)
    {
        if (lay->sel_graph < m->nb_edges)
        {
            const OV_EDGE *edge = &m->edges[lay->sel_graph];
            const OV_NODE *node = &m->nodes[edge->src_node];
            if (node->type == OV_NODE_STREAM) { lay->focus = OV_FOCUS_STREAMS; lay->sel_stream = node->index; }
            else if (node->type == OV_NODE_PROC) { lay->focus = OV_FOCUS_PROCS; lay->sel_proc = node->index; }
            else if (node->type == OV_NODE_FPS) { lay->focus = OV_FOCUS_FPS; lay->sel_fps = node->index; }
            lay->view = OV_VIEW_DASHBOARD;
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
            if (lay->focus == OV_FOCUS_PROCS) lay->focus = OV_FOCUS_STREAMS;
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
        case OV_FOCUS_STREAMS: lay->sort_key_stream = (lay->sort_key_stream + 1) % 7; break;
        case OV_FOCUS_PROCS:   lay->sort_key_proc = (lay->sort_key_proc + 1) % 5;   break;
        case OV_FOCUS_FPS:     lay->sort_key_fps = (lay->sort_key_fps + 1) % 3;     break;
        default: break;
        }
        lay->sort_pending = 1;
        return 1;
    }

    if (key == '<')
    {
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS: lay->sort_key_stream = (lay->sort_key_stream + 6) % 7; break;
        case OV_FOCUS_PROCS:   lay->sort_key_proc = (lay->sort_key_proc + 4) % 5;   break;
        case OV_FOCUS_FPS:     lay->sort_key_fps = (lay->sort_key_fps + 2) % 3;     break;
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
    if (key == 'k' || key == 'K' || key == 'x')
    {
        if (lay->focus == OV_FOCUS_PROCS && lay->sel_proc >= 0 && lay->sel_proc < m->nb_procs)
        {
            const OV_PROC *p = &m->procs[lay->sel_proc];
            if (key == 'k') ov_ctrl_proc_kill(p);
            else if (key == 'K') ov_ctrl_proc_sigkill(p);
            else if (key == 'x') ov_ctrl_proc_pause_toggle(p);
            return 1;
        }
        else if (lay->focus == OV_FOCUS_FPS && lay->sel_fps >= 0 && lay->sel_fps < m->nb_fps)
        {
            const OV_FPS *f = &m->fps[lay->sel_fps];
            if (key == 'k') ov_ctrl_fps_signal_pid(f, SIGTERM);
            else if (key == 'K') ov_ctrl_fps_signal_pid(f, SIGKILL);
            else if (key == 'x') ov_ctrl_fps_pause_toggle(f);
            return 1;
        }
    }

    if (lay->ctrl_mode)
    {
        if (lay->focus == OV_FOCUS_FPS && lay->sel_fps >= 0 && lay->sel_fps < m->nb_fps)
        {
            const OV_FPS *f = &m->fps[lay->sel_fps];
            if (key == 'r') { ov_ctrl_fps_run_toggle(f); return 1; }
            if (key == 's') { ov_ctrl_fps_conf_toggle(f); return 1; }
        }

        if (lay->focus == OV_FOCUS_STREAMS && lay->sel_stream >= 0 && lay->sel_stream < m->nb_streams)
        {
            const OV_STREAM *s = &m->streams[lay->sel_stream];
            if (key == 'd' || key == OV_KEY_DEL) { ov_ctrl_stream_delete(s); return 1; }
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
    default:
        break;
    }

    if (sel != NULL)
    {
        if (key == OV_KEY_UP && *sel > 0)
        {
            (*sel)--;
            return 1;
        }
        else if (key == OV_KEY_DOWN && *sel < count - 1)
        {
            (*sel)++;
            return 1;
        }
        else if (key == OV_KEY_PGUP)
        {
            *sel -= page_h;
            if (*sel < 0) *sel = 0;
            return 1;
        }
        else if (key == OV_KEY_PGDN)
        {
            *sel += page_h;
            if (*sel >= count) *sel = count - 1;
            if (*sel < 0) *sel = 0;
            return 1;
        }
        else if (key == OV_KEY_HOME)
        {
            *sel = 0;
            return 1;
        }
        else if (key == OV_KEY_END)
        {
            *sel = count - 1;
            if (*sel < 0) *sel = 0;
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
    if (key == 'q')
    {
        return 1;
    }

    /* CONTROL mode toggle — 'c' */
    if (key == 'c')
    {
        lay->ctrl_mode = !lay->ctrl_mode;
        return 0;
    }

    /* Help toggle */
    if (key == 'h')
    {
        lay->show_help = !lay->show_help;
        return 0;
    }

    /* If help is showing, any other key closes it */
    if (lay->show_help)
    {
        lay->show_help = 0;
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
        default: break;
        }

        if (sel != NULL && scroll != NULL && page_h > 0)
        {
            if (*sel < *scroll) *scroll = *sel;
            if (*sel >= *scroll + page_h) *scroll = *sel - page_h + 1;
        }
        return 0;
    }

    return 0;
}
