/**
 * @file overview_input.c
 * @brief Keyboard input handler for milk-CTRL
 */

#include <stdlib.h>
#include <string.h>

#include "overview_defs.h"
#include "overview_ansi.h"
#include "overview_data.h"
#include "overview_layout.h"
#include "overview_ctrl.h"
#include "overview_fps_edit.h"
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
 * hit_panel_tab - detect which tab label was clicked.
 * @mc:       mouse column (1-based)
 * @panel_col: panel left column
 * @tabs:     tab label array
 * @num_tabs: number of tabs
 *
 * Uses the same layout as ov_draw_panel_tabs: tabs start
 * at panel_col+2, each rendered as " LABEL " with 1-space
 * gap between tabs.
 *
 * Return: tab index [0..num_tabs-1], or -1 if no tab hit.
 */
static int hit_panel_tab(
    int         mc,
    int         panel_col,
    const char **tabs,
    int         num_tabs)
{
    int cur = panel_col + 2;
    for (int ii = 0; ii < num_tabs; ii++)
    {
        /* Tab text is " LABEL ", so width = strlen + 2 */
        int tw = (int) strlen(tabs[ii]) + 2;
        if (mc >= cur && mc < cur + tw)
        {
            return ii;
        }
        cur += tw + 1; /* +1 for gap between tabs */
    }
    return -1;
}

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

/**
 * ov_input__exec_preview_btn - execute a preview-bar
 * button action.
 *
 * Buttons are explicit UI affordances, so they auto-
 * enable ctrl_mode and log the action.
 */
/* Forward declarations for helpers defined below */
static const OV_PROC *ov_input_get_sel_proc(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m);
static const OV_FPS *ov_input_get_sel_fps(
    const OV_LAYOUT *lay,
    const OV_MODEL  *m);

static int ov_input_get_filtered_count(int focus, const OV_LAYOUT *lay, const OV_MODEL *m)
{
    int count = 0;
    if (focus == OV_FOCUS_STREAMS) {
        count = m->nb_streams;
        if (lay->filter_stream[0] != '\0') {
            const char *names[OV_MAX_STREAMS];
            for (int i = 0; i < count; i++) names[i] = m->streams[i].name;
            int fidx[OV_MAX_STREAMS];
            count = ov_filter_build(lay->filter_stream, names, count, fidx, OV_MAX_STREAMS);
        }
    } else if (focus == OV_FOCUS_PROCS) {
        count = m->nb_procs;
        if (lay->filter_proc[0] != '\0') {
            const char *names[OV_MAX_PROCS];
            for (int i = 0; i < count; i++) names[i] = m->procs[i].name;
            int fidx[OV_MAX_PROCS];
            count = ov_filter_build(lay->filter_proc, names, count, fidx, OV_MAX_PROCS);
        }
    } else if (focus == OV_FOCUS_FPS) {
        count = m->nb_fps;
        if (lay->filter_fps[0] != '\0') {
            const char *names[OV_MAX_FPS];
            for (int i = 0; i < count; i++) names[i] = m->fps[i].name;
            int fidx[OV_MAX_FPS];
            count = ov_filter_build(lay->filter_fps, names, count, fidx, OV_MAX_FPS);
        }
    }
    return count;
}

static const char *stream_col_names[] = {
    "NAME", "TYP", "SIZE", "Hz",
    "MB/s", "INODE", "COUNT", "ANCESTRY"
};

static void ov_input__streams_header_click(
    OV_LAYOUT *lay, int mc)
{
    int c = mc - lay->r_streams.col - 5
            + lay->hscroll_stream;
    if (c < 0)
    {
        return;
    }
    int col_idx = -1;
    if (c < 15)
    {
        col_idx = (lay->sort_key_stream == 7) ? 7 : 0;
    }
    else if (c < 20) { col_idx = 1; }
    else if (c < 32) { col_idx = 2; }
    else if (c < 39) { col_idx = 3; }
    else if (c < 47) { col_idx = 4; }
    else if (c < 58) { col_idx = 5; }
    else if (c >= 66 && c < 77) { col_idx = 6; }

    if (col_idx >= 0)
    {
        if (lay->sort_key_stream == col_idx)
        {
            lay->sort_dir_stream =
                !lay->sort_dir_stream;
        }
        else
        {
            lay->sort_key_stream = col_idx;
            lay->sort_dir_stream = 0;
        }
        ov_cmdlog_push(
            &lay->cmdlog, OV_CMDLOG_INFO,
            "Sort STREAMS by %s %s",
            stream_col_names[col_idx],
            lay->sort_dir_stream ? "▼" : "▲");
        lay->sort_pending = 1;
        ov_scan_force_update();
    }
}

static const char *proc_col_names[] = {
    "NAME", "PID", "STAT", "Hz",
    "MEM", "ANCESTRY"
};

static void ov_input__procs_header_click(
    OV_LAYOUT *lay, int mc)
{
    int c = mc - lay->r_procs.col - 5
            + lay->hscroll_proc;
    if (c < 0)
    {
        return;
    }
    int col_idx = -1;
    if (c < 15)
    {
        col_idx = (lay->sort_key_proc == 5) ? 5 : 0;
    }
    else if (c < 23) { col_idx = 1; }
    else if (c < 29) { col_idx = 2; }
    else if (c < 36) { col_idx = 3; }
    else if (c >= 84 && c < 90) { col_idx = 4; }

    if (col_idx >= 0)
    {
        if (lay->sort_key_proc == col_idx)
        {
            lay->sort_dir_proc =
                !lay->sort_dir_proc;
        }
        else
        {
            lay->sort_key_proc = col_idx;
            lay->sort_dir_proc = 0;
        }
        ov_cmdlog_push(
            &lay->cmdlog, OV_CMDLOG_INFO,
            "Sort PROCS by %s %s",
            proc_col_names[col_idx],
            lay->sort_dir_proc ? "▼" : "▲");
        lay->sort_pending = 1;
        ov_scan_force_update();
    }
}

static const char *fps_col_names[] = {
    "NAME", "CPID", "MEM", "ANCESTRY", "RPID"
};

static void ov_input__fps_header_click(
    OV_LAYOUT *lay, int mc)
{
    int c = mc - lay->r_fps.col - 4
            + lay->hscroll_fps;
    if (c < 0)
    {
        return;
    }
    int col_idx = -1;
    if (c <= 19)
    {
        col_idx = (lay->sort_key_fps == 3) ? 3 : 0;
    }
    else if (c >= 22 && c <= 30) { col_idx = 1; }
    else if (c >= 30 && c <= 38) { col_idx = 4; }
    else if (c >= 42 && c <= 48) { col_idx = 2; }

    if (col_idx >= 0)
    {
        if (lay->sort_key_fps == col_idx)
        {
            lay->sort_dir_fps =
                !lay->sort_dir_fps;
        }
        else
        {
            lay->sort_key_fps = col_idx;
            lay->sort_dir_fps = 0;
        }
        ov_cmdlog_push(
            &lay->cmdlog, OV_CMDLOG_INFO,
            "Sort FPS by %s %s",
            fps_col_names[col_idx],
            lay->sort_dir_fps ? "▼" : "▲");
        lay->sort_pending = 1;
        ov_scan_force_update();
    }
}

static void ov_input__exec_preview_btn(
    int              btn_id,
    OV_LAYOUT       *lay,
    const OV_MODEL  *m)
{
    OV_CMDLOG *log = &lay->cmdlog;

    if (!lay->ctrl_mode)
    {
        ov_cmdlog_push(
            &lay->cmdlog,
            OV_CMDLOG_WARN,
            "🚫 Action requires CONTROL mode (press c to toggle CTRL mode ON/OFF)");
        return;
    }

    switch (btn_id)
    {
    case OV_BTN_PROC_PAUSE:
    {
        const OV_PROC *p =
            ov_input_get_sel_proc(lay, m);
        if (p)
        {
            ov_ctrl_proc_set_ctrlval(p, -1, log);
        }
        break;
    }
    case OV_BTN_PROC_EXIT:
    {
        const OV_PROC *p =
            ov_input_get_sel_proc(lay, m);
        if (p)
        {
            ov_ctrl_proc_set_ctrlval(p, 3, log);
        }
        break;
    }
    case OV_BTN_PROC_KILL:
    {
        const OV_PROC *p =
            ov_input_get_sel_proc(lay, m);
        if (p)
        {
            ov_ctrl_proc_kill(p, log);
        }
        break;
    }
    case OV_BTN_PROC_STEP:
    {
        const OV_PROC *p =
            ov_input_get_sel_proc(lay, m);
        if (p)
        {
            ov_ctrl_proc_set_ctrlval(p, 2, log);
        }
        break;
    }
    case OV_BTN_FPS_CONF:
    {
        const OV_FPS *f =
            ov_input_get_sel_fps(lay, m);
        if (f)
        {
            ov_ctrl_fps_conf_toggle(f, log);
        }
        break;
    }
    case OV_BTN_FPS_RUN:
    {
        const OV_FPS *f =
            ov_input_get_sel_fps(lay, m);
        if (f)
        {
            ov_ctrl_fps_run_toggle(f, log);
        }
        break;
    }
    case OV_BTN_FPS_KILL:
    {
        const OV_FPS *f =
            ov_input_get_sel_fps(lay, m);
        if (f)
        {
            ov_ctrl_fps_signal_pid(
                f, SIGTERM, log);
        }
        break;
    }
    default:
        break;
    }
    
    ov_scan_force_update();
}

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
            /* Tab widths must match the renderer in ov_render_header():
             * each tab renders as " [" + " Fn:LABEL " + "] " (inactive)
             * or " " + ← + " Fn:LABEL " + → + " " (active),
             * both totalling strlen(label) + 9 visible columns. */
            static const char *vlabels[] =
                {"DASH", "STRM", "PROC", "FPS", "CONN"};
            int tab_widths[OV_VIEW_COUNT];
            int tabs_total_width = 0;
            for (int v = 0; v < OV_VIEW_COUNT; v++)
            {
                tab_widths[v] = (int) strlen(vlabels[v]) + 9;
                tabs_total_width += tab_widths[v];
            }
            
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

        /* --- Preview-bar button clicks (row 2) --- */
        if (mr == 2 && lay->nb_preview_btns > 0)
        {
            for (int bi = 0;
                 bi < lay->nb_preview_btns; bi++)
            {
                int bc = lay->preview_btns[bi].col;
                int bw = lay->preview_btns[bi].width;
                if (mc >= bc && mc < bc + bw)
                {
                    ov_input__exec_preview_btn(
                        lay->preview_btns[bi].id,
                        lay, m);
                    return 1;
                }
            }
        }

        /* --- View-aware panel dispatch ---
         * In single-panel views (F3–F6) the panel rects
         * for non-active panels all overlap the full
         * screen, so we must check the view first and
         * route directly to the correct handler.
         * on the Dashboard all four rects are distinct
         * so the cascade works normally.
         */

        if (lay->view == OV_VIEW_DASHBOARD)
        {
            /* Check for dashboard horizontal split drag.
             * The split border is at h_split_row, which
             * coincides with r_graph.row (the tab header
             * row of the bottom-right panel).  Only start
             * a drag when the click is in the LEFT half
             * (under streams/fps); clicks on the RIGHT
             * half fall through so tab labels remain
             * clickable. */
            int h_split_row =
                lay->r_streams.row + lay->r_streams.height;
            if (mr == h_split_row - 1 || mr == h_split_row)
            {
                if (mc < lay->r_graph.col)
                {
                    lay->dash_split_h_dragging = 1;
                    return 1;
                }
            }
            /* Check for dashboard vertical split drag */
            int v_split_col = lay->r_streams.width;
            if (mc == v_split_col || mc == v_split_col + 1)
            {
                if (mr >= lay->r_streams.row && mr < lay->r_streams.row + lay->r_streams.height + lay->r_fps.height)
                {
                    lay->dash_split_v_dragging = 1;
                    return 1;
                }
            }
        }

        if (lay->view == OV_VIEW_FPS)
        {
            /* F5: Check for split drag */
            if (mc == lay->r_fps_list.width || mc == lay->r_fps_list.width + 1)
            {
                lay->fps_split_dragging = 1;
                return 1;
            }

            /* F5: left = fps list, right = params */
            if (INSIDE(lay->r_fps_params, mr, mc))
            {
                lay->focus = OV_FOCUS_FPS;
                lay->fps_param_focus = 1;
                int body_row =
                    mr - lay->r_fps_params.row - 2;
                if (body_row >= 0)
                {
                    int idx =
                        lay->fps_param_scroll + body_row;
                    lay->fps_param_sel = idx;
                }
            }
            else if (INSIDE(lay->r_fps, mr, mc))
            {
                lay->focus = OV_FOCUS_FPS;
                lay->fps_param_focus = 0;
                int body_row =
                    mr - lay->r_fps.row - 2;
                if (body_row == -1)
                {
                    ov_input__fps_header_click(lay, mc);
                }
                else if (body_row >= 0)
                {
                    int idx =
                        lay->scroll_fps + body_row;
                    if (idx < m->nb_fps)
                    {
                        if (lay->sel_fps != idx)
                        {
                            lay->sel_fps = idx;
                            lay->sel_name_fps[0] = '\0';
                            lay->fps_param_path[0] = '\0';
                            lay->fps_param_sel = 0;
                            lay->fps_param_scroll = 0;
                        }
                        if (is_dbl)
                        {
                            lay->graph_tab_mode = 1;
                        }
                    }
                }
            }
        }
        else if (lay->view == OV_VIEW_STREAMS
                 && INSIDE(lay->r_streams, mr, mc))
        {
            lay->focus = OV_FOCUS_STREAMS;
            int body_row =
                mr - lay->r_streams.row - 2;
            if (body_row == -1)
            {
                ov_input__streams_header_click(lay, mc);
            }
            else if (body_row >= 0)
            {
                int idx =
                    lay->scroll_stream + body_row;
                if (idx < m->nb_streams)
                {
                    lay->sel_stream = idx;
                    lay->sel_name_stream[0] = '\0';
                    if (is_dbl)
                    {
                        lay->graph_tab_mode = 1;
                    }
                }
            }
        }
        else if (lay->view == OV_VIEW_PROCS
                 && INSIDE(lay->r_procs, mr, mc))
        {
            lay->focus = OV_FOCUS_PROCS;
            int body_row =
                mr - lay->r_procs.row - 2;
            if (body_row == -1)
            {
                ov_input__procs_header_click(lay, mc);
            }
            else if (body_row >= 0)
            {
                int idx =
                    lay->scroll_proc + body_row;
                if (idx < m->nb_procs)
                {
                    lay->sel_proc = idx;
                    lay->sel_name_proc[0] = '\0';
                    if (is_dbl)
                    {
                        lay->graph_tab_mode = 1;
                    }
                }
            }
        }
        else if (lay->view == OV_VIEW_GRAPH
                 && INSIDE(lay->r_graph, mr, mc))
        {
            lay->focus = OV_FOCUS_GRAPH;

            /* Tab header click */
            if (mr == lay->r_graph.row)
            {
                const char *dtabs[] = {
                    "CONNECTIONS",
                    "DETAILS",
                    "RESOURCES"
                };
                int ti = hit_panel_tab(
                    mc, lay->r_graph.col,
                    dtabs, 3);
                if (ti >= 0)
                {
                    lay->graph_tab_mode = ti;
                    ov_scan_force_update();
                }
            }
            else
            {
                int body_row =
                    mr - lay->r_graph.row - 2;
                if (body_row >= 0)
                {
                    int idx =
                        lay->scroll_graph
                        + body_row;
                    if (idx < m->nb_edges)
                    {
                        lay->sel_graph = idx;
                        if (is_dbl)
                        {
                            const OV_EDGE *edge =
                                &m->edges[idx];
                            const OV_NODE *node =
                                &m->nodes[
                                    edge->src_node];
                            if (node->type
                                == OV_NODE_STREAM)
                            {
                                lay->focus =
                                    OV_FOCUS_STREAMS;
                                lay->sel_stream =
                                    node->index;
                                lay->sel_name_stream
                                    [0] = '\0';
                            }
                            else if (node->type
                                == OV_NODE_PROC)
                            {
                                lay->focus =
                                    OV_FOCUS_PROCS;
                                lay->sel_proc =
                                    node->index;
                                lay->sel_name_proc
                                    [0] = '\0';
                            }
                            else if (node->type
                                == OV_NODE_FPS)
                            {
                                lay->focus =
                                    OV_FOCUS_FPS;
                                lay->sel_fps =
                                    node->index;
                                lay->sel_name_fps
                                    [0] = '\0';
                            }
                            lay->view =
                                OV_VIEW_DASHBOARD;
                        }
                    }
                }
            }
        }
        else if (lay->view == OV_VIEW_DASHBOARD)
        {
            /* Dashboard: four non-overlapping rects */
            if (INSIDE(lay->r_streams, mr, mc))
            {
                lay->focus = OV_FOCUS_STREAMS;
                int body_row =
                    mr - lay->r_streams.row - 2;
                if (body_row == -1)
                {
                    ov_input__streams_header_click(lay, mc);
                }
                else if (body_row >= 0)
                {
                    int idx =
                        lay->scroll_stream + body_row;
                    if (idx < m->nb_streams)
                    {
                        lay->sel_stream = idx;
                        lay->sel_name_stream[0] = '\0';
                        if (is_dbl)
                        {
                            lay->graph_tab_mode = 1;
                        }
                    }
                }
            }
            else if (INSIDE(lay->r_procs, mr, mc))
            {
                lay->focus = OV_FOCUS_PROCS;
                int body_row =
                    mr - lay->r_procs.row - 2;
                if (body_row == -1)
                {
                    ov_input__procs_header_click(lay, mc);
                }
                else if (body_row >= 0)
                {
                    int idx =
                        lay->scroll_proc + body_row;
                    if (idx < m->nb_procs)
                    {
                        lay->sel_proc = idx;
                        lay->sel_name_proc[0] = '\0';
                        if (is_dbl)
                        {
                            lay->graph_tab_mode = 1;
                        }
                    }
                }
            }
            else if (INSIDE(lay->r_fps, mr, mc))
            {
                lay->focus = OV_FOCUS_FPS;
                int body_row =
                    mr - lay->r_fps.row - 2;
                if (body_row == -1)
                {
                    ov_input__fps_header_click(lay, mc);
                }
                else if (body_row >= 0)
                {
                    int idx =
                        lay->scroll_fps + body_row;
                    if (idx < m->nb_fps)
                    {
                        lay->sel_fps = idx;
                        lay->sel_name_fps[0] = '\0';
                        if (is_dbl)
                        {
                            lay->graph_tab_mode = 1;
                        }
                    }
                }
            }
            else if (INSIDE(lay->r_graph, mr, mc))
            {
                lay->focus = OV_FOCUS_GRAPH;

                /* Tab header click */
                if (mr == lay->r_graph.row)
                {
                    const char *dtabs[] = {
                        "CONNECTIONS",
                        "DETAILS",
                        "RESOURCES"
                    };
                    int ti = hit_panel_tab(
                        mc, lay->r_graph.col,
                        dtabs, 3);
                    if (ti >= 0)
                    {
                        lay->graph_tab_mode = ti;
                        ov_scan_force_update();
                    }
                }
                else
                {
                    int body_row =
                        mr - lay->r_graph.row - 2;
                    if (body_row >= 0)
                    {
                        int idx =
                            lay->scroll_graph
                            + body_row;
                        if (idx < m->nb_edges)
                        {
                            lay->sel_graph = idx;
                            if (is_dbl)
                            {
                                const OV_EDGE *edge =
                                    &m->edges[idx];
                                const OV_NODE *node =
                                    &m->nodes[
                                        edge->src_node];
                                if (node->type
                                    == OV_NODE_STREAM)
                                {
                                    lay->focus =
                                        OV_FOCUS_STREAMS;
                                    lay->sel_stream =
                                        node->index;
                                    lay->sel_name_stream
                                        [0] = '\0';
                                }
                                else if (node->type
                                    == OV_NODE_PROC)
                                {
                                    lay->focus =
                                        OV_FOCUS_PROCS;
                                    lay->sel_proc =
                                        node->index;
                                    lay->sel_name_proc
                                        [0] = '\0';
                                }
                                else if (node->type
                                    == OV_NODE_FPS)
                                {
                                    lay->focus =
                                        OV_FOCUS_FPS;
                                    lay->sel_fps =
                                        node->index;
                                    lay->sel_name_fps
                                        [0] = '\0';
                                }
                                lay->view =
                                    OV_VIEW_DASHBOARD;
                            }
                        }
                    }
                }
            }
        }
        
        /* Check if clicking on cmdlog border to start dragging */
        {
            int cmdlog_top = (lay->cmdlog_rows > 0) ? (lay->term_rows - lay->cmdlog_rows) : lay->term_rows;
            if (mr == cmdlog_top - 1 || mr == cmdlog_top)
            {
                lay->cmdlog_dragging = 1;
            }
        }
        
        return 1;
    }

    if (key == OV_KEY_MOUSE_RELEASE)
    {
        lay->fps_split_dragging = 0;
        lay->dash_split_v_dragging = 0;
        lay->dash_split_h_dragging = 0;
        lay->cmdlog_dragging = 0;
        return 1;
    }

    if (key == OV_KEY_MOUSE_DRAG)
    {
        int mr = ov_mouse_row;
        int mc = ov_mouse_col;

        /* Global: Command log panel height drag */
        int cmdlog_top = (lay->cmdlog_rows > 0) ? (lay->term_rows - lay->cmdlog_rows) : lay->term_rows;
        if (lay->cmdlog_dragging || (mr == cmdlog_top - 1 || mr == cmdlog_top))
        {
            lay->cmdlog_dragging = 1;
            int new_h = lay->term_rows - 1 - mr;
            if (new_h < 0) new_h = 0;
            if (new_h > lay->term_rows / 2) new_h = lay->term_rows / 2;
            lay->cmdlog_rows = new_h;
            return 1;
        }

        if (lay->view == OV_VIEW_FPS)
        {
            if (lay->fps_split_dragging || (mc >= lay->r_fps_list.width - 1 && mc <= lay->r_fps_list.width + 2))
            {
                lay->fps_split_dragging = 1;
                float ratio = (float)mc / lay->term_cols;
                if (ratio < 0.1f) ratio = 0.1f;
                if (ratio > 0.9f) ratio = 0.9f;
                lay->fps_split_ratio = ratio;
                return 1;
            }
        }
        if (lay->view == OV_VIEW_DASHBOARD)
        {
            int h_split_row = lay->r_streams.row + lay->r_streams.height;
            int v_split_col = lay->r_streams.width;

            int handled = 0;
            if (lay->dash_split_h_dragging || (mr >= h_split_row - 1 && mr <= h_split_row + 1))
            {
                lay->dash_split_h_dragging = 1;
                int log_h = lay->cmdlog_rows;
                if (log_h < 0) log_h = 0;
                int body_top = 3;
                int body_h = lay->term_rows - 3 - log_h;
                if (body_h < 4) body_h = 4;
                float ratio = (float)(mr - body_top) / body_h;
                if (ratio < 0.1f) ratio = 0.1f;
                if (ratio > 0.9f) ratio = 0.9f;
                lay->dash_split_h_ratio = ratio;
                handled = 1;
            }
            if (lay->dash_split_v_dragging || (mc >= v_split_col - 1 && mc <= v_split_col + 2))
            {
                lay->dash_split_v_dragging = 1;
                float ratio = (float)mc / lay->term_cols;
                if (ratio < 0.1f) ratio = 0.1f;
                if (ratio > 0.9f) ratio = 0.9f;
                lay->dash_split_v_ratio = ratio;
                handled = 1;
            }
            if (handled) return 1;
        }
        return 0; /* Ignore other drags for now */
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

        /* View-aware dispatch (same rationale as click
         * handler — in single-panel views the panel
         * rects overlap).
         */
        if (lay->view == OV_VIEW_FPS)
        {
            if (INSIDE(lay->r_fps_params, mr, mc))
            {
                sel    = &lay->fps_param_sel;
                scroll = &lay->fps_param_scroll;
                count  = 1000;
                page_h = lay->r_fps_params.height - 3;
            }
            else if (INSIDE(lay->r_fps, mr, mc))
            {
                sel    = &lay->sel_fps;
                scroll = &lay->scroll_fps;
                count  = ov_input_get_filtered_count(OV_FOCUS_FPS, lay, m);
                page_h = lay->r_fps.height - 3;
            }
        }
        else if (lay->view == OV_VIEW_STREAMS)
        {
            sel    = &lay->sel_stream;
            scroll = &lay->scroll_stream;
            count  = ov_input_get_filtered_count(OV_FOCUS_STREAMS, lay, m);
            page_h = lay->r_streams.height - 3;
        }
        else if (lay->view == OV_VIEW_PROCS)
        {
            sel    = &lay->sel_proc;
            scroll = &lay->scroll_proc;
            count  = ov_input_get_filtered_count(OV_FOCUS_PROCS, lay, m);
            page_h = lay->r_procs.height - 3;
        }
        else if (lay->view == OV_VIEW_GRAPH)
        {
            if (lay->graph_tab_mode == 0)
            {
                sel    = &lay->sel_graph;
                scroll = &lay->scroll_graph;
                int start_node =
                    get_graph_start_node(lay, m);
                if (start_node >= 0)
                {
                    SG_RENDER_NODE rnodes[OV_MAX_NODES];
                    count = sg_compute_render_nodes(
                        m, start_node,
                        lay->lineage_mode, rnodes);
                }
                else
                {
                    count = m->nb_edges;
                }
                page_h = lay->r_graph.height - 3;
            }
            else if (lay->graph_tab_mode == 1)
            {
                page_h = lay->r_graph.height - 3;
                lay->scroll_detail += dir;
                if (lay->scroll_detail < 0)
                {
                    lay->scroll_detail = 0;
                }
                if (lay->scroll_detail >
                    lay->detail_total_lines - page_h)
                {
                    lay->scroll_detail =
                        lay->detail_total_lines - page_h;
                }
                if (lay->scroll_detail < 0)
                {
                    lay->scroll_detail = 0;
                }
                return 1;
            }
            else
            {
                /* RESOURCES panel */
                return 1;
            }
        }
        else if (lay->view == OV_VIEW_DASHBOARD)
        {
            /* Dashboard: non-overlapping rects */
            if (INSIDE(lay->r_streams, mr, mc))
            {
                sel    = &lay->sel_stream;
                scroll = &lay->scroll_stream;
                count  = ov_input_get_filtered_count(OV_FOCUS_STREAMS, lay, m);
                page_h = lay->r_streams.height - 3;
            }
            else if (INSIDE(lay->r_procs, mr, mc))
            {
                sel    = &lay->sel_proc;
                scroll = &lay->scroll_proc;
                count  = ov_input_get_filtered_count(OV_FOCUS_PROCS, lay, m);
                page_h = lay->r_procs.height - 3;
            }
            else if (INSIDE(lay->r_fps, mr, mc))
            {
                sel    = &lay->sel_fps;
                scroll = &lay->scroll_fps;
                count  = ov_input_get_filtered_count(OV_FOCUS_FPS, lay, m);
                page_h = lay->r_fps.height - 3;
            }
            else if (INSIDE(lay->r_graph, mr, mc))
            {
                if (lay->graph_tab_mode == 0)
                {
                    sel    = &lay->sel_graph;
                    scroll = &lay->scroll_graph;
                    int start_node =
                        get_graph_start_node(lay, m);
                    if (start_node >= 0)
                    {
                        SG_RENDER_NODE rnodes[OV_MAX_NODES];
                        count = sg_compute_render_nodes(
                            m, start_node,
                            lay->lineage_mode, rnodes);
                    }
                    else
                    {
                        count = m->nb_edges;
                    }
                    page_h = lay->r_graph.height - 3;
                }
                else if (lay->graph_tab_mode == 1)
                {
                    page_h = lay->r_graph.height - 3;
                    lay->scroll_detail += dir;
                    if (lay->scroll_detail < 0)
                    {
                        lay->scroll_detail = 0;
                    }
                    if (lay->scroll_detail >
                        lay->detail_total_lines - page_h)
                    {
                        lay->scroll_detail =
                            lay->detail_total_lines
                            - page_h;
                    }
                    if (lay->scroll_detail < 0)
                    {
                        lay->scroll_detail = 0;
                    }
                    return 1;
                }
                else
                {
                    /* RESOURCES panel */
                    return 1;
                }
            }
        }

        if (sel != NULL && scroll != NULL)
        {
            *scroll += dir;
            if (*scroll < 0) *scroll = 0;
            if (page_h > 0 && *scroll > count - page_h) {
                *scroll = count - page_h;
                if (*scroll < 0) *scroll = 0;
            }

            *sel += dir;
            if (*sel < *scroll) *sel = *scroll;
            if (page_h > 0 && *sel >= *scroll + page_h) *sel = *scroll + page_h - 1;
            if (*sel >= count) *sel = count - 1;
            if (*sel < 0) *sel = 0;

            if (sel == &lay->sel_stream) lay->sel_name_stream[0] = '\0';
            else if (sel == &lay->sel_proc) lay->sel_name_proc[0] = '\0';
            else if (sel == &lay->sel_fps) lay->sel_name_fps[0] = '\0';
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

    if (key == OV_KEY_BTAB)
    {
        lay->graph_tab_mode = (lay->graph_tab_mode + 1) % 3;
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
        lay->graph_tab_mode = (lay->graph_tab_mode == 1) ? 0 : 1;
        return 1;
    }
    if (key == 'L')
    {
        lay->lineage_mode = (lay->lineage_mode + 1) % 3;
        return 1;
    }
    if (key == 'v' || key == 'V')
    {
        int step = (key == 'v') ? -1 : 1;
        if (lay->cmdlog_rows == 0 && step > 0)
        {
            lay->cmdlog_rows = 4;
        }
        else
        {
            lay->cmdlog_rows += step;
        }
        if (lay->cmdlog_rows < 0) lay->cmdlog_rows = 0;
        if (lay->cmdlog_rows > lay->term_rows / 2) lay->cmdlog_rows = lay->term_rows / 2;
        return 1;
    }
    if (key == '{' || key == '}')
    {
        float step = (key == '{') ? -0.05f : 0.05f;
        if (lay->view == OV_VIEW_FPS)
        {
            lay->fps_split_ratio += step;
            if (lay->fps_split_ratio < 0.1f) lay->fps_split_ratio = 0.1f;
            if (lay->fps_split_ratio > 0.9f) lay->fps_split_ratio = 0.9f;
            return 1;
        }
        else if (lay->view == OV_VIEW_DASHBOARD)
        {
            lay->dash_split_v_ratio += step;
            if (lay->dash_split_v_ratio < 0.1f) lay->dash_split_v_ratio = 0.1f;
            if (lay->dash_split_v_ratio > 0.9f) lay->dash_split_v_ratio = 0.9f;
            return 1;
        }
    }
    if (key == '(' || key == ')')
    {
        float step = (key == '(') ? -0.05f : 0.05f;
        if (lay->view == OV_VIEW_DASHBOARD)
        {
            lay->dash_split_h_ratio += step;
            if (lay->dash_split_h_ratio < 0.1f) lay->dash_split_h_ratio = 0.1f;
            if (lay->dash_split_h_ratio > 0.9f) lay->dash_split_h_ratio = 0.9f;
            return 1;
        }
    }

    if (key == '{' || key == '}')
    {
        float step = (key == '{') ? -0.05f : 0.05f;
        if (lay->view == OV_VIEW_DASHBOARD)
        {
            lay->dash_split_v_ratio += step;
            if (lay->dash_split_v_ratio < 0.1f) lay->dash_split_v_ratio = 0.1f;
            if (lay->dash_split_v_ratio > 0.9f) lay->dash_split_v_ratio = 0.9f;
            return 1;
        }
        else if (lay->view == OV_VIEW_FPS)
        {
            lay->fps_split_ratio += step;
            if (lay->fps_split_ratio < 0.1f) lay->fps_split_ratio = 0.1f;
            if (lay->fps_split_ratio > 0.9f) lay->fps_split_ratio = 0.9f;
            return 1;
        }
    }
    if (key == 'F')
    {
        lay->paused = !lay->paused;
        ov_cmdlog_push(&lay->cmdlog,
                       OV_CMDLOG_INFO,
                       "%s Display %s",
                       lay->paused ? "⏸️" : "▶️",
                       lay->paused ? "paused" : "resumed");
        return 1;
    }
    if (key == 'W')
    {
        ov_model_export_snapshot(m);
        ov_cmdlog_push(&lay->cmdlog,
                       OV_CMDLOG_OK,
                       "📸 Snapshot exported");
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

    /* ENTER — open DETAILS for list panels; toggle when on graph */
    if ((key == 10 || key == 13) && m != NULL)
    {
        /* In dedicated FPS view, ENTER is used to edit parameters */
        if (lay->view == OV_VIEW_FPS)
        {
            return 0;
        }

        if (lay->focus == OV_FOCUS_STREAMS
            || lay->focus == OV_FOCUS_PROCS
            || lay->focus == OV_FOCUS_FPS)
        {
            /* Always jump to DETAILS sub-tab */
            lay->graph_tab_mode = 1;
        }
        else
        {
            lay->graph_tab_mode =
                (lay->graph_tab_mode == 1) ? 0 : 1;
        }
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
        /* In single-panel views (F3–F6), lock focus to
         * the panel being shown — only allow left-right
         * panel cycling on the Dashboard.
         */
        if (lay->view != OV_VIEW_DASHBOARD)
        {
            if (lay->view == OV_VIEW_FPS)
            {
                return 0; /* Let handle_navigation process it */
            }
            return 1;
        }

        if (key == OV_KEY_LEFT)
        {
            if (lay->focus == OV_FOCUS_STREAMS)
                lay->focus = OV_FOCUS_GRAPH;
            else if (lay->focus == OV_FOCUS_PROCS)
                lay->focus = OV_FOCUS_STREAMS;
            else if (lay->focus == OV_FOCUS_FPS)
                lay->focus = OV_FOCUS_PROCS;
            else if (lay->focus == OV_FOCUS_GRAPH)
                lay->focus = OV_FOCUS_FPS;
        }
        else
        {
            if (lay->focus == OV_FOCUS_STREAMS)
                lay->focus = OV_FOCUS_PROCS;
            else if (lay->focus == OV_FOCUS_PROCS)
                lay->focus = OV_FOCUS_FPS;
            else if (lay->focus == OV_FOCUS_FPS)
                lay->focus = OV_FOCUS_GRAPH;
            else if (lay->focus == OV_FOCUS_GRAPH)
                lay->focus = OV_FOCUS_STREAMS;
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

    if (key == 'A')
    {
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS: lay->sort_key_stream = 7; lay->sort_dir_stream = 0; break;
        case OV_FOCUS_PROCS:   lay->sort_key_proc = 5;   lay->sort_dir_proc = 0;   break;
        case OV_FOCUS_FPS:     lay->sort_key_fps = 3;    lay->sort_dir_fps = 0;    break;
        default: break;
        }
        lay->sort_pending = 1;
        return 1;
    }

    if (key == 's' && !(lay->ctrl_mode && (lay->focus == OV_FOCUS_FPS || lay->focus == OV_FOCUS_PROCS)))
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
        case OV_FOCUS_FPS:     lay->sort_key_fps = (lay->sort_key_fps + 1) % 5;     break;
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
        case OV_FOCUS_FPS:     lay->sort_key_fps = (lay->sort_key_fps + 4) % 5;     break;
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

static const OV_STREAM *ov_input_get_sel_stream(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (lay->sel_name_stream[0] == '\0') return NULL;
    for (int i = 0; i < m->nb_streams; i++) {
        if (strcmp(m->streams[i].name, lay->sel_name_stream) == 0) return &m->streams[i];
    }
    return NULL;
}

static const OV_PROC *ov_input_get_sel_proc(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (lay->sel_name_proc[0] == '\0') return NULL;
    for (int i = 0; i < m->nb_procs; i++) {
        if (strcmp(m->procs[i].name, lay->sel_name_proc) == 0) return &m->procs[i];
    }
    return NULL;
}

static const OV_FPS *ov_input_get_sel_fps(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (lay->sel_name_fps[0] == '\0') return NULL;
    for (int i = 0; i < m->nb_fps; i++) {
        if (strcmp(m->fps[i].name, lay->sel_name_fps) == 0) return &m->fps[i];
    }
    return NULL;
}

static int ov_input__handle_actions(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_CMDLOG *log = &lay->cmdlog;

    if (key == ctrl('e'))
    {
        if (!lay->ctrl_mode)
        {
            ov_cmdlog_push(log, OV_CMDLOG_WARN, "🚫 CTRL+e requires CONTROL mode (press c to toggle CTRL mode ON/OFF)");
            return 1;
        }

        if (lay->focus == OV_FOCUS_FPS)
        {
            const OV_FPS *f = ov_input_get_sel_fps(lay, m);
            if (f) ov_ctrl_fps_remove(f, log);
        }
        else if (lay->focus == OV_FOCUS_PROCS)
        {
            const OV_PROC *p = ov_input_get_sel_proc(lay, m);
            if (p) ov_ctrl_proc_remove(p, log);
        }
        else if (lay->focus == OV_FOCUS_STREAMS)
        {
            const OV_STREAM *s = ov_input_get_sel_stream(lay, m);
            if (s) ov_ctrl_stream_delete(s, log);
        }
        ov_scan_force_update();
        return 1;
    }

    if (key == 'i')
    {
        if (lay->focus == OV_FOCUS_STREAMS)
        {
            const OV_STREAM *s = ov_input_get_sel_stream(lay, m);
            if (s) ov_ctrl_inspect_item(OV_FOCUS_STREAMS, s);
        }
        else if (lay->focus == OV_FOCUS_PROCS)
        {
            const OV_PROC *p = ov_input_get_sel_proc(lay, m);
            if (p) ov_ctrl_inspect_item(OV_FOCUS_PROCS, p);
        }
        else if (lay->focus == OV_FOCUS_FPS)
        {
            const OV_FPS *f = ov_input_get_sel_fps(lay, m);
            if (f) ov_ctrl_inspect_item(OV_FOCUS_FPS, f);
        }
        return 1;
    }

    int is_ctrl_action = 0;

    if (lay->focus == OV_FOCUS_PROCS)
    {
        if (key == 'C')
        {
            is_ctrl_action = 1;
            if (lay->ctrl_mode)
            {
                ov_ctrl_procs_cleanup(log);
            }
        }
        else if (key == 'k' || key == 'K' || key == 'p' || key == 's' || key == ctrl('s') || key == 'e' || key == 'z')
        {
            is_ctrl_action = 1;
            if (lay->ctrl_mode)
            {
                const OV_PROC *p = ov_input_get_sel_proc(lay, m);
                if (p)
                {
                    if (key == 'k') ov_ctrl_proc_kill(p, log);
                    else if (key == 'K') ov_ctrl_proc_sigkill(p, log);
                    else if (key == 'p') ov_ctrl_proc_set_ctrlval(p, -1, log);
                    else if (key == 's' || key == ctrl('s')) ov_ctrl_proc_set_ctrlval(p, 2, log);
                    else if (key == 'e') ov_ctrl_proc_set_ctrlval(p, 3, log);
                    else if (key == 'z') ov_ctrl_proc_zero_counters(p, log);
                }
            }
        }
    }
    else if (lay->focus == OV_FOCUS_FPS)
    {
        if (key == 'k' || key == 'K' || key == 'x' || key == 'r' || key == 's')
        {
            is_ctrl_action = 1;
            if (lay->ctrl_mode)
            {
                const OV_FPS *f = ov_input_get_sel_fps(lay, m);
                if (f)
                {
                    if (key == 'k') ov_ctrl_fps_signal_pid(f, SIGTERM, log);
                    else if (key == 'K') ov_ctrl_fps_signal_pid(f, SIGKILL, log);
                    else if (key == 'x') ov_ctrl_fps_pause_toggle(f, log);
                    else if (key == 'r') ov_ctrl_fps_run_toggle(f, log);
                    else if (key == 's') ov_ctrl_fps_conf_toggle(f, log);
                }
            }
        }
    }
    else if (lay->focus == OV_FOCUS_STREAMS)
    {
        if (key == OV_KEY_DEL)
        {
            is_ctrl_action = 1;
            if (lay->ctrl_mode)
            {
                const OV_STREAM *s = ov_input_get_sel_stream(lay, m);
                if (s) ov_ctrl_stream_delete(s, log);
            }
        }
    }

    if (is_ctrl_action)
    {
        if (!lay->ctrl_mode)
        {
            char keyname[16];
            if (key == ctrl('s')) snprintf(keyname, sizeof(keyname), "CTRL+s");
            else if (key == OV_KEY_DEL) snprintf(keyname, sizeof(keyname), "DEL");
            else snprintf(keyname, sizeof(keyname), "'%c'", key);
            
            ov_cmdlog_push(log, OV_CMDLOG_WARN, "🚫 %s requires CONTROL mode (press c to toggle CTRL mode ON/OFF)", keyname);
        }
        else
        {
            ov_scan_force_update();
        }
        return 1;
    }

    return 0;
}

static int find_relative_node_of_type(const OV_MODEL *m, int start_node, int target_type, int upstream)
{
    int queue[OV_MAX_NODES];
    int visited[OV_MAX_NODES];
    memset(visited, 0, sizeof(visited));
    int head = 0, tail = 0;
    
    queue[tail++] = start_node;
    visited[start_node] = 1;
    
    while (head < tail) {
        int curr = queue[head++];
        
        for (int i = 0; i < m->nb_edges; i++) {
            int next = -1;
            if (upstream && m->edges[i].tgt_node == curr) {
                next = m->edges[i].src_node;
            } else if (!upstream && m->edges[i].src_node == curr) {
                next = m->edges[i].tgt_node;
            }
            
            if (next >= 0 && !visited[next]) {
                if (target_type < 0 || m->nodes[next].type == target_type) {
                    return next;
                }
                visited[next] = 1;
                queue[tail++] = next;
            }
        }
    }
    return -1;
}

static int ov_input__handle_ancestry_nav(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (key != OV_KEY_SHIFT_UP && key != OV_KEY_SHIFT_DOWN)
        return 0;

    int current_node = -1;
    int target_type = -1;
    if (lay->focus == OV_FOCUS_STREAMS && lay->sel_stream >= 0 && lay->sel_stream < m->nb_streams) {
        current_node = m->streams[lay->sel_stream].node_idx;
        target_type = OV_NODE_STREAM;
    } else if (lay->focus == OV_FOCUS_PROCS && lay->sel_proc >= 0 && lay->sel_proc < m->nb_procs) {
        current_node = m->procs[lay->sel_proc].node_idx;
        target_type = OV_NODE_PROC;
    } else if (lay->focus == OV_FOCUS_FPS && lay->sel_fps >= 0 && lay->sel_fps < m->nb_fps) {
        current_node = m->fps[lay->sel_fps].node_idx;
        target_type = OV_NODE_FPS;
    } else if (lay->focus == OV_FOCUS_GRAPH) {
        // Find selected graph node
        int start_node = get_graph_start_node(lay, m);
        if (start_node >= 0) {
            SG_RENDER_NODE rnodes[OV_MAX_NODES];
            int n_rnodes = sg_compute_render_nodes(m, start_node, lay->lineage_mode, rnodes);
            if (lay->sel_graph >= 0 && lay->sel_graph < n_rnodes) {
                current_node = rnodes[lay->sel_graph].node_idx;
            }
        }
        target_type = -1; // Any type
    }

    if (current_node < 0 || current_node >= m->nb_nodes) return 1;

    int target_node = find_relative_node_of_type(m, current_node, target_type, key == OV_KEY_SHIFT_UP);

    if (target_node >= 0 && target_node < m->nb_nodes) {
        const OV_NODE *tn = &m->nodes[target_node];
        if (lay->focus == OV_FOCUS_GRAPH) {
            // Find target_node in graph render list
            int start_node = get_graph_start_node(lay, m);
            if (start_node >= 0) {
                SG_RENDER_NODE rnodes[OV_MAX_NODES];
                int n_rnodes = sg_compute_render_nodes(m, start_node, lay->lineage_mode, rnodes);
                for (int i = 0; i < n_rnodes; i++) {
                    if (rnodes[i].node_idx == target_node) {
                        lay->sel_graph = i;
                        break;
                    }
                }
            }
        } else if (lay->focus == OV_FOCUS_STREAMS && tn->type == OV_NODE_STREAM) {
            lay->sel_stream = tn->index;
            lay->sel_name_stream[0] = '\0';
        } else if (lay->focus == OV_FOCUS_PROCS && tn->type == OV_NODE_PROC) {
            lay->sel_proc = tn->index;
            lay->sel_name_proc[0] = '\0';
        } else if (lay->focus == OV_FOCUS_FPS && tn->type == OV_NODE_FPS) {
            lay->sel_fps = tn->index;
            lay->sel_name_fps[0] = '\0';
            lay->fps_param_sel = 0;
            lay->fps_param_scroll = 0;
            lay->fps_param_focus = 0;
        }
    }
    return 1;
}

static int ov_input__handle_navigation(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    int *sel    = NULL;
    int *scroll = NULL;
    int  count  = 0;
    int  page_h = 10;

    /* -------------------------------------------------------
     * F5 view (OV_VIEW_FPS) param-tree intercept.
     *
     * When fps_param_focus == 1 (right-side param panel
     * is active), all navigation keys are consumed by the
     * param tree. RIGHT / ENTER from the list side switch
     * focus to the param panel.
     * ------------------------------------------------------- */
    if (lay->view == OV_VIEW_FPS)
    {
        int fsel = lay->sel_fps;
        int has_params =
            (fsel >= 0
             && fsel < m->nb_fps
             && m->fps[fsel].nb_disp_params > 0);

        int nitems = 0;
        fps_tree_item_t items[1024];
        if (has_params)
        {
            nitems = ov_get_fps_tree_items(&m->fps[fsel], lay->fps_param_path, items, 1024);
        }

        /* RIGHT from list → enter param panel */
        if (lay->fps_param_focus == 0
            && (key == OV_KEY_RIGHT || key == OV_KEY_ENTER || key == '\r' || key == '\n')
            && has_params)
        {
            lay->fps_param_focus = 1;
            lay->fps_param_sel   = 0;
            lay->fps_param_scroll = 0;
            return 1;
        }

        /* All nav/edit keys when param panel is focused */
        if (lay->fps_param_focus == 1)
        {
            /* ESC / LEFT from param panel → back to list or ascend dir */
            if (key == OV_KEY_LEFT || key == OV_KEY_ESC)
            {
                if (lay->fps_param_path[0] == '\0')
                {
                    lay->fps_param_focus = 0;
                }
                else
                {
                    /* Ascend directory */
                    char *last_dot = strrchr(lay->fps_param_path, '.');
                    if (last_dot)
                    {
                        *last_dot = '\0';
                    }
                    else
                    {
                        lay->fps_param_path[0] = '\0';
                    }
                    lay->fps_param_sel = 0;
                    lay->fps_param_scroll = 0;
                }
                return 1;
            }

            int ph = lay->r_fps_params.height - 3;
            if (ph < 1)
            {
                ph = 1;
            }

            if (key == OV_KEY_UP)
            {
                if (lay->fps_param_sel > 0)
                {
                    lay->fps_param_sel--;
                }
                return 1;
            }
            if (key == OV_KEY_DOWN)
            {
                if (lay->fps_param_sel < nitems - 1)
                {
                    lay->fps_param_sel++;
                }
                return 1;
            }
            if (key == OV_KEY_PGUP)
            {
                lay->fps_param_sel -= ph;
                if (lay->fps_param_sel < 0)
                {
                    lay->fps_param_sel = 0;
                }
                return 1;
            }
            if (key == OV_KEY_PGDN)
            {
                lay->fps_param_sel += ph;
                if (lay->fps_param_sel >= nitems)
                {
                    lay->fps_param_sel = nitems - 1;
                }
                return 1;
            }
            if (key == OV_KEY_HOME)
            {
                lay->fps_param_sel = 0;
                return 1;
            }
            if (key == OV_KEY_END)
            {
                lay->fps_param_sel = nitems - 1;
                if (lay->fps_param_sel < 0)
                {
                    lay->fps_param_sel = 0;
                }
                return 1;
            }
            if (key == OV_KEY_RIGHT || key == OV_KEY_ENTER || key == '\r' || key == '\n')
            {
                if (lay->fps_param_sel >= 0 && lay->fps_param_sel < nitems)
                {
                    fps_tree_item_t *item = &items[lay->fps_param_sel];
                    if (item->is_dir)
                    {
                        /* Descend directory */
                        if (lay->fps_param_path[0] == '\0')
                        {
                            strncpy(lay->fps_param_path, item->name, sizeof(lay->fps_param_path) - 1);
                        }
                        else
                        {
                            char tmp[200];
                            snprintf(tmp, sizeof(tmp), "%s.%s", lay->fps_param_path, item->name);
                            strncpy(lay->fps_param_path, tmp, sizeof(lay->fps_param_path) - 1);
                        }
                        lay->fps_param_sel = 0;
                        lay->fps_param_scroll = 0;
                    }
                    else if (key == OV_KEY_ENTER || key == '\r' || key == '\n')
                    {
                        if (!lay->ctrl_mode)
                        {
                            ov_cmdlog_push(
                                &lay->cmdlog,
                                OV_CMDLOG_WARN,
                                "Edit requires CONTROL mode (press c to toggle CTRL mode ON/OFF)");
                        }
                        else
                        {
                            ov_fps_inline_edit(
                                lay,
                                m->fps[fsel].name,
                                item->param_idx);
                        }
                    }
                }
                return 1;
            }
            return 0; /* pass through unmapped keys */
        }
    } /* if OV_VIEW_FPS */

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
        if (lay->graph_tab_mode == 0) {
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
        } else if (lay->graph_tab_mode == 1) {
            /* DETAILS tab: param nav if FPS selected */
            int fps_sel = lay->freeze
                ? lay->freeze_sel_fps
                : lay->sel_fps;
            int has_fps_params =
                (fps_sel >= 0
                 && fps_sel < m->nb_fps
                 && m->fps[fps_sel].nb_disp_params > 0);
            int nparams = has_fps_params
                ? m->fps[fps_sel].nb_disp_params : 0;

            if (has_fps_params && lay->param_sel >= 0)
            {
                /* Navigate parameter cursor */
                if (key == OV_KEY_UP)
                {
                    if (lay->param_sel > 0)
                    {
                        lay->param_sel--;
                    }
                }
                else if (key == OV_KEY_DOWN)
                {
                    if (lay->param_sel
                        < nparams - 1)
                    {
                        lay->param_sel++;
                    }
                }
                else if (key == OV_KEY_PGUP)
                {
                    lay->param_sel -= page_h;
                    if (lay->param_sel < 0)
                    {
                        lay->param_sel = 0;
                    }
                }
                else if (key == OV_KEY_PGDN)
                {
                    lay->param_sel += page_h;
                    if (lay->param_sel
                        >= nparams)
                    {
                        lay->param_sel =
                            nparams - 1;
                    }
                }
                else if (key == OV_KEY_HOME)
                {
                    lay->param_sel = 0;
                }
                else if (key == OV_KEY_END)
                {
                    lay->param_sel =
                        nparams - 1;
                    if (lay->param_sel < 0)
                    {
                        lay->param_sel = 0;
                    }
                }
                else if (key == OV_KEY_ESC)
                {
                    lay->param_sel = -1;
                }
                else if (key == OV_KEY_ENTER
                         || key == '\r' || key == '\n')
                {
                    if (!lay->ctrl_mode)
                    {
                        ov_cmdlog_push(
                            &lay->cmdlog,
                            OV_CMDLOG_WARN,
                            "Edit requires "
                            "CONTROL mode (press c to toggle CTRL mode ON/OFF)");
                    }
                    else
                    {
                        ov_fps_inline_edit(
                            lay,
                            m->fps[fps_sel].name,
                            lay->param_sel);
                    }
                }
                return 1;
            }

            /* Init param_sel on first nav key */
            if (has_fps_params
                && lay->param_sel < 0
                && (key == OV_KEY_DOWN
                    || key == OV_KEY_UP))
            {
                lay->param_sel = 0;
                return 1;
            }

            /* Fallback: detail scroll */
            sel = NULL;
            scroll = NULL;
            count = lay->detail_total_lines;
            page_h = lay->r_graph.height - 3;

            if (key == OV_KEY_UP) {
                if (lay->scroll_detail > 0)
                    lay->scroll_detail--;
            } else if (key == OV_KEY_DOWN) {
                if (lay->scroll_detail
                    < lay->detail_total_lines
                      - page_h)
                    lay->scroll_detail++;
            } else if (key == OV_KEY_PGUP) {
                lay->scroll_detail -= page_h;
                if (lay->scroll_detail < 0)
                    lay->scroll_detail = 0;
            } else if (key == OV_KEY_PGDN) {
                lay->scroll_detail += page_h;
                if (lay->scroll_detail
                    > lay->detail_total_lines
                      - page_h)
                {
                    lay->scroll_detail =
                        lay->detail_total_lines
                        - page_h;
                }
                if (lay->scroll_detail < 0)
                    lay->scroll_detail = 0;
            } else if (key == OV_KEY_HOME) {
                lay->scroll_detail = 0;
            } else if (key == OV_KEY_END) {
                lay->scroll_detail =
                    lay->detail_total_lines
                    - page_h;
                if (lay->scroll_detail < 0)
                    lay->scroll_detail = 0;
            }
            return 1;
        } else {
            sel = NULL;
            scroll = NULL;
        }
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
                if (lay->focus == OV_FOCUS_STREAMS) lay->sel_name_stream[0] = '\0';
                else if (lay->focus == OV_FOCUS_PROCS) lay->sel_name_proc[0] = '\0';
                else if (lay->focus == OV_FOCUS_FPS)
                {
                    lay->sel_name_fps[0] = '\0';
                    /* Reset param tree cursor when FPS selection moves */
                    if (lay->view == OV_VIEW_FPS)
                    {
                        lay->fps_param_sel    = 0;
                        lay->fps_param_scroll = 0;
                        lay->fps_param_focus  = 0;
                    }
                }

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
                                lay->sel_name_stream[0] = '\0';
                                if (lay->sel_stream < lay->scroll_stream) lay->scroll_stream = lay->sel_stream;
                                if (lay->r_streams.height > 3 && lay->sel_stream >= lay->scroll_stream + lay->r_streams.height - 3) {
                                    lay->scroll_stream = lay->sel_stream - (lay->r_streams.height - 3) + 1;
                                }
                            } else if (node->type == OV_NODE_PROC) {
                                lay->sel_proc = node->index;
                                lay->sel_name_proc[0] = '\0';
                                if (lay->sel_proc < lay->scroll_proc) lay->scroll_proc = lay->sel_proc;
                                if (lay->r_procs.height > 3 && lay->sel_proc >= lay->scroll_proc + lay->r_procs.height - 3) {
                                    lay->scroll_proc = lay->sel_proc - (lay->r_procs.height - 3) + 1;
                                }
                            } else if (node->type == OV_NODE_FPS) {
                                lay->sel_fps = node->index;
                                lay->sel_name_fps[0] = '\0';
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

    if (key == 'c')
    {
        lay->ctrl_mode = !lay->ctrl_mode;
        ov_cmdlog_push(&lay->cmdlog,
                       OV_CMDLOG_INFO,
                       "Control mode %s",
                       lay->ctrl_mode
                       ? "ON" : "OFF");
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
    
    if (ov_input__handle_ancestry_nav(key, lay, m) || ov_input__handle_navigation(key, lay, m))
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

    if (key == '?')
    {
        char keys_str[128];
        char ctrl_str[128] = "";

        // Base keys always available
        snprintf(keys_str, sizeof(keys_str), "Keys: q c h G v V ? TAB D L F W i <>[]sS +-= F2-F6 / ESC");

        if (lay->ctrl_mode)
        {
            if (lay->focus == OV_FOCUS_FPS)
                snprintf(ctrl_str, sizeof(ctrl_str), " | CTRL (FPS): x r s k K ^e");
            else if (lay->focus == OV_FOCUS_PROCS)
                snprintf(ctrl_str, sizeof(ctrl_str), " | CTRL (PROC): p ^s e z C k K ^e");
            else if (lay->focus == OV_FOCUS_STREAMS)
                snprintf(ctrl_str, sizeof(ctrl_str), " | CTRL (STRM): DEL ^e");
        }

        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO, "⌨️ %s%s", keys_str, ctrl_str);
        if (lay->cmdlog_rows == 0) lay->cmdlog_rows = 4;
        return 0;
    }

    if (key >= 32 && key <= 126) {
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN, "❔ Unmapped key: '%c' (code %d)", key, key);
    } else {
        ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN, "❔ Unmapped key code: %d", key);
    }
    return 0;
}
