// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_subtab.c
 * @brief Clicks and keys of the sub-tab panel (DASH bottom-right, F6) and of F7's loop list
 *
 * Sub-tab keys run before the generic toggles, so ENTER edits a DETAILS parameter instead of toggling
 * the sub-tab, and selects the CONNECTIONS row's stream.
 */

#include "overview_input_internal.h"

/** utf8_cols - number of code points in @s (tree glyphs and names are one column each). */
static int utf8_cols(const char *s)
{
    int n = 0;
    for (int i = 0; s[i] != '\0'; i += utf8_char_length((unsigned char) s[i]))
    {
        n++;
    }
    return n;
}

/** jump_to - leave the sub-tab for DASH with list panel @f on model entry @idx. */
static void jump_to(OV_LAYOUT *lay, const OV_MODEL *m, ov_focus_t f, int idx)
{
    lay->freeze = 0; /* before resolving: freeze changes the drawn rows */
    ov_set_view(lay, OV_VIEW_DASHBOARD);
    lay->focus = f;
    ov_sel_model(lay, m, f, idx);
}

/**
 * conn_click - CONNECTIONS click on @body_row: move the graph cursor (the root stays pinned). A double
 * click selects the row's stream in DASH, or its reader process when clicked right of the name.
 */
static void conn_click(OV_LAYOUT *lay, const OV_MODEL *m, int body_row, int mc, int is_dbl)
{
    int                 n;
    const SG_TREE_NODE *rows = ov_graph_rows(lay, m, &n);
    int                 idx  = lay->sub[OV_SUB_CONN].scroll + body_row;
    if (idx < 0 || idx >= n)
    {
        return;
    }
    lay->sub[OV_SUB_CONN].sel = idx;
    if (is_dbl)
    {
        const SG_TREE_NODE *rn = &rows[idx];
        int name_end = utf8_cols(rn->tree_prefix) + (rn->is_target ? 2 : 0) + utf8_cols(rn->name);
        int pi       = sg_tree_reader_proc(m, rn);
        int on_proc  = (pi >= 0 && mc - lay->r_graph.col - 1 > name_end + 1);
        jump_to(lay, m, on_proc ? OV_FOCUS_PROCS : OV_FOCUS_STREAMS, on_proc ? pi : rn->stream_idx);
    }
}

/** loops_click - loop list click on @body_row; a double click in DASH opens F7 on that loop. */
static void loops_click(OV_LAYOUT *lay, const OV_MODEL *m, int body_row, int mc, int is_dbl)
{
    int li = lay->sub[OV_SUB_LOOPS].scroll + body_row;
    if (body_row >= ov_loop_list_rows(lay) || li < 0 || li >= m->nb_loops)
    {
        return;
    }
    if (lay->view == OV_VIEW_LOOPS && mc > ov_loops_list_width(lay))
    {
        return; /* F7's circuit pane */
    }
    lay->sub[OV_SUB_LOOPS].sel = li;
    if (is_dbl && lay->view != OV_VIEW_LOOPS)
    {
        ov_set_view(lay, OV_VIEW_LOOPS);
    }
}

/** details_fps - FPS whose parameters DETAILS lists, or -1 (a stream, a process, or no parameters). */
static int details_fps(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    ov_focus_t panel = OV_FOCUS_STREAMS;
    int        fi    = ov_detail_target(lay, m, &panel);
    return (panel == OV_FOCUS_FPS && fi >= 0 && fi < m->nb_fps && m->fps[fi].nb_disp_params > 0)
               ? fi
               : -1;
}

/** details_click - click on a DETAILS parameter: select it; a stream parameter selects that stream. */
static void details_click(OV_LAYOUT *lay, const OV_MODEL *m, int body_row, int mc, int is_dbl)
{
    (void) mc;
    int fi = details_fps(lay, m);
    if (fi < 0)
    {
        return;
    }
    const OV_FPS *f  = &m->fps[fi];
    int           dp = lay->sub[OV_SUB_DETAILS].scroll + body_row - 3 - (f->description[0] != '\0');
    const OV_FPS_PARAMS *params = ov_fps_get_params(f->name);
    if (params == NULL || dp < 0 || dp >= params->nb_disp_params)
    {
        return;
    }
    lay->sub[OV_SUB_DETAILS].sel = dp;
    int si                       = (params->disp_param_type[dp] == FPTYPE_STREAMNAME)
                                       ? ov_find_stream_by_name(m, params->disp_param_value[dp])
                                       : -1;
    if (si >= 0)
    {
        lay->focus = OV_FOCUS_STREAMS;
        ov_sel_model(lay, m, OV_FOCUS_STREAMS, si);
        if (is_dbl)
        {
            ov_set_view(lay, OV_VIEW_STREAMS);
        }
    }
}

/* Body click of each sub-tab, indexed by ov_subtab_t (RESOURCES has none) */
static void (*const subtab_click[OV_SUBTAB_COUNT])(OV_LAYOUT *, const OV_MODEL *, int, int, int) = {
    [OV_SUB_CONN]    = conn_click,
    [OV_SUB_LOOPS]   = loops_click,
    [OV_SUB_DETAILS] = details_click,
};

/**
 * ov_input__subtab_click - click at (@mr, @mc) in the sub-tab panel: a label on its top border switches
 * sub-tab, a body row goes to that sub-tab's handler (F7: the loop list).
 */
void ov_input__subtab_click(OV_LAYOUT *lay, const OV_MODEL *m, int mr, int mc, int is_dbl)
{
    lay->focus = OV_FOCUS_GRAPH;
    if (lay->view != OV_VIEW_LOOPS && mr == lay->r_graph.row)
    {
        if (ov_subtab_hit(mc) >= 0)
        {
            lay->subtab = (ov_subtab_t) ov_subtab_hit(mc);
            ov_scan_force_update();
        }
        return;
    }
    int         body_row = mr - lay->r_graph.row - 2;
    ov_subtab_t t        = (lay->view == OV_VIEW_LOOPS) ? OV_SUB_LOOPS : lay->subtab;
    if (body_row >= 0 && (int) t >= 0 && t < OV_SUBTAB_COUNT && subtab_click[t] != NULL)
    {
        subtab_click[t](lay, m, body_row, mc, is_dbl);
    }
}

/** move_cursor - apply arrow/page/home/end @key to *@sel over @count rows. Return: 1 for those keys. */
static int move_cursor(int key, int *sel, int count, int page)
{
    switch (key)
    {
    case OV_KEY_UP:
        *sel -= (*sel > 0);
        return 1;
    case OV_KEY_DOWN:
        *sel += (*sel < count - 1);
        return 1;
    case OV_KEY_PGUP:
        *sel = (*sel - page < 0) ? 0 : *sel - page;
        return 1;
    case OV_KEY_PGDN:
        *sel = (*sel + page >= count) ? count - 1 : *sel + page;
        return 1;
    case OV_KEY_HOME:
        *sel = 0;
        return 1;
    case OV_KEY_END:
        *sel = (count > 0) ? count - 1 : 0;
        return 1;
    default:
        return 0;
    }
}

/** details_keys - DETAILS parameter cursor: arrows move it (the first arrow shows it), ENTER edits. */
static int details_keys(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    int  fi  = details_fps(lay, m);
    int *sel = &lay->sub[OV_SUB_DETAILS].sel;
    if (fi < 0)
    {
        return 0;
    }
    if (*sel < 0)
    {
        *sel = (key == OV_KEY_DOWN || key == OV_KEY_UP) ? 0 : -1;
        return *sel == 0;
    }
    if (key == OV_KEY_ENTER || key == '\r' || key == '\n')
    {
        if (!lay->ctrl_mode)
        {
            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN,
                           "Edit requires CONTROL mode (press c to toggle CTRL mode ON/OFF)");
        }
        else
        {
            ov_fps_inline_edit(lay, m->fps[fi].name, *sel);
        }
        return 1;
    }
    return move_cursor(key, sel, m->fps[fi].nb_disp_params, ov_subtab_page(lay));
}

/** conn_keys - ENTER selects the CONNECTIONS row's stream in DASH. */
static int conn_keys(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (key != '\n' && key != '\r')
    {
        return 0;
    }
    int                 n;
    const SG_TREE_NODE *rows = ov_graph_rows(lay, m, &n);
    if (lay->sub[OV_SUB_CONN].sel >= 0 && lay->sub[OV_SUB_CONN].sel < n)
    {
        jump_to(lay, m, OV_FOCUS_STREAMS, rows[lay->sub[OV_SUB_CONN].sel].stream_idx);
    }
    return 1;
}

/* Keys of each sub-tab, indexed by ov_subtab_t; they run before the generic key groups */
static int (*const subtab_keys[OV_SUBTAB_COUNT])(int, OV_LAYOUT *, const OV_MODEL *) = {
    [OV_SUB_CONN]    = conn_keys,
    [OV_SUB_DETAILS] = details_keys,
};

/** ov_input__handle_subtab_keys - keys of the focused sub-tab. Return: 1 if consumed. */
int ov_input__handle_subtab_keys(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (m == NULL || lay->focus != OV_FOCUS_GRAPH || lay->view == OV_VIEW_LOOPS ||
        !ov_view_has_panel(lay->view, OV_FOCUS_GRAPH) || (int) lay->subtab < 0 ||
        lay->subtab >= OV_SUBTAB_COUNT || subtab_keys[lay->subtab] == NULL)
    {
        return 0;
    }
    return subtab_keys[lay->subtab](key, lay, m);
}
