// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_nav.c
 * @brief Cursor movement, scrolling, and page navigation across panels and graph
 */

#include "overview_input_internal.h"

/** ov_input__detail_scroll_to - make line @top the first one of the DETAILS / RESOURCES text shown. */
void ov_input__detail_scroll_to(OV_LAYOUT *lay, int top)
{
    int last           = lay->detail_total_lines - (lay->r_graph.height - 3);
    last               = (last > 0) ? last : 0;
    lay->scroll_detail = (top < 0) ? 0 : (top > last) ? last : top;
}

/** detail_scroll - scroll DETAILS / RESOURCES text by line, page, or to an end. Return: 1 for those keys. */
static int detail_scroll(int key, OV_LAYOUT *lay)
{
    int page = lay->r_graph.height - 3;
    int top  = lay->scroll_detail;
    switch (key)
    {
    case OV_KEY_UP:
        top--;
        break;
    case OV_KEY_DOWN:
        top++;
        break;
    case OV_KEY_PGUP:
        top -= page;
        break;
    case OV_KEY_PGDN:
        top += page;
        break;
    case OV_KEY_HOME:
        top = 0;
        break;
    case OV_KEY_END:
        top = lay->detail_total_lines;
        break;
    default:
        return 0;
    }
    ov_input__detail_scroll_to(lay, top);
    return 1;
}

/**
 * ov_input__handle_navigation - dispatch cursor keys, paging, and scroll navigation.
 * @key: Pressed key code (arrows, Home/End, PgUp/PgDn, etc.).
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: 1 if navigation key was consumed, 0 otherwise.
 */
int ov_input__handle_navigation(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    int *sel                            = NULL;
    int *scroll __attribute__((unused)) = NULL;
    int  count                          = 0;
    int  page_h                         = 10;

    /* F5 view (OV_VIEW_FPS) param-tree intercept */
    if (lay->view == OV_VIEW_FPS)
    {
        int rc = ov_input_nav_fps(key, lay, m);
        if (rc != 0)
        {
            return rc;
        }
    }

    switch (lay->focus)
    {
    case OV_FOCUS_STREAMS:
        sel    = &lay->panel[OV_FOCUS_STREAMS].sel;
        scroll = &lay->panel[OV_FOCUS_STREAMS].scroll;
        {
            int fidx[OV_MAX_STREAMS];
            count = ov_rows(lay, m, OV_FOCUS_STREAMS, fidx, OV_MAX_STREAMS);
        }
        page_h = lay->r_streams.height - 3;
        break;

    case OV_FOCUS_PROCS:
        sel    = &lay->panel[OV_FOCUS_PROCS].sel;
        scroll = &lay->panel[OV_FOCUS_PROCS].scroll;
        {
            int fidx[OV_MAX_PROCS];
            count = ov_rows(lay, m, OV_FOCUS_PROCS, fidx, OV_MAX_PROCS);
        }
        page_h = lay->r_procs.height - 3;
        break;

    case OV_FOCUS_FPS:
        sel    = &lay->panel[OV_FOCUS_FPS].sel;
        scroll = &lay->panel[OV_FOCUS_FPS].scroll;
        {
            int fidx[OV_MAX_FPS];
            count = ov_rows(lay, m, OV_FOCUS_FPS, fidx, OV_MAX_FPS);
        }
        page_h = lay->r_fps.height - 3;
        break;

    case OV_FOCUS_GRAPH:
        /* F7 always draws the loop list, whatever the DASH sub-tab */
        if (lay->view == OV_VIEW_LOOPS || lay->subtab == OV_SUB_LOOPS)
        {
            sel   = &lay->sub[OV_SUB_LOOPS].sel;
            count = m->nb_loops;
        }
        else if (lay->subtab == OV_SUB_CONN)
        {
            sel = &lay->sub[OV_SUB_CONN].sel;
            ov_graph_rows(lay, m, &count);
        }
        else
        {
            return detail_scroll(key, lay); /* DETAILS text (parameters: sub-tab keys), RESOURCES */
        }
        page_h = ov_subtab_page(lay);
        break;

    default:
        break;
    }

    if (sel != NULL)
    {
        int old_sel    = *sel;
        int navigated  = 0;
        int is_nav_key = 0;

        if (key == OV_KEY_UP)
        {
            is_nav_key = 1;
            if (*sel > 0)
            {
                (*sel)--;
                navigated = 1;
            }
        }
        else if (key == OV_KEY_DOWN)
        {
            is_nav_key = 1;
            if (*sel < count - 1)
            {
                (*sel)++;
                navigated = 1;
            }
        }
        else if (key == OV_KEY_PGUP)
        {
            is_nav_key = 1;
            *sel -= page_h;
            if (*sel < 0)
            {
                *sel = 0;
            }
            navigated = 1;
        }
        else if (key == OV_KEY_PGDN)
        {
            is_nav_key = 1;
            *sel += page_h;
            if (*sel >= count)
            {
                *sel = count - 1;
            }
            if (*sel < 0)
            {
                *sel = 0;
            }
            navigated = 1;
        }
        else if (key == OV_KEY_HOME)
        {
            is_nav_key = 1;
            *sel       = 0;
            navigated  = 1;
        }
        else if (key == OV_KEY_END)
        {
            is_nav_key = 1;
            *sel       = count - 1;
            if (*sel < 0)
            {
                *sel = 0;
            }
            navigated = 1;
        }

        if (is_nav_key)
        {
            if (navigated)
            {
                if (lay->focus < OV_FOCUS_GRAPH)
                {
                    ov_sel_row(lay, m, lay->focus,
                               *sel); /* the cursor is the identity of this row */
                }
                if (lay->focus == OV_FOCUS_FPS && lay->view == OV_VIEW_FPS)
                {
                    lay->fps_param_focus = 0;
                }
            }
            return 1;
        }
    }

    return 0;
}
