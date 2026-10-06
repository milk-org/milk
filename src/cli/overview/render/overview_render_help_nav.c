// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_help_nav.c
 * @brief   Help overlay navigation, expansion, and mouse hit testing for milk-CTRL.
 */

#include "overview_render_help_internal.h"
#include <stdio.h>
#include <string.h>

/**
 * ov_help_visible_count - public helper returning number of visible rows.
 * @lay: layout state
 *
 * Return: visible row count.
 */
int ov_help_visible_count(const OV_LAYOUT *lay)
{
    int map[128];
    return help_visible_rows(lay, map);
}

/**
 * ov_help_section_first_vis_row - find visible row index for a section header.
 * @lay: layout state
 * @sec: section index
 *
 * Return: visible row index in map[], or 0 if not found.
 */
int ov_help_section_first_vis_row(const OV_LAYOUT *lay, int sec)
{
    int map[128];
    int nvis = help_visible_rows(lay, map);
    for (int vr = 0; vr < nvis; vr++)
    {
        int idx = map[vr];
        if (g_help_entries[idx].section == sec && (g_help_entries[idx].flags & HF_SECTION))
        {
            return vr;
        }
    }
    return 0;
}

/**
 * ov_help_open - open help overlay with contextual initial focus and expansion.
 * @lay: layout state (modified)
 */
void ov_help_open(OV_LAYOUT *lay)
{
    lay->show_help          = 1;
    lay->help_expand        = 0;
    lay->filter_editing     = 0;
    lay->help_search[0]     = '\0';
    lay->help_search_active = 0;
    lay->help_search_cursor = 0;
    lay->help_mode          = 0;
    lay->help_intro_scroll  = 0;
    lay->help_sel           = 0;
}

/**
 * ov_help_toggle_at - toggle expansion for section header at visible row.
 * @lay:     layout state (help_expand bitmask modified)
 * @vis_row: 0-based visible row index
 *
 * Return: section index if toggled, or -1 if row is not a section header.
 */
int ov_help_toggle_at(OV_LAYOUT *lay, int vis_row)
{
    if (lay->help_search[0] != '\0')
    {
        return -1;
    }

    int map[128];
    int nvis = help_visible_rows(lay, map);

    if (vis_row < 0 || vis_row >= nvis)
    {
        return -1;
    }

    int idx = map[vis_row];
    if (!(g_help_entries[idx].flags & HF_SECTION))
    {
        return -1;
    }

    int sec = g_help_entries[idx].section;
    lay->help_expand ^= (1U << sec);
    return sec;
}

/**
 * ov_help_expand_at - expand or collapse section at visible row.
 * @lay:     layout state (help_expand bitmask modified)
 * @vis_row: 0-based visible row index
 * @expand:  1 to expand, 0 to collapse
 *
 * Return: section index if modified, or -1 if row is not eligible.
 */
int ov_help_expand_at(OV_LAYOUT *lay, int vis_row, int expand)
{
    if (lay->help_search[0] != '\0')
    {
        return -1;
    }

    int map[128];
    int nvis = help_visible_rows(lay, map);

    if (vis_row < 0 || vis_row >= nvis)
    {
        return -1;
    }

    int idx = map[vis_row];
    int sec = g_help_entries[idx].section;

    if (expand)
    {
        if (g_help_entries[idx].flags & HF_SECTION)
        {
            if (!help_is_expanded(lay, sec))
            {
                lay->help_expand |= (1U << sec);
                return sec;
            }
            else
            {
                int new_nvis = ov_help_visible_count(lay);
                if (lay->help_sel + 1 < new_nvis)
                {
                    lay->help_sel++;
                }
                return sec;
            }
        }
    }
    else
    {
        if (g_help_entries[idx].flags & HF_SECTION)
        {
            if (help_is_expanded(lay, sec))
            {
                lay->help_expand &= ~(1U << sec);
                return sec;
            }
        }
        else
        {
            /* On child item: collapse parent section and land on section header */
            lay->help_expand &= ~(1U << sec);
            lay->help_sel = ov_help_section_first_vis_row(lay, sec);
            return sec;
        }
    }

    return -1;
}

/**
 * ov_help_get_rect - compute bounding box for help overlay.
 * @lay: layout state
 * @pr:  output top row (1-based)
 * @pc:  output left column (1-based)
 * @ph:  output height in rows
 * @pw:  output width in columns
 *
 * Takes the whole available terminal space: starts below the dedicated tab bar
 * (row 3) and extends across the entire terminal width to the row above the status bar.
 */
void ov_help_get_rect(const OV_LAYOUT *lay, int *pr, int *pc, int *ph, int *pw)
{
    int W = lay->term_cols;
    int H = lay->term_rows;

    *pc = 1;
    *pw = W;

    if (H <= 6)
    {
        *pr = 1;
        *ph = H;
    }
    else
    {
        *pr = 3;
        *ph = H - 3;
    }
}

/**
 * ov_help_handle_click - handle mouse click within help overlay.
 * @lay: layout state
 * @mr:  clicked terminal row (1-based)
 * @mc:  clicked terminal column (1-based)
 *
 * Return: 1 if click was handled inside help overlay, 0 otherwise.
 */
int ov_help_handle_click(OV_LAYOUT *lay, int mr, int mc)
{
    int pr, pc, ph, pw;
    ov_help_get_rect(lay, &pr, &pc, &ph, &pw);

    /* Click outside popup -> close help overlay */
    if (mr < pr || mr >= pr + ph || mc < pc || mc >= pc + pw)
    {
        lay->show_help = 0;
        ov_buf_force_clear();
        if (mr == lay->r_tabs.row)
        {
            int tx = 1;
            for (int v = 0; v < OV_VIEW_COUNT; v++)
            {
                int tw = (int) strlen(ov_view_label((ov_view_t) v)) + 9;
                if (mc >= tx && mc < tx + tw)
                {
                    lay->view = (ov_view_t) v;
                    break;
                }
                tx += tw;
            }
        }
        return 1;
    }

    /* Click close button area on top border or header row */
    if ((mr == pr && mc >= pc + pw - 6) || (mr == pr + 1 && mc >= pc + pw - 6))
    {
        lay->show_help = 0;
        ov_buf_force_clear();
        return 1;
    }

    /* Click on header row (pr + 1): Mode selector tabs */
    if (mr == pr + 1)
    {
        int tab1_w = (pw >= 100) ? 26 : 14;
        int tab2_w = (pw >= 100) ? 32 : 20;
        if (mc >= pc + 2 && mc < pc + 2 + tab1_w)
        {
            lay->help_mode         = 1;
            lay->help_intro_scroll = 0;
            return 1;
        }
        if (mc >= pc + 2 + tab1_w && mc < pc + 2 + tab1_w + tab2_w)
        {
            lay->help_mode = 0;
            return 1;
        }
    }

    /* If in Intro mode (help_mode == 1), clicks in body do not alter list */
    if (lay->help_mode == 1)
    {
        return 1;
    }

    /* Click on search bar area on row pr + 2 (in Controls mode) */
    if (mr == pr + 2 && mc >= pc + 1 && mc < pc + pw - 1)
    {
        if (lay->help_search[0] != '\0' && mc >= pc + pw - 16)
        {
            /* Clicked on [ESC: clear] button */
            lay->help_search[0]     = '\0';
            lay->help_search_cursor = 0;
            lay->help_search_active = 0;
            lay->help_sel           = 0;
        }
        else
        {
            /* Clicked on search input box */
            lay->help_search_active = 1;
            lay->help_search_cursor = (int) strlen(lay->help_search);
        }
        return 1;
    }

    /* Detail pane height and split line */
    int detail_h = (ph >= 36) ? 10 : ((ph >= 28) ? 8 : ((ph >= 22) ? 6 : 5));
    int split_r  = (pr + ph - 1) - detail_h;
    int list_top = pr + 4;
    int list_h   = split_r - list_top;

    /* Check if click is inside the list area */
    if (mr >= list_top && mr < split_r)
    {
        int map[128];
        int nvis = help_visible_rows(lay, map);

        int sel = lay->help_sel;
        if (sel < 0)
        {
            sel = 0;
        }
        if (sel >= nvis)
        {
            sel = nvis - 1;
        }

        int scroll = 0;
        if (sel >= list_h)
        {
            scroll = sel - list_h + 1;
        }

        int vis_row = (mr - list_top) + scroll;
        if (vis_row >= 0 && vis_row < nvis)
        {
            int idx = map[vis_row];
            if (g_help_entries[idx].section == HS_INTRO &&
                (lay->help_sel == vis_row || mr == list_top))
            {
                /* Clicking on Introduction entry/header opens full intro guide */
                lay->help_mode         = 1;
                lay->help_intro_scroll = 0;
                return 1;
            }
            if (lay->help_search[0] == '\0' && lay->help_sel == vis_row)
            {
                /* Clicking selected header toggles expansion */
                ov_help_toggle_at(lay, vis_row);
            }
            else
            {
                lay->help_sel = vis_row;
            }
            return 1;
        }
    }

    return 0;
}
