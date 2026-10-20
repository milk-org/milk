// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_sort_keys.c
 * @brief Keyboard shortcuts for column highlighting, column collapse, and sorting
 */

#include "overview_input_internal.h"

/** list_panel - state of the focused list panel, or NULL (sub-tab panel). */
static OV_PANEL_STATE *list_panel(OV_LAYOUT *lay)
{
    return (lay->focus < OV_NB_LIST_PANELS) ? &lay->panel[lay->focus] : NULL;
}

/** col_cursor - Shift+LEFT / RIGHT: move the column cursor over the visible columns. */
static int col_cursor(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) m;
    OV_PANEL_STATE *ps = list_panel(lay);
    if (ps != NULL)
    {
        int vis[OV_TABLE_MAX_COLS];
        int num = ov_table_visible(ov_table_for(lay->focus), lay, vis);
        ps->highlight_col =
            ((ps->highlight_col % num) + ((key == OV_KEY_SHIFT_LEFT) ? -1 : 1) + num) % num;
    }
    return 1;
}

/** col_collapse - t / T: hide / show the column under the cursor (collapse bits are table indices). */
static int col_collapse(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) m;
    OV_PANEL_STATE *ps = list_panel(lay);
    if (ps != NULL)
    {
        int vis[OV_TABLE_MAX_COLS];
        int nv = ov_table_visible(ov_table_for(lay->focus), lay, vis);
        if (ps->highlight_col >= 0 && ps->highlight_col < nv)
        {
            ps->collapsed ^= (1U << vis[ps->highlight_col]);
        }
    }
    return 1;
}

/** sort - s name, S rate, A ancestry, < > ] previous / next sortable column, [ reverse. */
static int sort(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    /* Per panel (ov_focus_t): column 'S' sorts by ('A' sorts by "anc") */
    static const char *const rate_id[OV_NB_LIST_PANELS] = { "hz", "hz", "cpid" };
    (void) m;
    OV_PANEL_STATE *ps = list_panel(lay);
    if (ps != NULL)
    {
        const OV_TABLE *t = ov_table_for(lay->focus);
        switch (key)
        {
        case 'S':
            ps->sort_key = t->cols[ov_table_find_id(t, rate_id[lay->focus])].sort_key;
            break;
        case 'A':
            ps->sort_key = t->cols[ov_table_find_id(t, "anc")].sort_key;
            ps->sort_dir = 0;
            break;
        case 's':
            ps->sort_key = 0;
            break;
        case '[':
            ps->sort_dir = !ps->sort_dir;
            break;
        case '<':
            ps->sort_key = ov_table_next_sort(t, lay, ps->sort_key, -1);
            break;
        default: /* > ] skip hidden and unsortable columns */
            ps->sort_key = ov_table_next_sort(t, lay, ps->sort_key, +1);
            break;
        }
    }
    lay->sort_pending = 1;
    return 1;
}

static const OV_KEYBIND column_keys[] = {
    { { OV_KEY_SHIFT_LEFT, OV_KEY_SHIFT_RIGHT }, 0, 0, col_cursor },
    { { 't', 'T' }, 0, 0, col_collapse },
    { { 'S', 'A', '>', ']', '<', '[' }, 0, 0, sort },
    { { 's' }, 0, OV_CTX_CTRL, sort }, /* CONTROL mode, PROC / FPS: 's' stops / starts */
    { { 's' }, OV_CTX_CTRL | OV_CTX_FOCUS_STREAMS, 0, sort },
    { { 's' }, OV_CTX_CTRL | OV_CTX_FOCUS_GRAPH, 0, sort },
};

/** ov_input__handle_column_keys - column cursor, hide / show, sorting. Return: 1 consumed. */
int ov_input__handle_column_keys(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    return ov_key_dispatch(column_keys, OV_KEYBIND_N(column_keys), key, lay, m);
}
