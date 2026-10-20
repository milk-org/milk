// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_sort_keys.c
 * @brief Keyboard shortcuts for column highlighting, column collapse, and sorting
 */

#include "overview_input_internal.h"

/**
 * ov_input__handle_column_highlights - handle column highlight cycling shortcuts.
 * @key: Pressed key code.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: 1 if key was consumed, 0 otherwise.
 */
int ov_input__handle_column_highlights(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) m;

    if (key == OV_KEY_SHIFT_LEFT || key == OV_KEY_SHIFT_RIGHT)
    {
        int dir = (key == OV_KEY_SHIFT_LEFT) ? -1 : 1;
        if (lay->focus < OV_NB_LIST_PANELS)
        {
            OV_PANEL_STATE *ps = &lay->panel[lay->focus];
            int             vis[OV_TABLE_MAX_COLS];
            int             num = ov_table_visible(ov_table_for(lay->focus), lay, vis);
            ps->highlight_col   = ((ps->highlight_col % num) + dir + num) % num;
        }
        return 1;
    }

    if (key == 't' || key == 'T')
    {
        /* collapse bits are table indices; the cursor counts visible columns */
        if (lay->focus < OV_NB_LIST_PANELS)
        {
            OV_PANEL_STATE *ps = &lay->panel[lay->focus];
            int             vis[OV_TABLE_MAX_COLS];
            int             nv = ov_table_visible(ov_table_for(lay->focus), lay, vis);
            if (ps->highlight_col >= 0 && ps->highlight_col < nv)
            {
                ps->collapsed ^= (1U << vis[ps->highlight_col]);
            }
        }
        return 1;
    }

    return 0;
}

/**
 * ov_input__handle_sorting - handle column sorting shortcuts and direction toggles.
 * @key: Pressed key code.
 * @lay: Pointer to layout structure.
 *
 * Return: 1 if key was consumed, 0 otherwise.
 */
int ov_input__handle_sorting(int key, OV_LAYOUT *lay)
{
    /* Per panel (ov_focus_t): column 'S' sorts by ('A' sorts by "anc") */
    static const char *const rate_id[OV_NB_LIST_PANELS] = { "hz", "hz", "cpid" };

    int             list = (lay->focus < OV_NB_LIST_PANELS);
    OV_PANEL_STATE *ps   = list ? &lay->panel[lay->focus] : NULL;

    if (key == 'S' || key == 'A' || key == '>' || key == ']' || key == '<' || key == '[' ||
        (key == 's' &&
         !(lay->ctrl_mode && (lay->focus == OV_FOCUS_FPS || lay->focus == OV_FOCUS_PROCS))))
    {
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
    return 0;
}
