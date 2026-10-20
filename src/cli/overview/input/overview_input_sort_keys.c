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
            const OV_TABLE *t  = ov_table_for(lay->focus);
            int             vis[OV_TABLE_MAX_COLS];
            int             num =
                (t != NULL) ? ov_table_visible(t, lay, vis) : ov_get_num_cols(lay, lay->focus);
            ps->highlight_col = ((ps->highlight_col % num) + dir + num) % num;
        }
        return 1;
    }

    if ((key == 't' || key == 'T') && lay->focus < OV_NB_LIST_PANELS &&
        ov_table_for(lay->focus) != NULL)
    {
        /* collapse bits are table indices; the cursor counts visible columns */
        OV_PANEL_STATE *ps = &lay->panel[lay->focus];
        int             vis[OV_TABLE_MAX_COLS];
        int             nv = ov_table_visible(ov_table_for(lay->focus), lay, vis);
        if (ps->highlight_col >= 0 && ps->highlight_col < nv)
        {
            ps->collapsed ^= (1U << vis[ps->highlight_col]);
        }
        return 1;
    }
    if (key == 't' || key == 'T')
    {
        if (lay->focus == OV_FOCUS_STREAMS)
        {
            int logical_col = ov_get_logical_col_stream(lay->panel[OV_FOCUS_STREAMS].highlight_col,
                                                        lay->compact_mode);
            lay->panel[OV_FOCUS_STREAMS].collapsed ^= (1U << logical_col);
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
    /* Per panel (ov_focus_t): sort key of 'S' (rate) and 'A' (ancestry), and number of sort keys */
    static const int key_rate[OV_NB_LIST_PANELS] = { 3, 3, 1 };
    static const int key_anc[OV_NB_LIST_PANELS]  = { 7, 5, 3 };
    static const int nb_keys[OV_NB_LIST_PANELS]  = { 8, 11, 7 };

    int             list = (lay->focus < OV_NB_LIST_PANELS);
    OV_PANEL_STATE *ps   = list ? &lay->panel[lay->focus] : NULL;

    if (key == 'S' || key == 'A' || key == '>' || key == ']' || key == '<' || key == '[' ||
        (key == 's' &&
         !(lay->ctrl_mode && (lay->focus == OV_FOCUS_FPS || lay->focus == OV_FOCUS_PROCS))))
    {
        if (ps != NULL)
        {
            int n = nb_keys[lay->focus];
            switch (key)
            {
            case 'S':
                ps->sort_key = key_rate[lay->focus];
                break;
            case 'A':
                ps->sort_key = key_anc[lay->focus];
                ps->sort_dir = 0;
                break;
            case 's':
                ps->sort_key = 0;
                break;
            case '[':
                ps->sort_dir = !ps->sort_dir;
                break;
            case '<':
                ps->sort_key =
                    (ov_table_for(lay->focus) != NULL)
                        ? ov_table_next_sort(ov_table_for(lay->focus), lay, ps->sort_key, -1)
                        : (ps->sort_key + n - 1) % n;
                break;
            default: /* > ] skip hidden and unsortable columns */
                ps->sort_key =
                    (ov_table_for(lay->focus) != NULL)
                        ? ov_table_next_sort(ov_table_for(lay->focus), lay, ps->sort_key, +1)
                        : (ps->sort_key + 1) % n;
                break;
            }
        }
        lay->sort_pending = 1;
        return 1;
    }
    return 0;
}
