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
        if (lay->focus == OV_FOCUS_STREAMS)
        {
            int num_cols = ov_get_num_cols(lay, OV_FOCUS_STREAMS);
            lay->panel[OV_FOCUS_STREAMS].highlight_col =
                (lay->panel[OV_FOCUS_STREAMS].highlight_col + dir + num_cols) % num_cols;
        }
        else if (lay->focus == OV_FOCUS_PROCS)
        {
            int num_cols = ov_get_num_cols(lay, OV_FOCUS_PROCS);
            lay->panel[OV_FOCUS_PROCS].highlight_col =
                (lay->panel[OV_FOCUS_PROCS].highlight_col + dir + num_cols) % num_cols;
        }
        else if (lay->focus == OV_FOCUS_FPS)
        {
            int num_cols = ov_get_num_cols(lay, OV_FOCUS_FPS);
            lay->panel[OV_FOCUS_FPS].highlight_col =
                (lay->panel[OV_FOCUS_FPS].highlight_col + dir + num_cols) % num_cols;
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
        else if (lay->focus == OV_FOCUS_PROCS)
        {
            int logical_col = ov_get_logical_col_proc(lay->panel[OV_FOCUS_PROCS].highlight_col,
                                                      lay->compact_mode);
            lay->panel[OV_FOCUS_PROCS].collapsed ^= (1U << logical_col);
        }
        else if (lay->focus == OV_FOCUS_FPS)
        {
            int logical_col =
                ov_get_logical_col_fps(lay->panel[OV_FOCUS_FPS].highlight_col, lay->compact_mode);
            lay->panel[OV_FOCUS_FPS].collapsed ^= (1U << logical_col);
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
    if (key == 'S')
    {
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS:
            lay->panel[OV_FOCUS_STREAMS].sort_key = 3;
            break;
        case OV_FOCUS_PROCS:
            lay->panel[OV_FOCUS_PROCS].sort_key = 3;
            break;
        case OV_FOCUS_FPS:
            lay->panel[OV_FOCUS_FPS].sort_key = 1;
            break;
        default:
            break;
        }
        lay->sort_pending = 1;
        return 1;
    }

    if (key == 'A')
    {
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS:
            lay->panel[OV_FOCUS_STREAMS].sort_key = 7;
            lay->panel[OV_FOCUS_STREAMS].sort_dir = 0;
            break;
        case OV_FOCUS_PROCS:
            lay->panel[OV_FOCUS_PROCS].sort_key = 5;
            lay->panel[OV_FOCUS_PROCS].sort_dir = 0;
            break;
        case OV_FOCUS_FPS:
            lay->panel[OV_FOCUS_FPS].sort_key = 3;
            lay->panel[OV_FOCUS_FPS].sort_dir = 0;
            break;
        default:
            break;
        }
        lay->sort_pending = 1;
        return 1;
    }

    if (key == 's' &&
        !(lay->ctrl_mode && (lay->focus == OV_FOCUS_FPS || lay->focus == OV_FOCUS_PROCS)))
    {
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS:
            lay->panel[OV_FOCUS_STREAMS].sort_key = 0;
            break;
        case OV_FOCUS_PROCS:
            lay->panel[OV_FOCUS_PROCS].sort_key = 0;
            break;
        case OV_FOCUS_FPS:
            lay->panel[OV_FOCUS_FPS].sort_key = 0;
            break;
        default:
            break;
        }
        lay->sort_pending = 1;
        return 1;
    }

    if (key == '>' || key == ']')
    {
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS:
            lay->panel[OV_FOCUS_STREAMS].sort_key = (lay->panel[OV_FOCUS_STREAMS].sort_key + 1) % 8;
            break;
        case OV_FOCUS_PROCS:
            lay->panel[OV_FOCUS_PROCS].sort_key = (lay->panel[OV_FOCUS_PROCS].sort_key + 1) % 11;
            break;
        case OV_FOCUS_FPS:
            lay->panel[OV_FOCUS_FPS].sort_key = (lay->panel[OV_FOCUS_FPS].sort_key + 1) % 7;
            break;
        default:
            break;
        }
        lay->sort_pending = 1;
        return 1;
    }

    if (key == '<')
    {
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS:
            lay->panel[OV_FOCUS_STREAMS].sort_key = (lay->panel[OV_FOCUS_STREAMS].sort_key + 7) % 8;
            break;
        case OV_FOCUS_PROCS:
            lay->panel[OV_FOCUS_PROCS].sort_key = (lay->panel[OV_FOCUS_PROCS].sort_key + 10) % 11;
            break;
        case OV_FOCUS_FPS:
            lay->panel[OV_FOCUS_FPS].sort_key = (lay->panel[OV_FOCUS_FPS].sort_key + 6) % 7;
            break;
        default:
            break;
        }
        lay->sort_pending = 1;
        return 1;
    }

    if (key == '[')
    {
        switch (lay->focus)
        {
        case OV_FOCUS_STREAMS:
            lay->panel[OV_FOCUS_STREAMS].sort_dir = !lay->panel[OV_FOCUS_STREAMS].sort_dir;
            break;
        case OV_FOCUS_PROCS:
            lay->panel[OV_FOCUS_PROCS].sort_dir = !lay->panel[OV_FOCUS_PROCS].sort_dir;
            break;
        case OV_FOCUS_FPS:
            lay->panel[OV_FOCUS_FPS].sort_dir = !lay->panel[OV_FOCUS_FPS].sort_dir;
            break;
        default:
            break;
        }
        lay->sort_pending = 1;
        return 1;
    }

    return 0;
}
