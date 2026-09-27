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
            int num_cols              = ov_get_num_cols(lay, OV_FOCUS_STREAMS);
            lay->highlight_col_stream = (lay->highlight_col_stream + dir + num_cols) % num_cols;
        }
        else if (lay->focus == OV_FOCUS_PROCS)
        {
            int num_cols            = ov_get_num_cols(lay, OV_FOCUS_PROCS);
            lay->highlight_col_proc = (lay->highlight_col_proc + dir + num_cols) % num_cols;
        }
        else if (lay->focus == OV_FOCUS_FPS)
        {
            int num_cols           = ov_get_num_cols(lay, OV_FOCUS_FPS);
            lay->highlight_col_fps = (lay->highlight_col_fps + dir + num_cols) % num_cols;
        }
        return 1;
    }

    if (key == 't' || key == 'T')
    {
        if (lay->focus == OV_FOCUS_STREAMS)
        {
            int logical_col =
                ov_get_logical_col_stream(lay->highlight_col_stream, lay->compact_mode);
            lay->col_collapsed_stream ^= (1U << logical_col);
        }
        else if (lay->focus == OV_FOCUS_PROCS)
        {
            int logical_col = ov_get_logical_col_proc(lay->highlight_col_proc, lay->compact_mode);
            lay->col_collapsed_proc ^= (1U << logical_col);
        }
        else if (lay->focus == OV_FOCUS_FPS)
        {
            int logical_col = ov_get_logical_col_fps(lay->highlight_col_fps, lay->compact_mode);
            lay->col_collapsed_fps ^= (1U << logical_col);
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
            lay->sort_key_stream = 3;
            break;
        case OV_FOCUS_PROCS:
            lay->sort_key_proc = 3;
            break;
        case OV_FOCUS_FPS:
            lay->sort_key_fps = 1;
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
            lay->sort_key_stream = 7;
            lay->sort_dir_stream = 0;
            break;
        case OV_FOCUS_PROCS:
            lay->sort_key_proc = 5;
            lay->sort_dir_proc = 0;
            break;
        case OV_FOCUS_FPS:
            lay->sort_key_fps = 3;
            lay->sort_dir_fps = 0;
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
            lay->sort_key_stream = 0;
            break;
        case OV_FOCUS_PROCS:
            lay->sort_key_proc = 0;
            break;
        case OV_FOCUS_FPS:
            lay->sort_key_fps = 0;
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
            lay->sort_key_stream = (lay->sort_key_stream + 1) % 8;
            break;
        case OV_FOCUS_PROCS:
            lay->sort_key_proc = (lay->sort_key_proc + 1) % 11;
            break;
        case OV_FOCUS_FPS:
            lay->sort_key_fps = (lay->sort_key_fps + 1) % 7;
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
            lay->sort_key_stream = (lay->sort_key_stream + 7) % 8;
            break;
        case OV_FOCUS_PROCS:
            lay->sort_key_proc = (lay->sort_key_proc + 10) % 11;
            break;
        case OV_FOCUS_FPS:
            lay->sort_key_fps = (lay->sort_key_fps + 6) % 7;
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
            lay->sort_dir_stream = !lay->sort_dir_stream;
            break;
        case OV_FOCUS_PROCS:
            lay->sort_dir_proc = !lay->sort_dir_proc;
            break;
        case OV_FOCUS_FPS:
            lay->sort_dir_fps = !lay->sort_dir_fps;
            break;
        default:
            break;
        }
        lay->sort_pending = 1;
        return 1;
    }

    return 0;
}
