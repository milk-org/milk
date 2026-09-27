// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_history.c
 * @brief Directory and FPS parameter navigation history serialization.
 */

#include "overview_input_internal.h"

/**
 * ov_input_save_dir_history - save cursor position and scroll for current FPS parameter directory.
 * @lay:      Pointer to overview layout structure.
 * @fps_name: Active FPS instance name.
 * @path:     Directory path within the parameter tree.
 */
void ov_input_save_dir_history(OV_LAYOUT *lay, const char *fps_name, const char *path)
{
    if (fps_name == NULL || fps_name[0] == '\0')
    {
        return;
    }

    int found_idx = -1;
    for (int i = 0; i < lay->nb_fps_dir_history; i++)
    {
        if (strcmp(lay->fps_dir_history[i].fps_name, fps_name) == 0 &&
            strcmp(lay->fps_dir_history[i].path, path) == 0)
        {
            found_idx = i;
            break;
        }
    }

    if (found_idx == -1)
    {
        if (lay->nb_fps_dir_history < 1000)
        {
            found_idx = lay->nb_fps_dir_history;
            lay->nb_fps_dir_history++;
            strncpy(lay->fps_dir_history[found_idx].fps_name, fps_name,
                    sizeof(lay->fps_dir_history[found_idx].fps_name) - 1);
            lay->fps_dir_history[found_idx]
                .fps_name[sizeof(lay->fps_dir_history[found_idx].fps_name) - 1] = '\0';
            strncpy(lay->fps_dir_history[found_idx].path, path,
                    sizeof(lay->fps_dir_history[found_idx].path) - 1);
            lay->fps_dir_history[found_idx].path[sizeof(lay->fps_dir_history[found_idx].path) - 1] =
                '\0';
        }
        else
        {
            return;
        }
    }

    lay->fps_dir_history[found_idx].sel    = lay->fps_param_sel;
    lay->fps_dir_history[found_idx].scroll = lay->fps_param_scroll;
}

/**
 * ov_input_load_dir_history - restore cursor position and scroll for given FPS parameter directory.
 * @lay:      Pointer to overview layout structure.
 * @fps_name: Active FPS instance name.
 * @path:     Directory path within the parameter tree.
 */
void ov_input_load_dir_history(OV_LAYOUT *lay, const char *fps_name, const char *path)
{
    if (fps_name == NULL || fps_name[0] == '\0')
    {
        return;
    }

    int found_idx = -1;
    for (int i = 0; i < lay->nb_fps_dir_history; i++)
    {
        if (strcmp(lay->fps_dir_history[i].fps_name, fps_name) == 0 &&
            strcmp(lay->fps_dir_history[i].path, path) == 0)
        {
            found_idx = i;
            break;
        }
    }

    if (found_idx != -1)
    {
        lay->fps_param_sel    = lay->fps_dir_history[found_idx].sel;
        lay->fps_param_scroll = lay->fps_dir_history[found_idx].scroll;
    }
    else
    {
        lay->fps_param_sel    = 0;
        lay->fps_param_scroll = 0;
    }
}

/**
 * ov_input_save_fps_history - save last visited path and directory state for an FPS module.
 * @lay:      Pointer to overview layout structure.
 * @fps_name: Active FPS instance name.
 */
void ov_input_save_fps_history(OV_LAYOUT *lay, const char *fps_name)
{
    if (fps_name == NULL || fps_name[0] == '\0')
    {
        return;
    }

    /* 1. Save the directory history for the current path */
    ov_input_save_dir_history(lay, fps_name, lay->fps_param_path);

    /* 2. Save the last path visited for this FPS */
    int found_idx = -1;
    for (int i = 0; i < lay->nb_fps_last_path; i++)
    {
        if (strcmp(lay->fps_last_path[i].fps_name, fps_name) == 0)
        {
            found_idx = i;
            break;
        }
    }

    if (found_idx == -1)
    {
        if (lay->nb_fps_last_path < 200)
        {
            found_idx = lay->nb_fps_last_path;
            lay->nb_fps_last_path++;
            strncpy(lay->fps_last_path[found_idx].fps_name, fps_name,
                    sizeof(lay->fps_last_path[found_idx].fps_name) - 1);
            lay->fps_last_path[found_idx]
                .fps_name[sizeof(lay->fps_last_path[found_idx].fps_name) - 1] = '\0';
        }
        else
        {
            return;
        }
    }

    strncpy(lay->fps_last_path[found_idx].path, lay->fps_param_path,
            sizeof(lay->fps_last_path[found_idx].path) - 1);
    lay->fps_last_path[found_idx].path[sizeof(lay->fps_last_path[found_idx].path) - 1] = '\0';
}

/**
 * ov_input_load_fps_history - restore last visited path and directory state for an FPS module.
 * @lay:      Pointer to overview layout structure.
 * @fps_name: Active FPS instance name.
 */
void ov_input_load_fps_history(OV_LAYOUT *lay, const char *fps_name)
{
    if (fps_name == NULL || fps_name[0] == '\0')
    {
        return;
    }

    /* 1. Load the last path visited for this FPS */
    int found_idx = -1;
    for (int i = 0; i < lay->nb_fps_last_path; i++)
    {
        if (strcmp(lay->fps_last_path[i].fps_name, fps_name) == 0)
        {
            found_idx = i;
            break;
        }
    }

    if (found_idx != -1)
    {
        strncpy(lay->fps_param_path, lay->fps_last_path[found_idx].path,
                sizeof(lay->fps_param_path) - 1);
        lay->fps_param_path[sizeof(lay->fps_param_path) - 1] = '\0';
    }
    else
    {
        lay->fps_param_path[0] = '\0';
    }

    /* 2. Load the directory history for this path */
    ov_input_load_dir_history(lay, fps_name, lay->fps_param_path);
}
