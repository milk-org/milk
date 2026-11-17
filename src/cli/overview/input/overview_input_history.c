// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_history.c
 * @brief Directory and FPS parameter navigation history serialization.
 */

#include "overview_input_internal.h"
#include <stdlib.h>

/* F5 parameter-tree positions: the last directory per FPS, and the cursor per FPS directory */
struct OV_INPUT_HISTORY
{
    struct
    {
        char fps_name[80];
        char path[200];
    } fps_last_path[200];
    int nb_fps_last_path;

    struct
    {
        char fps_name[80];
        char path[200];
        int  sel;
        int  scroll;
    } fps_dir_history[1000];
    int nb_fps_dir_history;
};

/** ov_hist - the layout's history, allocated on first use (344 KB: kept off the stack), or NULL. */
static OV_INPUT_HISTORY *ov_hist(OV_LAYOUT *lay)
{
    if (lay->hist == NULL)
    {
        lay->hist = calloc(1, sizeof(*lay->hist));
    }
    return lay->hist;
}

/** ov_input_history_free - release the layout's F5 tree history. */
void ov_input_history_free(OV_LAYOUT *lay)
{
    free(lay->hist);
    lay->hist = NULL;
}

/**
 * ov_input_save_dir_history - save cursor position and scroll for current FPS parameter directory.
 * @lay:      Pointer to overview layout structure.
 * @fps_name: Active FPS instance name.
 * @path:     Directory path within the parameter tree.
 */
void ov_input_save_dir_history(OV_LAYOUT *lay, const char *fps_name, const char *path)
{
    OV_INPUT_HISTORY *h = ov_hist(lay);
    if (h == NULL)
    {
        return;
    }
    if (fps_name == NULL || fps_name[0] == '\0')
    {
        return;
    }

    int found_idx = -1;
    for (int i = 0; i < h->nb_fps_dir_history; i++)
    {
        if (strcmp(h->fps_dir_history[i].fps_name, fps_name) == 0 &&
            strcmp(h->fps_dir_history[i].path, path) == 0)
        {
            found_idx = i;
            break;
        }
    }

    if (found_idx == -1)
    {
        if (h->nb_fps_dir_history < 1000)
        {
            found_idx = h->nb_fps_dir_history;
            h->nb_fps_dir_history++;
            strncpy(h->fps_dir_history[found_idx].fps_name, fps_name,
                    sizeof(h->fps_dir_history[found_idx].fps_name) - 1);
            h->fps_dir_history[found_idx]
                .fps_name[sizeof(h->fps_dir_history[found_idx].fps_name) - 1] = '\0';
            strncpy(h->fps_dir_history[found_idx].path, path,
                    sizeof(h->fps_dir_history[found_idx].path) - 1);
            h->fps_dir_history[found_idx].path[sizeof(h->fps_dir_history[found_idx].path) - 1] =
                '\0';
        }
        else
        {
            return;
        }
    }

    h->fps_dir_history[found_idx].sel    = lay->fps_param_sel;
    h->fps_dir_history[found_idx].scroll = lay->fps_param_scroll;
}

/**
 * ov_input_load_dir_history - restore cursor position and scroll for given FPS parameter directory.
 * @lay:      Pointer to overview layout structure.
 * @fps_name: Active FPS instance name.
 * @path:     Directory path within the parameter tree.
 */
void ov_input_load_dir_history(OV_LAYOUT *lay, const char *fps_name, const char *path)
{
    OV_INPUT_HISTORY *h = ov_hist(lay);
    if (h == NULL)
    {
        return;
    }
    if (fps_name == NULL || fps_name[0] == '\0')
    {
        return;
    }

    int found_idx = -1;
    for (int i = 0; i < h->nb_fps_dir_history; i++)
    {
        if (strcmp(h->fps_dir_history[i].fps_name, fps_name) == 0 &&
            strcmp(h->fps_dir_history[i].path, path) == 0)
        {
            found_idx = i;
            break;
        }
    }

    if (found_idx != -1)
    {
        lay->fps_param_sel    = h->fps_dir_history[found_idx].sel;
        lay->fps_param_scroll = h->fps_dir_history[found_idx].scroll;
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
    OV_INPUT_HISTORY *h = ov_hist(lay);
    if (h == NULL)
    {
        return;
    }
    if (fps_name == NULL || fps_name[0] == '\0')
    {
        return;
    }

    /* 1. Save the directory history for the current path */
    ov_input_save_dir_history(lay, fps_name, lay->fps_param_path);

    /* 2. Save the last path visited for this FPS */
    int found_idx = -1;
    for (int i = 0; i < h->nb_fps_last_path; i++)
    {
        if (strcmp(h->fps_last_path[i].fps_name, fps_name) == 0)
        {
            found_idx = i;
            break;
        }
    }

    if (found_idx == -1)
    {
        if (h->nb_fps_last_path < 200)
        {
            found_idx = h->nb_fps_last_path;
            h->nb_fps_last_path++;
            strncpy(h->fps_last_path[found_idx].fps_name, fps_name,
                    sizeof(h->fps_last_path[found_idx].fps_name) - 1);
            h->fps_last_path[found_idx].fps_name[sizeof(h->fps_last_path[found_idx].fps_name) - 1] =
                '\0';
        }
        else
        {
            return;
        }
    }

    strncpy(h->fps_last_path[found_idx].path, lay->fps_param_path,
            sizeof(h->fps_last_path[found_idx].path) - 1);
    h->fps_last_path[found_idx].path[sizeof(h->fps_last_path[found_idx].path) - 1] = '\0';
}

/**
 * ov_input_load_fps_history - restore last visited path and directory state for an FPS module.
 * @lay:      Pointer to overview layout structure.
 * @fps_name: Active FPS instance name.
 */
void ov_input_load_fps_history(OV_LAYOUT *lay, const char *fps_name)
{
    OV_INPUT_HISTORY *h = ov_hist(lay);
    if (h == NULL)
    {
        return;
    }
    if (fps_name == NULL || fps_name[0] == '\0')
    {
        return;
    }

    /* 1. Load the last path visited for this FPS */
    int found_idx = -1;
    for (int i = 0; i < h->nb_fps_last_path; i++)
    {
        if (strcmp(h->fps_last_path[i].fps_name, fps_name) == 0)
        {
            found_idx = i;
            break;
        }
    }

    if (found_idx != -1)
    {
        strncpy(lay->fps_param_path, h->fps_last_path[found_idx].path,
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
