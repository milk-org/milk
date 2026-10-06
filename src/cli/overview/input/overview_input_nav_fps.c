// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_input_nav_fps.c
 * @brief Navigation and interaction in the FPS parameter hierarchy (F5 view)
 */

#include "overview_input_internal.h"
#include <stdio.h>
#include <string.h>

/**
 * ov_input_nav_fps - handle navigation within FPS parameter tree (F5 view).
 * @key: Pressed key code.
 * @lay: Pointer to layout structure.
 * @m:   Pointer to data model snapshot.
 *
 * Return: 1 if key was consumed, 0 otherwise.
 */
int ov_input_nav_fps(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    int fsel       = lay->sel_fps;
    int has_params = (fsel >= 0 && fsel < m->nb_fps && m->fps[fsel].nb_disp_params > 0);

    int             nitems = 0;
    fps_tree_item_t items[1024];
    if (has_params)
    {
        nitems = ov_get_fps_tree_items(&m->fps[fsel], lay->fps_param_path, items, 1024);
    }

    /* RIGHT from list -> enter param panel */
    if (lay->fps_param_focus == 0 &&
        (key == OV_KEY_RIGHT || key == OV_KEY_ENTER || key == '\r' || key == '\n') && has_params)
    {
        lay->fps_param_focus = 1;
        if (nitems > 0)
        {
            if (lay->fps_param_sel < 0)
            {
                lay->fps_param_sel = 0;
            }
            else if (lay->fps_param_sel >= nitems)
            {
                lay->fps_param_sel = nitems - 1;
            }
        }
        return 1;
    }

    /* All nav/edit keys when param panel is focused */
    if (lay->fps_param_focus == 1)
    {
        if (nitems > 0)
        {
            if (lay->fps_param_sel < 0)
            {
                lay->fps_param_sel = 0;
            }
            else if (lay->fps_param_sel >= nitems)
            {
                lay->fps_param_sel = nitems - 1;
            }
        }
        /* ESC / LEFT from param panel -> back to list or ascend dir */
        if (key == OV_KEY_LEFT || key == OV_KEY_ESC)
        {
            if (lay->fps_param_path[0] == '\0')
            {
                lay->fps_param_focus = 0;
            }
            else
            {
                ov_input_save_dir_history(lay, m->fps[fsel].name, lay->fps_param_path);

                /* Ascend directory */
                char  exited_dir[100] = { 0 };
                char *last_dot        = strrchr(lay->fps_param_path, '.');
                if (last_dot)
                {
                    strncpy(exited_dir, last_dot + 1, sizeof(exited_dir) - 1);
                    *last_dot = '\0';
                }
                else
                {
                    strncpy(exited_dir, lay->fps_param_path, sizeof(exited_dir) - 1);
                    lay->fps_param_path[0] = '\0';
                }

                ov_input_load_dir_history(lay, m->fps[fsel].name, lay->fps_param_path);

                int found_sel = -1;
                if (has_params)
                {
                    fps_tree_item_t parent_items[1024];
                    int n_parent_items = ov_get_fps_tree_items(&m->fps[fsel], lay->fps_param_path,
                                                               parent_items, 1024);

                    for (int i = 0; i < n_parent_items; i++)
                    {
                        if (parent_items[i].is_dir && strcmp(parent_items[i].name, exited_dir) == 0)
                        {
                            found_sel = i;
                            break;
                        }
                    }
                }
                if (found_sel != -1)
                {
                    lay->fps_param_sel = found_sel;
                }
            }
            return 1;
        }

        int ph = lay->r_fps_params.height - 3;
        if (ph < 1)
        {
            ph = 1;
        }

        if (key == OV_KEY_UP)
        {
            if (lay->fps_param_sel > 0)
            {
                lay->fps_param_sel--;
            }
            return 1;
        }
        if (key == OV_KEY_DOWN)
        {
            if (lay->fps_param_sel < nitems - 1)
            {
                lay->fps_param_sel++;
            }
            return 1;
        }
        if (key == OV_KEY_PGUP)
        {
            lay->fps_param_sel -= ph;
            if (lay->fps_param_sel < 0)
            {
                lay->fps_param_sel = 0;
            }
            return 1;
        }
        if (key == OV_KEY_PGDN)
        {
            lay->fps_param_sel += ph;
            if (lay->fps_param_sel >= nitems)
            {
                lay->fps_param_sel = nitems - 1;
            }
            return 1;
        }
        if (key == OV_KEY_HOME)
        {
            lay->fps_param_sel = 0;
            return 1;
        }
        if (key == OV_KEY_END)
        {
            lay->fps_param_sel = nitems - 1;
            if (lay->fps_param_sel < 0)
            {
                lay->fps_param_sel = 0;
            }
            return 1;
        }
        if (key == OV_KEY_RIGHT || key == OV_KEY_ENTER || key == '\r' || key == '\n')
        {
            if (lay->fps_param_sel >= 0 && lay->fps_param_sel < nitems)
            {
                fps_tree_item_t *item = &items[lay->fps_param_sel];
                if (item->is_dir)
                {
                    ov_input_save_dir_history(lay, m->fps[fsel].name, lay->fps_param_path);

                    /* Descend directory */
                    if (lay->fps_param_path[0] == '\0')
                    {
                        strncpy(lay->fps_param_path, item->name, sizeof(lay->fps_param_path) - 1);
                    }
                    else
                    {
                        char tmp[200];
                        snprintf(tmp, sizeof(tmp), "%s.%s", lay->fps_param_path, item->name);
                        strncpy(lay->fps_param_path, tmp, sizeof(lay->fps_param_path) - 1);
                    }

                    ov_input_load_dir_history(lay, m->fps[fsel].name, lay->fps_param_path);
                }
                else if (key == OV_KEY_ENTER || key == '\r' || key == '\n')
                {
                    if (!lay->ctrl_mode)
                    {
                        ov_cmdlog_push(
                            &lay->cmdlog, OV_CMDLOG_WARN,
                            "Edit requires CONTROL mode (press c to toggle CTRL mode ON/OFF)");
                    }
                    else
                    {
                        ov_fps_inline_edit(lay, m->fps[fsel].name, item->param_idx);
                    }
                }
            }
            return 1;
        }
        if (key == 'o')
        {
            if (lay->fps_param_sel >= 0 && lay->fps_param_sel < nitems)
            {
                fps_tree_item_t *item = &items[lay->fps_param_sel];
                if (!item->is_dir)
                {
                    int                  pi     = item->param_idx;
                    const OV_FPS_PARAMS *params = ov_fps_get_params(m->fps[fsel].name);
                    if (params != NULL && pi >= 0 && pi < params->nb_disp_params &&
                        params->disp_param_type[pi] == FPTYPE_ONOFF)
                    {
                        if (!lay->ctrl_mode)
                        {
                            ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_WARN,
                                           "Toggle requires CONTROL mode (press c to toggle "
                                           "CTRL mode ON/OFF)");
                        }
                        else
                        {
                            char kw[FUNCTION_PARAMETER_STRMAXLEN] = { 0 };
                            int  newval                           = 0;
                            if (ov_fcache_toggle_param(m->fps[fsel].name, pi, kw, sizeof(kw),
                                                       &newval) == 0)
                            {
                                ov_cmdlog_push(&lay->cmdlog, OV_CMDLOG_INFO,
                                               "Toggled parameter %s to %s", kw,
                                               newval ? "ON" : "OFF");
                            }
                        }
                    }
                }
            }
            return 1;
        }
        return 0; /* pass through unmapped keys */
    }

    return 0;
}
