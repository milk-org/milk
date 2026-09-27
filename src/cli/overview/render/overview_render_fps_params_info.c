// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_render_fps_params_info.c
 * @brief FPS parameter type badges, hierarchical tree helper, and top-row parameter inspector.
 */

#include <string.h>
#include <strings.h>
#include <stdint.h>

#include "overview_render_internal.h"
#include "overview_render_fps_params.h"
#include "fps_types.h"

/**
 * fps_param_type_badge - short type label for badge column.
 * @type: FPS parameter type code.
 *
 * Return: Pointer to a static string like "INT ", "FLT ", etc.
 */
const char *fps_param_type_badge(uint32_t type)
{
    if (type == FPTYPE_INT64 || type == FPTYPE_INT32)
    {
        return "INT ";
    }
    if (type == FPTYPE_UINT64 || type == FPTYPE_UINT32)
    {
        return "UINT";
    }
    if (type == FPTYPE_FLOAT64 || type == FPTYPE_FLOAT32)
    {
        return "FLT ";
    }
    if (type == FPTYPE_ONOFF)
    {
        return "ON/F";
    }
    if (type == FPTYPE_STREAMNAME)
    {
        return "STRM";
    }
    if (FPTYPE_IS_STRING(type))
    {
        return "STR ";
    }
    if (type == FPTYPE_PID)
    {
        return "PID ";
    }
    if (type == FPTYPE_TIMESPEC)
    {
        return "TIME";
    }
    return "??? ";
}

/**
 * fps_param_type_color - color for a type badge.
 * @type: FPS parameter type code.
 *
 * Return: ov_rgb_t color value.
 */
ov_rgb_t fps_param_type_color(uint32_t type)
{
    if (type == FPTYPE_INT64 || type == FPTYPE_INT32 || type == FPTYPE_UINT64 ||
        type == FPTYPE_UINT32)
    {
        return (ov_rgb_t) { 80, 140, 220 }; /* blue-ish */
    }
    if (type == FPTYPE_FLOAT64 || type == FPTYPE_FLOAT32)
    {
        return (ov_rgb_t) { 80, 200, 140 }; /* teal */
    }
    if (type == FPTYPE_ONOFF)
    {
        return (ov_rgb_t) { 200, 150, 60 }; /* amber */
    }
    if (type == FPTYPE_STREAMNAME)
    {
        return (ov_rgb_t) { 160, 100, 220 }; /* purple */
    }
    return (ov_rgb_t) { 130, 130, 130 }; /* dim grey */
}

/**
 * ov_get_fps_tree_items - enumerate hierarchical parameter directory nodes for an FPS module.
 * @fps:       Pointer to FPS module descriptor.
 * @path:      Directory sub-path within parameter tree.
 * @items:     Output array to receive tree items.
 * @max_items: Maximum capacity of items array.
 *
 * Return: Number of items populated into array.
 */
int ov_get_fps_tree_items(const OV_FPS    *fps,
                          const char      *path,
                          fps_tree_item_t *items,
                          int              max_items)
{
    if (fps == NULL)
    {
        return 0;
    }

    const OV_FPS_PARAMS *params = ov_fps_get_params(fps->name);
    if (params == NULL)
    {
        return 0;
    }

    int count    = 0;
    int path_len = (int) strlen(path);

    for (int i = 0; i < params->nb_disp_params; i++)
    {
        const char *pname = params->disp_param_name[i];
        if (pname[0] == '.')
        {
            pname++;
        }

        if (path_len > 0)
        {
            if (strncmp(pname, path, path_len) != 0)
            {
                continue;
            }
            if (pname[path_len] != '.')
            {
                continue;
            }
            pname += path_len + 1;
        }

        const char *next_dot = strchr(pname, '.');
        char        seg[80]  = { 0 };
        int         is_dir   = 0;

        if (next_dot)
        {
            int seg_len = (int) (next_dot - pname);
            if (seg_len >= (int) sizeof(seg))
            {
                seg_len = (int) sizeof(seg) - 1;
            }
            strncpy(seg, pname, seg_len);
            is_dir = 1;
        }
        else
        {
            strncpy(seg, pname, sizeof(seg) - 1);
            is_dir = 0;
        }

        int dup = 0;
        if (is_dir)
        {
            for (int k = 0; k < count; k++)
            {
                if (items[k].is_dir && strcmp(items[k].name, seg) == 0)
                {
                    dup = 1;
                    break;
                }
            }
        }

        if (!dup && count < max_items)
        {
            strncpy(items[count].name, seg, sizeof(items[count].name) - 1);
            items[count].is_dir    = is_dir;
            items[count].param_idx = i;
            count++;
        }
    }

    for (int i = 0; i < count - 1; i++)
    {
        for (int j = i + 1; j < count; j++)
        {
            int swap = 0;
            if (items[i].is_dir != items[j].is_dir)
            {
                if (items[j].is_dir)
                {
                    swap = 1;
                }
            }
            else
            {
                if (strcasecmp(items[i].name, items[j].name) > 0)
                {
                    swap = 1;
                }
            }
            if (swap)
            {
                fps_tree_item_t tmp = items[i];
                items[i]            = items[j];
                items[j]            = tmp;
            }
        }
    }

    return count;
}

/**
 * ov_render_fps_param_info - draw FPS parameter metadata header on rows 3 and 4.
 * @lay: Layout state.
 * @m:   Data model snapshot.
 */
void ov_render_fps_param_info(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (lay->fps_param_focus == 0)
    {
        /* Clear row 4 to prevent stale parameter info */
        ov_buf_pos(4, 1);
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_hline(' ', lay->term_cols);
        return;
    }

    /* Clear rows 3 and 4 */
    ov_buf_pos(3, 1);
    ov_theme_bg(OV_BG_PANEL);
    ov_buf_hline(' ', lay->term_cols);

    ov_buf_pos(4, 1);
    ov_theme_bg(OV_BG_PANEL);
    ov_buf_hline(' ', lay->term_cols);

    int fsel = lay->sel_fps;
    if (fsel < 0 || fsel >= m->nb_fps)
    {
        return;
    }

    const OV_FPS *fps = &m->fps[fsel];

    fps_tree_item_t items[1024];
    int             nitems = ov_get_fps_tree_items(fps, lay->fps_param_path, items, 1024);

    if (lay->fps_param_sel < 0 || lay->fps_param_sel >= nitems)
    {
        return;
    }

    const fps_tree_item_t *item = &items[lay->fps_param_sel];
    if (item->is_dir)
    {
        /* Draw directory info */
        ov_buf_pos(3, 2);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_bold();
        ov_buf_printf("DIR: ");
        ov_theme_fg(OV_FG_TEXT);
        ov_buf_printf("%s%s%s", lay->fps_param_path, lay->fps_param_path[0] ? "." : "", item->name);

        ov_buf_pos(4, 2);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("Description: (Directory)");
        ov_buf_reset_attr();
        return;
    }

    int pi = item->param_idx;

    const OV_FPS_PARAMS *params = ov_fps_get_params(fps->name);
    if (params == NULL || pi < 0 || pi >= params->nb_disp_params)
    {
        return;
    }

    /* Format full parameter path/name */
    char full_name[256];
    if (lay->fps_param_path[0] != '\0')
    {
        snprintf(full_name, sizeof(full_name), "%s.%s", lay->fps_param_path, item->name);
    }
    else
    {
        snprintf(full_name, sizeof(full_name), "%s", item->name);
    }

    /* Type badge */
    const char *type_name = "UNKNOWN";
    uint32_t    type      = params->disp_param_type[pi];
    if (type == FPTYPE_INT64 || type == FPTYPE_INT32)
    {
        type_name = "INT";
    }
    else if (type == FPTYPE_UINT64 || type == FPTYPE_UINT32)
    {
        type_name = "UINT";
    }
    else if (type == FPTYPE_FLOAT64 || type == FPTYPE_FLOAT32)
    {
        type_name = "FLOAT";
    }
    else if (type == FPTYPE_ONOFF)
    {
        type_name = "ON/OFF";
    }
    else if (type == FPTYPE_STREAMNAME)
    {
        type_name = "STREAM";
    }
    else if (FPTYPE_IS_STRING(type))
    {
        type_name = "STRING";
    }
    else if (type == FPTYPE_PID)
    {
        type_name = "PID";
    }
    else if (type == FPTYPE_TIMESPEC)
    {
        type_name = "TIMESPEC";
    }

    /* Display values and limits on Row 3 */
    ov_buf_pos(3, 2);
    ov_theme_fg(OV_FG_FPS);
    ov_buf_bold();
    ov_buf_printf("PARAM: ");
    ov_theme_fg(OV_FG_TEXT);
    ov_buf_printf("%s ", full_name);

    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("[%s]  ", type_name);

    ov_theme_fg(OV_FG_FPS);
    ov_buf_printf("Value: ");
    ov_theme_fg(OV_FG_TEXT);
    ov_buf_printf("%s  ", params->disp_param_value[pi]);

    if (params->disp_param_has_min[pi])
    {
        ov_theme_fg(OV_FG_FPS);
        ov_buf_printf("Min: ");
        ov_theme_fg(OV_FG_TEXT);
        ov_buf_printf("%s  ", params->disp_param_min[pi]);
    }
    if (params->disp_param_has_max[pi])
    {
        ov_theme_fg(OV_FG_FPS);
        ov_buf_printf("Max: ");
        ov_theme_fg(OV_FG_TEXT);
        ov_buf_printf("%s  ", params->disp_param_max[pi]);
    }

    /* Display description on Row 4 */
    ov_buf_pos(4, 2);
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("Description: ");
    ov_theme_fg(OV_FG_TEXT);
    ov_buf_printf("%s", params->disp_param_descr[pi][0] ? params->disp_param_descr[pi] : "(none)");

    if (type == FPTYPE_ONOFF)
    {
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("  (Press 'o' to toggle. Control mode must be ON [press 'c'])");
    }

    ov_buf_reset_attr();
}
