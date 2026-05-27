/**
 * @file    overview_render_fps_params.c
 * @brief   FPS parameter tree panel for milk-CTRL F5 view
 *
 * Renders the right-side parameter panel when the F5
 * (OV_VIEW_FPS) tab is active. Shows a scrollable flat
 * list of all visible parameters for the selected FPS,
 * with type badge, current value, and writability icon.
 */

#include <string.h>
#include <stdint.h>

#include "overview_render_internal.h"
#include "fps_types.h"

/* =========================================================
 * Helpers
 * ========================================================= */

/**
 * fps_param_type_badge - short type label for badge column.
 * @type: FPS parameter type code
 *
 * Return: pointer to a static string like "INT", "FLT", etc.
 */
static const char *fps_param_type_badge(uint32_t type)
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
 * @type: FPS parameter type code
 *
 * Return: ov_rgb_t color value.
 */
static ov_rgb_t fps_param_type_color(uint32_t type)
{
    if (type == FPTYPE_INT64 || type == FPTYPE_INT32 || type == FPTYPE_UINT64 ||
        type == FPTYPE_UINT32)
    {
        return (ov_rgb_t) { 80, 140, 220 }; /* blue-ish  */
    }
    if (type == FPTYPE_FLOAT64 || type == FPTYPE_FLOAT32)
    {
        return (ov_rgb_t) { 80, 200, 140 }; /* teal      */
    }
    if (type == FPTYPE_ONOFF)
    {
        return (ov_rgb_t) { 200, 150, 60 }; /* amber     */
    }
    if (type == FPTYPE_STREAMNAME)
    {
        return (ov_rgb_t) { 160, 100, 220 }; /* purple    */
    }
    return (ov_rgb_t) { 130, 130, 130 }; /* dim grey  */
}

/**
 * render_param_breadcrumb - draw the FPS name / param count
 * header at the top of r_fps_params.
 *
 * @lay: layout state
 * @fps: FPS data entry
 * @r:   panel rectangle (r_fps_params)
 */
static void render_param_breadcrumb(const OV_LAYOUT *lay, const OV_FPS *fps, OV_RECT r)
{
    int      focused   = (lay->fps_param_focus == 1);
    ov_rgb_t border_fg = focused ? OV_FG_FPS : OV_FG_DIM;

    /* Draw border and title in one call */
    char title[256];
    if (lay->fps_param_path[0] != '\0')
    {
        snprintf(title, sizeof(title), "PARAMS  %s / %s", fps->name, lay->fps_param_path);
    }
    else
    {
        snprintf(title, sizeof(title), "PARAMS  %s", fps->name);
    }
    ov_draw_panel_border(r.row, r.col, r.height, r.width, title, border_fg, focused, 0);

    /* Header row: col labels */
    int hrow = r.row + 1;
    ov_buf_pos(hrow, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_DIM);

    int kw = r.width - 18; /* keyword column width */
    if (kw < 6)
    {
        kw = 6;
    }
    ov_buf_printf(" %-*s %4s  %-*s", kw, "PARAMETER", "TYPE", r.width - kw - 12, "VALUE");
    ov_buf_reset_attr();
}

/* =========================================================
 * Tree Helpers
 * ========================================================= */

int ov_get_fps_tree_items(const OV_FPS    *fps,
                          const char      *path,
                          fps_tree_item_t *items,
                          int              max_items)
{
    int count    = 0;
    int path_len = strlen(path);

    for (int i = 0; i < fps->nb_disp_params; i++)
    {
        const char *pname = fps->disp_param_name[i];
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
            int seg_len = next_dot - pname;
            if (seg_len >= (int) sizeof(seg))
            {
                seg_len = sizeof(seg) - 1;
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

/* =========================================================
 * Public renderer
 * ========================================================= */

/**
 * ov_render_fps_params_panel - draw parameter tree panel.
 * @lay: layout state (fps_param_focus/sel/scroll used)
 * @m:   data model
 *
 * Draws the right panel in the F5 split view. Shows all
 * visible parameters for the currently selected FPS entry
 * as a virtual hierarchical tree.
 */
void ov_render_fps_params_panel(OV_LAYOUT *lay, const OV_MODEL *m)
{
    OV_RECT r = lay->r_fps_params;

    /* --- Guard: need a valid FPS selection with params --- */
    int fsel = lay->sel_fps;
    if (fsel < 0 || fsel >= m->nb_fps)
    {
        /* Draw empty border */
        ov_draw_panel_border(r.row, r.col, r.height, r.width, "PARAMS", OV_FG_DIM, 0, 0);
        return;
    }

    const OV_FPS *fps = &m->fps[fsel];

    fps_tree_item_t items[1024];
    int             nitems = ov_get_fps_tree_items(fps, lay->fps_param_path, items, 1024);

    if (lay->fps_param_focus == 1 && nitems > 0)
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

    render_param_breadcrumb(lay, fps, r);

    if (nitems <= 0)
    {
        if (lay->fps_param_path[0] != '\0')
        {
            ov_buf_pos(r.row + 2, r.col + 2);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("(Empty directory)");
        }
        else
        {
            ov_buf_pos(r.row + 2, r.col + 2);
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("(No params)");
        }
        return;
    }

    /* Clamp scroll */
    int max_rows = r.height - 3; /* header row + borders */
    if (max_rows < 1)
    {
        max_rows = 1;
    }

    int scroll = lay->fps_param_scroll;
    int psel   = lay->fps_param_sel;

    /* Keep cursor visible */
    if (psel < scroll)
    {
        scroll = psel;
    }
    if (psel >= scroll + max_rows)
    {
        scroll = psel - max_rows + 1;
    }
    if (scroll < 0)
    {
        scroll = 0;
    }
    if (scroll > nitems - max_rows)
    {
        scroll = nitems - max_rows;
    }
    if (scroll < 0)
    {
        scroll = 0;
    }
    lay->fps_param_scroll = scroll;

    /* Keyword column width */
    int kw = r.width / 2 - 2;
    if (kw > 35)
    {
        kw = 35;
    }
    if (kw < 12)
    {
        kw = 12;
    }

    for (int i = 0; i < max_rows; i++)
    {
        int row      = r.row + 2 + i;
        int list_idx = scroll + i;

        if (list_idx >= nitems)
        {
            clear_row(row, r.col + 1, r.width - 2, OV_BG_PANEL);
            continue;
        }

        int is_sel = (lay->fps_param_focus == 1 && list_idx == psel);

        ov_rgb_t row_bg = is_sel ? OV_BG_SELECTED : OV_BG_PANEL;

        ov_buf_pos(row, r.col + 1);
        ov_theme_bg(row_bg);

        fps_tree_item_t *item = &items[list_idx];

        if (item->is_dir)
        {
            /* Directory row */
            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("  "); /* no pencil */

            ov_theme_fg(is_sel ? OV_FG_TEXT : (ov_rgb_t) { 220, 200, 100 }); /* folder color */
            ov_buf_printf("%-*.*s/", kw - 1, kw - 1, item->name);

            ov_theme_fg(OV_FG_DIM);
            ov_buf_printf("%4s  ", "DIR ");

            int vw = r.width - kw - 12;
            if (vw < 4)
            {
                vw = 4;
            }
            ov_buf_printf("%-*s", vw, ""); /* no value */
        }
        else
        {
            /* Leaf parameter row */
            int pi = item->param_idx;

            /* Write-status badge: W (writable now) / C (conf only) / NW (locked)
             * Mirrors the W /NW display in milk-fpsCTRL. Column is 3 chars wide. */
            {
                uint64_t fl = fps->disp_param_flags[pi];
                int      ws = (fl & FPFLAG_WRITESTATUS) != 0;
                int      wc = (fl & FPFLAG_WRITECONF) != 0;

                if (ws)
                {
                    /* Currently writable — green "W " */
                    ov_theme_fg(is_sel ? OV_FG_ACTIVE : (ov_rgb_t) { 80, 200, 80 });
                    ov_buf_printf("W ");
                }
                else if (wc)
                {
                    /* Writable in conf mode only — amber "C " */
                    ov_theme_fg(is_sel ? OV_FG_TEXT : (ov_rgb_t) { 220, 180, 60 });
                    ov_buf_printf("C ");
                }
                else
                {
                    /* Not writable — dim red "NW" */
                    ov_theme_fg(is_sel ? OV_FG_DIM : (ov_rgb_t) { 160, 60, 60 });
                    ov_buf_printf("NW");
                }
            }
            ov_buf_printf(" "); /* separator after badge */

            /* Parameter keyword */
            ov_theme_fg(is_sel ? OV_FG_TEXT : OV_FG_FPS);
            ov_buf_printf("%-*.*s ", kw, kw, item->name);

            /* Type badge */
            {
                ov_rgb_t tc = fps_param_type_color(fps->disp_param_type[pi]);
                if (is_sel)
                {
                    tc = OV_FG_TEXT;
                }
                ov_theme_fg(tc);
                ov_buf_printf("%4s  ", fps_param_type_badge(fps->disp_param_type[pi]));
            }

            /* Value */
            {
                const char *val = fps->disp_param_value[pi];
                /* budget: badge(3) + sep(1) + kw + 1 + typebadge(6) = kw + 11 */
                int vw = r.width - kw - 14;
                if (vw < 4)
                {
                    vw = 4;
                }

                /* ONOFF: color based on string value "ON" / "OFF" */
                if (fps->disp_param_type[pi] == FPTYPE_ONOFF)
                {
                    int is_on = (val[0] == 'O' && val[1] == 'N');
                    if (is_on)
                    {
                        ov_theme_fg(is_sel ? OV_FG_ACTIVE : (ov_rgb_t) { 60, 220, 60 });
                        ov_buf_printf("%-*s", vw, "ON");
                    }
                    else
                    {
                        ov_theme_fg(is_sel ? OV_FG_DIM : (ov_rgb_t) { 160, 60, 60 });
                        ov_buf_printf("%-*s", vw, "OFF");
                    }
                }
                else
                {
                    ov_theme_fg(is_sel ? OV_FG_TEXT : OV_FG_DIM);
                    ov_buf_printf("%-*.*s", vw, vw, val);
                }
            }
        }

        ov_buf_reset_attr();
    } /* for each row */

    render_scroll_indicators(r, scroll, max_rows, nitems, OV_FG_FPS);

    /* Footer: item count */
    {
        char fbuf[48];
        snprintf(fbuf, sizeof(fbuf), " %d/%d items ", psel + 1, nitems);
        int flen = (int) strlen(fbuf);
        int bcol = r.col + r.width - flen - 2;
        if (bcol > r.col + 1)
        {
            ov_buf_pos(r.row + r.height - 1, bcol);
            ov_theme_fg(OV_FG_DIM);
            ov_theme_bg(OV_BG_PANEL);
            ov_buf_printf("%s", fbuf);
        }
    }

    ov_buf_reset_attr();
}

/**
 * ov_render_fps_param_info - draw FPS parameter metadata header on rows 2 and 3.
 * @lay: layout state
 * @m:   data model
 */
void ov_render_fps_param_info(const OV_LAYOUT *lay, const OV_MODEL *m)
{
    if (lay->fps_param_focus == 0)
    {
        /* Clear row 3 to prevent stale parameter info */
        ov_buf_pos(3, 1);
        ov_theme_bg(OV_BG_PANEL);
        ov_buf_hline(' ', lay->term_cols);
        return;
    }

    /* Clear rows 2 and 3 */
    ov_buf_pos(2, 1);
    ov_theme_bg(OV_BG_PANEL);
    ov_buf_hline(' ', lay->term_cols);

    ov_buf_pos(3, 1);
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
        ov_buf_pos(2, 2);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_bold();
        ov_buf_printf("DIR: ");
        ov_theme_fg(OV_FG_TEXT);
        ov_buf_printf("%s%s%s", lay->fps_param_path, lay->fps_param_path[0] ? "." : "", item->name);

        ov_buf_pos(3, 2);
        ov_theme_fg(OV_FG_DIM);
        ov_buf_printf("Description: (Directory)");
        ov_buf_reset_attr();
        return;
    }

    int pi = item->param_idx;

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
    uint32_t    type      = fps->disp_param_type[pi];
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

    /* Display values and limits on Row 2 */
    ov_buf_pos(2, 2);
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
    ov_buf_printf("%s  ", fps->disp_param_value[pi]);

    if (fps->disp_param_has_min[pi])
    {
        ov_theme_fg(OV_FG_FPS);
        ov_buf_printf("Min: ");
        ov_theme_fg(OV_FG_TEXT);
        ov_buf_printf("%s  ", fps->disp_param_min[pi]);
    }
    if (fps->disp_param_has_max[pi])
    {
        ov_theme_fg(OV_FG_FPS);
        ov_buf_printf("Max: ");
        ov_theme_fg(OV_FG_TEXT);
        ov_buf_printf("%s  ", fps->disp_param_max[pi]);
    }

    /* Display description on Row 3 */
    ov_buf_pos(3, 2);
    ov_theme_fg(OV_FG_DIM);
    ov_buf_printf("Description: ");
    ov_theme_fg(OV_FG_TEXT);
    ov_buf_printf("%s", fps->disp_param_descr[pi][0] ? fps->disp_param_descr[pi] : "(none)");

    if (type == FPTYPE_ONOFF)
    {
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("  (Press 'o' to toggle. Control mode must be ON [press 'c'])");
    }

    ov_buf_reset_attr();
}
