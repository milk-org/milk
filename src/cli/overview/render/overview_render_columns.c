// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_columns.c
 * @brief   Draw list-panel headers and rows from their column tables; record header hit rectangles.
 *
 * Column c of a table is collapsed when bit c of the panel's collapsed mask is set ('t'), and the
 * column cursor (Shift+Left/Right) counts visible columns. One space separates a column from the next
 * unless it is collapsed, in the header and in the rows alike.
 */

#include "overview_render_internal.h"

/* Header cells as last drawn: screen column, width (separator included), table column */
static struct
{
    int col;
    int w;
    int c;
} s_hit[OV_NB_LIST_PANELS][OV_TABLE_MAX_COLS];
static int s_nhit[OV_NB_LIST_PANELS];

/**
 * ov_cell_fit - @s laid out in exactly @width columns (one per code point, \x01/\x02 markup kept),
 * truncated or padded on the side opposite to @right alignment, into @out.
 */
static void ov_cell_fit(const char *s, int width, int right, char *out, size_t n)
{
    char   body[256];
    size_t b    = 0;
    int    cols = 0, open = 0;
    for (size_t i = 0; s[i] != '\0' && b + 8 < sizeof(body);)
    {
        unsigned char ch = (unsigned char) s[i];
        if (ch == 0x01 || ch == 0x02)
        {
            open      = (ch == 0x01);
            body[b++] = s[i++];
            continue;
        }
        if (cols == width)
        {
            break;
        }
        int clen = utf8_char_length(ch);
        for (int k = 0; k < clen && s[i] != '\0'; k++)
        {
            body[b++] = s[i++];
        }
        cols++;
    }
    if (open)
    {
        body[b++] = 0x02;
    }
    body[b] = '\0';
    int pad = width - cols;
    snprintf(out, n, "%*s%s%*s", right ? pad : 0, "", body, right ? 0 : pad, "");
}

/** ov_table_sep - one separator space, counted against horizontal scroll and the panel width. */
static void ov_table_sep(int *hs_rem, int *printed, int avail)
{
    if (*hs_rem > 0)
    {
        (*hs_rem)--;
    }
    else if (*printed < avail)
    {
        ov_buf_printf(" ");
        (*printed)++;
    }
}

/**
 * ov_table_header - draw the header row of @t in panel rect @r (sorted column marked, cursor column
 * highlighted) and record where each column is, for ov_table_col_at().
 */
void ov_table_header(const OV_TABLE *t, const OV_LAYOUT *lay, OV_RECT r, ov_rgb_t fg)
{
    const OV_PANEL_STATE *ps = &lay->panel[t->panel];
    int                   vis[OV_TABLE_MAX_COLS];
    int                   nv     = ov_table_visible(t, lay, vis);
    int                   hs_rem = ps->hscroll, printed = 1, avail = r.width - 2;

    ov_buf_pos(r.row + 1, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_buf_printf(" ");
    s_nhit[t->panel] = 0;
    for (int k = 0; k < nv; k++)
    {
        const OV_COLUMN *col  = &t->cols[vis[k]];
        int              coll = (ps->collapsed >> vis[k]) & 1;
        if (k > 0 && !((ps->collapsed >> vis[k - 1]) & 1))
        {
            ov_theme_bg(OV_BG_HEADER);
            ov_table_sep(&hs_rem, &printed, avail);
        }
        int  w = ov_table_col_width(t, lay, vis[k]);
        char label[96], cell[256];
        if (col->flags & OV_COLF_NOLABEL)
        {
            label[0] = '\0';
        }
        else if (col->sort_key >= 0 && col->sort_key == ps->sort_key)
        {
            /* the arrow takes the label's last column when there is no room for both */
            int lw = (int) strlen(col->label) < w ? (int) strlen(col->label) : w - 1;
            snprintf(label, sizeof(label), "\x01%.*s%s\x02", lw, col->label,
                     ps->sort_dir ? "\xe2\x96\xbc" : "\xe2\x96\xb2");
        }
        else
        {
            snprintf(label, sizeof(label), "%s", col->label);
        }
        ov_cell_fit(label, w, (col->flags & OV_COLF_RIGHT) != 0, cell, sizeof(cell));
        if (s_nhit[t->panel] < OV_TABLE_MAX_COLS)
        {
            int h                  = s_nhit[t->panel]++;
            s_hit[t->panel][h].col = r.col + 1 + printed;
            s_hit[t->panel][h].w   = (coll ? 1 : w) + ((k + 1 < nv && !coll) ? 1 : 0);
            s_hit[t->panel][h].c   = vis[k];
        }
        ov_render_cell(vis[k], k, fg, OV_BG_HEADER, cell, &hs_rem, &printed, avail,
                       ps->highlight_col, ps->collapsed);
    }
    render_pad_spaces(printed, r.width);
    render_separator(r.row + 2, r.col + 1, r.width - 2, fg);
}

/** ov_table_col_at - table column under screen column @mc in @panel's header as last drawn, or -1. */
int ov_table_col_at(ov_focus_t panel, int mc)
{
    for (int h = 0; panel < OV_NB_LIST_PANELS && h < s_nhit[panel]; h++)
    {
        if (mc >= s_hit[panel][h].col && mc < s_hit[panel][h].col + s_hit[panel][h].w)
        {
            return s_hit[panel][h].c;
        }
    }
    return -1;
}

/**
 * ov_table_row - draw the cells of one row of @t at the cursor (after the 1-column focus strip) on
 * @row_bg. Return: columns printed, counting the strip.
 */
int ov_table_row(const OV_TABLE *t, const OV_ROW_CTX *rc, ov_rgb_t row_bg, int avail)
{
    const OV_PANEL_STATE *ps = &rc->lay->panel[t->panel];
    int                   vis[OV_TABLE_MAX_COLS];
    int                   nv     = ov_table_visible(t, rc->lay, vis);
    int                   hs_rem = ps->hscroll, printed = 1;
    for (int k = 0; k < nv; k++)
    {
        const OV_COLUMN *col = &t->cols[vis[k]];
        if (k > 0 && !((ps->collapsed >> vis[k - 1]) & 1))
        {
            ov_theme_bg(row_bg);
            ov_table_sep(&hs_rem, &printed, avail);
        }
        OV_COLCELL c;
        memset(&c, 0, sizeof(c));
        c.fg = OV_FG_TEXT;
        col->cell(&c, rc);
        char cell[256];
        ov_cell_fit(c.text, ov_table_col_width(t, rc->lay, vis[k]),
                    (col->flags & OV_COLF_RIGHT) != 0, cell, sizeof(cell));
        if (c.bold)
        {
            ov_buf_bold();
        }
        ov_render_cell(vis[k], k, c.fg, c.has_bg ? c.bg : row_bg, cell, &hs_rem, &printed, avail,
                       ps->highlight_col, ps->collapsed);
        if (c.bold)
        {
            ov_buf_reset_attr();
        }
    }
    ov_theme_bg(row_bg);
    return printed;
}
