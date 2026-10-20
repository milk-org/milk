// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_render_columns.c
 * @brief   Draw list-panel headers and rows from their column tables; record header hit rectangles.
 *
 * Column c of a table is collapsed when bit c of the panel's collapsed mask is set ('t'), and the
 * column cursor (Shift+Left/Right) is a table index. Header and rows share one ov_table_layout() of the
 * panel width; one space separates a column from the next unless it is collapsed.
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

/** fit_cps - code points of @s, \x01/\x02 markup excluded. */
static int fit_cps(const char *s)
{
    int n = 0;
    for (; *s != '\0'; s++)
    {
        n += ((*s & 0xC0) != 0x80 && *s != 0x01 && *s != 0x02);
    }
    return n;
}

/**
 * ov_cell_fit - @s laid out in exactly @width columns (one per code point, \x01/\x02 markup kept) into @out:
 * padded on the side opposite to @right alignment, or cut with a trailing ellipsis.
 */
static void ov_cell_fit(const char *s, int width, int right, char *out, size_t n)
{
    char   body[256];
    size_t b     = 0;
    int    cut   = (fit_cps(s) > width && width > 1);
    int    limit = cut ? width - 1 : width;
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
        if (cols == limit)
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
    if (cut)
    {
        memcpy(body + b, "\xe2\x80\xa6", 3); /* … */
        b += 3;
        cols++;
    }
    body[b] = '\0';
    int pad = width - cols;
    snprintf(out, n, "%*s%s%*s", right ? pad : 0, "", body, right ? 0 : pad, "");
}

/** ov_table_sep - one separator space, within the panel width. */
static void ov_table_sep(int *printed, int avail)
{
    if (*printed < avail)
    {
        ov_buf_printf(" ");
        (*printed)++;
    }
}

/**
 * ov_table_header - draw the header row of @t in panel rect @r (sorted column marked, cursor column
 * highlighted, ‹ › where columns are scrolled off or dropped) and record where each column is, for
 * ov_table_col_at().
 */
void ov_table_header(const OV_TABLE *t, const OV_LAYOUT *lay, OV_RECT r, ov_rgb_t fg)
{
    const OV_PANEL_STATE *ps = &lay->panel[t->panel];
    OV_TABLE_LAYOUT       lo;
    int                   printed = 1, avail = r.width - 2;
    ov_table_set_width(t->panel, avail - 2); /* the focus strip, and the last column for › */
    ov_table_layout(t, lay, avail - 2, ps->hscroll, &lo);

    ov_buf_pos(r.row + 1, r.col + 1);
    ov_theme_bg(OV_BG_HEADER);
    ov_theme_fg(OV_FG_WARN);
    ov_buf_printf("%s", lo.more_left ? "\xe2\x80\xb9" : " "); /* ‹ */
    s_nhit[t->panel] = 0;
    for (int k = 0; k < lo.n; k++)
    {
        int              c    = lo.c[k];
        const OV_COLUMN *col  = &t->cols[c];
        int              coll = (ps->collapsed >> c) & 1;
        if (k > 0 && !((ps->collapsed >> lo.c[k - 1]) & 1))
        {
            ov_theme_bg(OV_BG_HEADER);
            ov_table_sep(&printed, avail);
        }
        int  w = lo.w[k];
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
            s_hit[t->panel][h].w   = (coll ? 1 : w) + ((k + 1 < lo.n && !coll) ? 1 : 0);
            s_hit[t->panel][h].c   = c;
        }
        ov_render_cell(c, fg, OV_BG_HEADER, cell, &printed, avail, ps->highlight_col,
                       ps->collapsed);
    }
    if (lo.more_right && printed < avail)
    {
        ov_theme_bg(OV_BG_HEADER);
        ov_buf_hline(' ', avail - 1 - printed);
        ov_theme_fg(OV_FG_WARN);
        ov_buf_printf("\xe2\x80\xba"); /* › */
        printed = avail;
    }
    ov_theme_bg(OV_BG_HEADER);
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

/** ov_table_spans - draw the styled runs of @c in @w columns (cut at @w, padded on the free side). */
static void ov_table_spans(const OV_COLCELL     *c,
                           int                   w,
                           int                   right,
                           int                   lc,
                           ov_rgb_t              row_bg,
                           const OV_PANEL_STATE *ps,
                           int                  *printed,
                           int                   avail)
{
    int total = c->span[c->nspan - 1].end;
    int pad   = (total < w) ? w - total : 0;
    int cut   = (total > w && w > 1);
    w         = cut ? w - 1 : w;
    char piece[256];
    snprintf(piece, sizeof(piece), "%*s", pad, "");
    if (right && pad > 0)
    {
        ov_render_cell(lc, OV_FG_DIM, row_bg, piece, printed, avail, ps->highlight_col, 0);
    }
    const char *p = c->text;
    for (int k = 0, cp = 0; k < c->nspan && cp < w; k++)
    {
        const OV_COLSPAN *sp = &c->span[k];
        size_t            b  = 0;
        for (; cp < sp->end && *p != '\0'; cp++)
        {
            int clen = utf8_char_length((unsigned char) *p);
            for (int i = 0; i < clen && *p != '\0'; i++, p++)
            {
                if (cp < w && b + 1 < sizeof(piece))
                {
                    piece[b++] = *p;
                }
            }
        }
        piece[b] = '\0';
        if (sp->bold)
        {
            ov_buf_bold();
        }
        ov_render_cell(lc, sp->fg, sp->has_bg ? sp->bg : row_bg, piece, printed, avail,
                       ps->highlight_col, 0);
        if (sp->bold)
        {
            ov_buf_reset_attr();
        }
    }
    if (cut)
    {
        ov_render_cell(lc, OV_FG_DIM, row_bg, "\xe2\x80\xa6", printed, avail, ps->highlight_col, 0);
    }
    if (!right && pad > 0)
    {
        snprintf(piece, sizeof(piece), "%*s", pad, "");
        ov_render_cell(lc, OV_FG_DIM, row_bg, piece, printed, avail, ps->highlight_col, 0);
    }
}

/**
 * ov_table_row - draw the cells of one row of @t at the cursor (after the 1-column focus strip) on
 * @row_bg, laid out as its header. Return: columns printed, counting the strip.
 */
int ov_table_row(const OV_TABLE *t, const OV_ROW_CTX *rc, ov_rgb_t row_bg, int avail)
{
    const OV_PANEL_STATE *ps = &rc->lay->panel[t->panel];
    OV_TABLE_LAYOUT       lo;
    int                   printed = 1;
    ov_table_layout(t, rc->lay, avail - 2, ps->hscroll, &lo);
    for (int k = 0; k < lo.n; k++)
    {
        int              c   = lo.c[k];
        const OV_COLUMN *col = &t->cols[c];
        if (k > 0 && !((ps->collapsed >> lo.c[k - 1]) & 1))
        {
            ov_theme_bg(row_bg);
            ov_table_sep(&printed, avail);
        }
        OV_COLCELL cc;
        memset(&cc, 0, sizeof(cc));
        cc.fg = OV_FG_TEXT;
        col->cell(&cc, rc);
        int w = lo.w[k];
        if (cc.nspan > 0 && !((ps->collapsed >> c) & 1))
        {
            ov_table_spans(&cc, w, (col->flags & OV_COLF_RIGHT) != 0, c, row_bg, ps, &printed,
                           avail);
            continue;
        }
        char cell[256];
        ov_cell_fit(cc.text, w, (col->flags & OV_COLF_RIGHT) != 0, cell, sizeof(cell));
        if (cc.bold)
        {
            ov_buf_bold();
        }
        ov_render_cell(c, cc.fg, cc.has_bg ? cc.bg : row_bg, cell, &printed, avail,
                       ps->highlight_col, ps->collapsed);
        if (cc.bold)
        {
            ov_buf_reset_attr();
        }
    }
    ov_theme_bg(row_bg);
    return printed;
}
