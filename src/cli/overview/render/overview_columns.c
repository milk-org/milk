// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_columns.c
 * @brief   Column table lookups: visible columns, sort keys, labels (no drawing).
 */

#include "overview_render_internal.h"
#include <stdarg.h>
#include <string.h>

/** ov_table_for - column table of list panel @panel, or NULL for other panels. */
const OV_TABLE *ov_table_for(ov_focus_t panel)
{
    static const OV_TABLE *const tables[OV_NB_LIST_PANELS] = { &ov_streams_table, &ov_procs_table,
                                                               &ov_fps_table };
    return ((int) panel >= 0 && panel < OV_NB_LIST_PANELS) ? tables[panel] : NULL;
}

/** ov_table_col_width - content width of column @c (view-dependent widths included). */
int ov_table_col_width(const OV_TABLE *t, const OV_LAYOUT *lay, int c)
{
    return (t->cols[c].width_fn != NULL) ? t->cols[c].width_fn(lay) : t->cols[c].width;
}

/* Content width of each list panel as last drawn, for input that works on visible columns */
static int s_width[OV_NB_LIST_PANELS] = { 1000, 1000, 1000 };

/** ov_table_set_width - record the content width of @panel's table (called when its header is drawn). */
void ov_table_set_width(ov_focus_t panel, int width)
{
    if ((int) panel >= 0 && panel < OV_NB_LIST_PANELS)
    {
        s_width[panel] = width;
    }
}

/** ov_table_width - content width of @panel's table as last drawn. */
int ov_table_width(ov_focus_t panel)
{
    return ((int) panel >= 0 && panel < OV_NB_LIST_PANELS) ? s_width[panel] : 1000;
}

/* Columns used by the kept columns: collapsed ones take 1 and no separator after them */
static int measure(const OV_TABLE *t, const OV_LAYOUT *lay, const int *keep, int *w)
{
    uint32_t coll = lay->panel[t->panel].collapsed;
    int      used = 0, prev_sep = 0;
    for (int c = 0; c < t->n; c++)
    {
        if (!keep[c])
        {
            continue;
        }
        w[c] = ((coll >> c) & 1) ? 1 : ov_table_col_width(t, lay, c);
        used += prev_sep + w[c];
        prev_sep = !((coll >> c) & 1);
    }
    return used;
}

/**
 * ov_table_layout - lay out @t in @width columns: pinned columns, then the scrollable ones from @hscroll on,
 * dropping the highest priority first (the first scrollable one is always kept) until the rest fit; the spare
 * room widens elastic columns (NAME) up to OV_COL_ELASTIC_MAX, then fill columns.
 */
void ov_table_layout(const OV_TABLE  *t,
                     const OV_LAYOUT *lay,
                     int              width,
                     int              hscroll,
                     OV_TABLE_LAYOUT *lo)
{
    int keep[OV_TABLE_MAX_COLS] = { 0 }, w[OV_TABLE_MAX_COLS] = { 0 }, nscroll = 0, first = -1;
    for (int c = 0; c < t->n && c < OV_TABLE_MAX_COLS; c++)
    {
        nscroll += !(lay->compact_mode && (t->cols[c].flags & OV_COLF_COMPACT_HIDE)) &&
                   !(t->cols[c].flags & OV_COLF_PIN);
    }
    hscroll = (hscroll >= nscroll) ? nscroll - 1 : hscroll;
    hscroll = (hscroll < 0) ? 0 : hscroll;
    *lo     = (OV_TABLE_LAYOUT) { .hscroll = hscroll, .more_left = hscroll > 0 };
    for (int c = 0, j = 0; c < t->n && c < OV_TABLE_MAX_COLS; c++)
    {
        int flags = t->cols[c].flags;
        if (lay->compact_mode && (flags & OV_COLF_COMPACT_HIDE))
        {
            continue;
        }
        keep[c] = (flags & OV_COLF_PIN) || j++ >= hscroll;
        first   = (first < 0 && keep[c] && !(flags & OV_COLF_PIN)) ? c : first;
    }

    int used = measure(t, lay, keep, w);
    while (used > width)
    {
        int drop = -1;
        for (int c = 0; c < t->n; c++)
        {
            if (keep[c] && c != first && !(t->cols[c].flags & OV_COLF_PIN) &&
                (drop < 0 || t->cols[c].priority >= t->cols[drop].priority))
            {
                drop = c;
            }
        }
        if (drop < 0)
        {
            break;
        }
        keep[drop]     = 0;
        lo->more_right = 1;
        used           = measure(t, lay, keep, w);
    }

    uint32_t coll  = lay->panel[t->panel].collapsed;
    int      spare = width - used;
    for (int pass = 0; pass < 2 && spare > 0; pass++)
    {
        for (int c = 0; c < t->n && spare > 0; c++)
        {
            int flag = pass ? OV_COLF_FILL : OV_COLF_ELASTIC;
            if (keep[c] && (t->cols[c].flags & flag) && !((coll >> c) & 1))
            {
                int add = pass ? spare : OV_COL_ELASTIC_MAX - w[c];
                add     = (add > spare) ? spare : (add < 0) ? 0 : add;
                w[c] += add;
                spare -= add;
                used += add;
            }
        }
    }
    for (int c = 0; c < t->n; c++)
    {
        if (keep[c])
        {
            lo->c[lo->n]   = c;
            lo->w[lo->n++] = w[c];
        }
    }
    lo->used = used;
}

/** ov_table_visible - table indices of the columns shown as the panel was last drawn. Return: count. */
int ov_table_visible(const OV_TABLE *t, const OV_LAYOUT *lay, int *vis)
{
    OV_TABLE_LAYOUT lo;
    ov_table_layout(t, lay, s_width[t->panel], lay->panel[t->panel].hscroll, &lo);
    memcpy(vis, lo.c, sizeof(int) * (size_t) lo.n);
    return lo.n;
}

/* Whether table column @c is shown at scroll offset @hs */
static int shown_at(const OV_TABLE *t, const OV_LAYOUT *lay, int width, int hs, int c)
{
    OV_TABLE_LAYOUT lo;
    ov_table_layout(t, lay, width, hs, &lo);
    for (int k = 0; k < lo.n; k++)
    {
        if (lo.c[k] == c)
        {
            return 1;
        }
    }
    return 0;
}

/**
 * ov_table_cursor_move - move column cursor @cur (a table index) @step columns in table order, wrapping, and
 * scroll @hscroll the least that shows it. Return: the new cursor.
 */
int ov_table_cursor_move(const OV_TABLE  *t,
                         const OV_LAYOUT *lay,
                         int              width,
                         int             *cur,
                         int             *hscroll,
                         int              step)
{
    int all[OV_TABLE_MAX_COLS], na = 0, pos = 0, j = -1, nscroll = 0;
    for (int c = 0; c < t->n && c < OV_TABLE_MAX_COLS; c++)
    {
        if (!(lay->compact_mode && (t->cols[c].flags & OV_COLF_COMPACT_HIDE)))
        {
            pos       = (c == *cur) ? na : pos;
            all[na++] = c;
        }
    }
    if (na == 0)
    {
        return *cur;
    }
    pos = (pos + step + na) % na;
    for (int k = 0; k < na; k++)
    {
        j = (k == pos) ? nscroll : j;
        nscroll += !(t->cols[all[k]].flags & OV_COLF_PIN);
    }
    *cur = all[pos];
    if (t->cols[*cur].flags & OV_COLF_PIN)
    {
        *hscroll = (pos == 0) ? 0 : *hscroll;
        return *cur;
    }
    int hs = (*hscroll >= nscroll) ? nscroll - 1 : *hscroll;
    hs     = (hs < 0) ? 0 : hs;
    while (!shown_at(t, lay, width, hs, *cur) && hs != j)
    {
        hs += (j < hs) ? -1 : 1;
    }
    *hscroll = hs;
    return *cur;
}

/** ov_table_find_key - table index of the column sorting by @sort_key, or -1. */
int ov_table_find_key(const OV_TABLE *t, int sort_key)
{
    for (int c = 0; c < t->n; c++)
    {
        if (t->cols[c].sort_key >= 0 && t->cols[c].sort_key == sort_key)
        {
            return c;
        }
    }
    return -1;
}

/** ov_table_find_id - table index of column @id, or -1. */
int ov_table_find_id(const OV_TABLE *t, const char *id)
{
    for (int c = 0; c < t->n; c++)
    {
        if (strcmp(t->cols[c].id, id) == 0)
        {
            return c;
        }
    }
    return -1;
}

/**
 * ov_table_next_sort - sort key @step (+1/-1) visible sortable columns away from @sort_key, wrapping;
 * hidden and unsortable columns are skipped.
 */
int ov_table_next_sort(const OV_TABLE *t, const OV_LAYOUT *lay, int sort_key, int step)
{
    int vis[OV_TABLE_MAX_COLS], keys[OV_TABLE_MAX_COLS], nk = 0, pos = -1;
    int nv = ov_table_visible(t, lay, vis);
    for (int k = 0; k < nv; k++)
    {
        if (t->cols[vis[k]].sort_key >= 0)
        {
            pos        = (t->cols[vis[k]].sort_key == sort_key) ? nk : pos;
            keys[nk++] = t->cols[vis[k]].sort_key;
        }
    }
    if (nk == 0)
    {
        return sort_key;
    }
    if (pos < 0)
    {
        pos = (step > 0) ? -1 : 0;
    }
    return keys[((pos + step) % nk + nk) % nk];
}

/** ov_table_status_label - [sort:...] text of @sort_key ("name" when no column has it). */
const char *ov_table_status_label(const OV_TABLE *t, int sort_key)
{
    int c = ov_table_find_key(t, sort_key);
    return (c < 0) ? "name" : (t->cols[c].status != NULL) ? t->cols[c].status : t->cols[c].label;
}

/** ov_cp_count - code points in @s. */
static int ov_cp_count(const char *s)
{
    int n = 0;
    for (; *s != '\0'; s++)
    {
        n += ((*s & 0xC0) != 0x80);
    }
    return n;
}

/** ov_cell_span - append printf text to @c as one run in @style (its end is set here). */
void ov_cell_span(OV_COLCELL *c, OV_COLSPAN style, const char *fmt, ...)
{
    size_t  len = strlen(c->text);
    va_list ap;
    if (c->nspan >= OV_COLCELL_MAX_SPANS)
    {
        return;
    }
    va_start(ap, fmt);
    vsnprintf(c->text + len, sizeof(c->text) - len, fmt, ap);
    va_end(ap);
    style.end           = ov_cp_count(c->text);
    c->span[c->nspan++] = style;
}
