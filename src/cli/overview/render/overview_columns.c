// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_columns.c
 * @brief   Column table lookups: visible columns, sort keys, labels (no drawing).
 */

#include "overview_render_internal.h"
#include <stdarg.h>

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

/** ov_table_visible - table indices of the columns shown (compact mode hides some). Return: count. */
int ov_table_visible(const OV_TABLE *t, const OV_LAYOUT *lay, int *vis)
{
    int n = 0;
    for (int c = 0; c < t->n; c++)
    {
        if (!(lay->compact_mode && (t->cols[c].flags & OV_COLF_COMPACT_HIDE)))
        {
            vis[n++] = c;
        }
    }
    return n;
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
