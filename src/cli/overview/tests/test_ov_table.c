// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkctrl_unit_table: column layout of the list panels by width, priority dropping, scrolling. */

#include "overview_data_internal.h"
#include "overview_render_internal.h"

#include <stdlib.h>

static int failures = 0;

#define CHECK(c)                                                         \
    do                                                                   \
    {                                                                    \
        if (!(c))                                                        \
        {                                                                \
            fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #c); \
            failures++;                                                  \
        }                                                                \
    } while (0)

static OV_LAYOUT lay;

static int shown(const OV_TABLE_LAYOUT *lo, int c)
{
    for (int k = 0; k < lo->n; k++)
    {
        if (lo->c[k] == c)
        {
            return 1;
        }
    }
    return 0;
}

/* Fits the width, keeps pinned columns, drops only columns of higher priority than every kept one */
static void check_fit(const OV_TABLE *t, int width)
{
    OV_TABLE_LAYOUT lo;
    ov_table_layout(t, &lay, width, 0, &lo);
    int sum = 0, maxkept = -1, first = -1;
    for (int k = 0; k < lo.n; k++)
    {
        sum += lo.w[k] + (k > 0);
        const OV_COLUMN *col = &t->cols[lo.c[k]];
        if (!(col->flags & OV_COLF_PIN))
        {
            first   = (first < 0) ? lo.c[k] : first;
            maxkept = (lo.c[k] != first && col->priority > maxkept) ? col->priority : maxkept;
        }
        if (col->flags & OV_COLF_ELASTIC)
        {
            CHECK(lo.w[k] >= col->width && lo.w[k] <= OV_COL_ELASTIC_MAX);
        }
    }
    CHECK(sum == lo.used && lo.used <= width);
    for (int c = 0; c < t->n; c++)
    {
        if (t->cols[c].flags & OV_COLF_PIN)
        {
            CHECK(shown(&lo, c));
        }
        else if (!shown(&lo, c))
        {
            CHECK(t->cols[c].priority >= maxkept && lo.more_right);
        }
    }
}

/* Shift+RIGHT visits every column in table order, each shown when the cursor is on it; LEFT comes back */
static void check_scroll(const OV_TABLE *t, int width)
{
    int cur = 0, hs = 0, seen = 0;
    for (int i = 0; i < t->n; i++)
    {
        OV_TABLE_LAYOUT lo;
        ov_table_layout(t, &lay, width, hs, &lo);
        CHECK(shown(&lo, cur));
        seen |= 1 << cur;
        ov_table_cursor_move(t, &lay, width, &cur, &hs, +1);
    }
    CHECK(cur == 0 && hs == 0); /* wrapped */
    CHECK(seen == (1 << t->n) - 1);
    for (int i = 0; i < t->n; i++)
    {
        ov_table_cursor_move(t, &lay, width, &cur, &hs, -1);
        OV_TABLE_LAYOUT lo;
        ov_table_layout(t, &lay, width, hs, &lo);
        CHECK(cur == t->n - 1 - i && shown(&lo, cur));
    }
}

int main(void)
{
    const OV_TABLE  *tab[3]   = { &ov_streams_table, &ov_procs_table, &ov_fps_table };
    static const int widths[] = { 60, 77, 117, 197, 400 };
    for (int i = 0; i < 3; i++)
    {
        for (size_t k = 0; k < sizeof(widths) / sizeof(widths[0]); k++)
        {
            check_fit(tab[i], widths[k]);
            check_scroll(tab[i], widths[k]);
        }
    }

    /* Wide: nothing dropped, NAME at its maximum; narrow: NAME at its minimum */
    OV_TABLE_LAYOUT lo;
    ov_table_layout(&ov_streams_table, &lay, 400, 0, &lo);
    CHECK(lo.n == ov_streams_table.n && !lo.more_right && lo.w[1] == OV_COL_ELASTIC_MAX);
    ov_table_layout(&ov_streams_table, &lay, 77, 0, &lo);
    CHECK(lo.more_right && lo.w[1] == ov_streams_table.cols[1].width);

    /* Compact mode hides its columns; collapsing a column takes one column and frees room */
    lay.compact_mode = 1;
    ov_table_layout(&ov_streams_table, &lay, 400, 0, &lo);
    CHECK(!shown(&lo, ov_table_find_id(&ov_streams_table, "inode")));
    lay.compact_mode                      = 0;
    int size                              = ov_table_find_id(&ov_streams_table, "size");
    lay.panel[OV_FOCUS_STREAMS].collapsed = 1U << size;
    ov_table_layout(&ov_streams_table, &lay, 400, 0, &lo);
    for (int k = 0; k < lo.n; k++)
    {
        CHECK(lo.c[k] != size || lo.w[k] == 1);
    }
    lay.panel[OV_FOCUS_STREAMS].collapsed = 0;

    /* Scrolling past the end clamps, and the first scrollable column always shows */
    ov_table_layout(&ov_procs_table, &lay, 40, 99, &lo);
    CHECK(lo.hscroll < ov_procs_table.n && lo.more_left && shown(&lo, ov_procs_table.n - 1));

    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}
