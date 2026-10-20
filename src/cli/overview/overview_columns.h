// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_columns.h
 * @brief Column tables of the list panels: one entry per column drives header, rows, hit-testing and sort
 *
 * Included by render/overview_render_internal.h, after the model, layout, theme and OV_RELATED types.
 */

#ifndef OVERVIEW_COLUMNS_H
#define OVERVIEW_COLUMNS_H

#include <regex.h>
#include <sys/types.h>

#define OV_COLF_RIGHT 0x01        /* right-aligned content */
#define OV_COLF_COMPACT_HIDE 0x02 /* hidden in compact mode ('d') */
#define OV_COLF_NOLABEL 0x04      /* no header text (marks, sparklines) */
#define OV_COLF_PIN 0x08          /* never dropped or scrolled away (mark, A, NAME) */
#define OV_COLF_ELASTIC 0x10 /* width is a minimum; spare room widens it up to OV_COL_ELASTIC_MAX */
#define OV_COLF_FILL 0x20    /* width is a minimum; takes the room left after the elastic columns */

#define OV_COL_ELASTIC_MAX 48

#define OV_TABLE_MAX_COLS 32 /* collapse bits are one per table index */

/* What a cell function sees of its row */
typedef struct
{
    const OV_LAYOUT  *lay;
    const OV_MODEL   *m;
    const OV_RELATED *rel;
    int               idx;       /* model index of the row's entry */
    int               is_sel;    /* cursor row */
    int               is_frozen; /* frozen entry */
    int               is_rel;    /* related to the selection in another panel */
    ov_focus_t        eff_focus; /* panel of the selection (frozen panel when frozen) */
    int8_t            depth;     /* ancestry distance to the selection, 0 = none */
    pid_t             sel_pid;   /* PID of the selected process, 0 = none */
    const regex_t    *re;        /* compiled panel filter, NULL = none */
} OV_ROW_CTX;

#define OV_COLCELL_MAX_SPANS 24

/* A styled run of a cell, from the previous run's end to code point @end (exclusive) */
typedef struct
{
    int      end;
    ov_rgb_t fg;
    int      has_bg;
    ov_rgb_t bg;
    int      bold;
} OV_COLSPAN;

/* One cell as a column function fills it: text may carry \x01..\x02 highlight markup, or be built from
 * styled runs with ov_cell_span() (then fg/bg/bold are unused) */
typedef struct
{
    char       text[192];
    ov_rgb_t   fg;
    int        has_bg; /* bg overrides the row background */
    ov_rgb_t   bg;
    int        bold;
    int        nspan;
    OV_COLSPAN span[OV_COLCELL_MAX_SPANS];
} OV_COLCELL;

typedef struct
{
    const char *id;       /* stable name: docs, future --sort */
    const char *label;    /* header text */
    const char *status;   /* [sort:...] text; NULL = label */
    int         width;    /* content columns, separator excluded */
    int         priority; /* higher is dropped first when the panel is too narrow */
    int         flags;    /* OV_COLF_* */
    int         sort_key; /* data-layer sort key (ov_sort_*), -1 = not sortable */
    int (*width_fn)(const OV_LAYOUT *lay); /* overrides width when set */
    void (*cell)(OV_COLCELL *c, const OV_ROW_CTX *rc);
} OV_COLUMN;

typedef struct
{
    const OV_COLUMN *cols;
    int              n;
    ov_focus_t       panel;
    const char      *what; /* "streams", "processes", "FPS": cmdlog wording */
} OV_TABLE;

/* Columns of a table as laid out in a panel: shown columns in screen order and their drawn widths */
typedef struct
{
    int n;
    int c[OV_TABLE_MAX_COLS]; /* table index */
    int w[OV_TABLE_MAX_COLS]; /* content width (1 when collapsed) */
    int used;                 /* columns used, separators included */
    int hscroll;              /* scroll offset actually applied */
    int more_left;            /* columns scrolled off on the left */
    int more_right;           /* columns dropped for room on the right */
} OV_TABLE_LAYOUT;

const OV_TABLE *ov_table_for(ov_focus_t panel);
void            ov_table_layout(const OV_TABLE  *t,
                                const OV_LAYOUT *lay,
                                int              width,
                                int              hscroll,
                                OV_TABLE_LAYOUT *lo);
void            ov_table_set_width(ov_focus_t panel, int width);
int             ov_table_width(ov_focus_t panel);
int             ov_table_visible(const OV_TABLE *t, const OV_LAYOUT *lay, int *vis);
int             ov_table_cursor_move(const OV_TABLE  *t,
                                     const OV_LAYOUT *lay,
                                     int              width,
                                     int             *cur,
                                     int             *hscroll,
                                     int              step);
int             ov_table_col_width(const OV_TABLE *t, const OV_LAYOUT *lay, int c);
int             ov_table_find_key(const OV_TABLE *t, int sort_key);
int             ov_table_find_id(const OV_TABLE *t, const char *id);
int             ov_table_next_sort(const OV_TABLE *t, const OV_LAYOUT *lay, int sort_key, int step);
const char     *ov_table_status_label(const OV_TABLE *t, int sort_key);
void            ov_table_header(const OV_TABLE *t, const OV_LAYOUT *lay, OV_RECT r, ov_rgb_t fg);
int             ov_table_row(const OV_TABLE *t, const OV_ROW_CTX *rc, ov_rgb_t row_bg, int avail);
int             ov_table_col_at(ov_focus_t panel, int mc);
void            ov_cell_span(OV_COLCELL *c, OV_COLSPAN style, const char *fmt, ...)
    __attribute__((format(printf, 3, 4)));

extern const OV_TABLE ov_fps_table;
extern const OV_TABLE ov_procs_table;
extern const OV_TABLE ov_streams_table;

#endif /* OVERVIEW_COLUMNS_H */
