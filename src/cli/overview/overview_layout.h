// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_layout.h
 * @brief Panel layout definitions for milk-CTRL
 */

#ifndef OVERVIEW_LAYOUT_H
#define OVERVIEW_LAYOUT_H

#include <stdint.h>
#include <time.h>
#include "overview_target.h"

/* View modes */
typedef enum
{
    OV_VIEW_DASHBOARD = 0,
    OV_VIEW_STREAMS,
    OV_VIEW_PROCS,
    OV_VIEW_FPS,
    OV_VIEW_GRAPH,
    OV_VIEW_LOOPS,
    OV_VIEW_COUNT,
} ov_view_t;

static inline const char *ov_view_label(ov_view_t v)
{
    switch (v)
    {
    case OV_VIEW_DASHBOARD:
        return "DASH";
    case OV_VIEW_STREAMS:
        return "STRM";
    case OV_VIEW_PROCS:
        return "PROC";
    case OV_VIEW_FPS:
        return "FPS";
    case OV_VIEW_GRAPH:
        return "CONN";
    case OV_VIEW_LOOPS:
        return "LOOPS";
    default:
        return "";
    }
}

/* Panel rectangle */
typedef struct
{
    int row, col, height, width;
} OV_RECT;

/* Panel focus */
typedef enum
{
    OV_FOCUS_STREAMS = 0,
    OV_FOCUS_PROCS,
    OV_FOCUS_FPS,
    OV_FOCUS_GRAPH,
    OV_FOCUS_COUNT,
} ov_focus_t;

/* Preview-bar button action IDs */
#define OV_BTN_NONE 0
#define OV_BTN_PROC_PAUSE 1 /* toggle pause/resume  */
#define OV_BTN_PROC_EXIT 2  /* clean exit (CTRLval=3)*/
#define OV_BTN_PROC_KILL 3  /* SIGTERM              */
#define OV_BTN_PROC_STEP 4  /* step (CTRLval=2)     */
#define OV_BTN_FPS_CONF 5   /* toggle conf          */
#define OV_BTN_FPS_RUN 6    /* toggle run           */
#define OV_BTN_FPS_KILL 7   /* SIGTERM FPS pids     */
#define OV_BTN_STREAM_DEL 8 /* delete stream SHM    */
#define OV_BTN_INSPECT 9    /* inspect sel. item    */

/* ---- Command Log ---- */
#define OV_CMDLOG_MAX 32 /* ring buffer capacity */
#define OV_CMDLOG_MSG 96 /* max message length   */

typedef enum
{
    OV_CMDLOG_INFO = 0, /* neutral informational */
    OV_CMDLOG_OK,       /* action succeeded      */
    OV_CMDLOG_FAIL,     /* action failed         */
    OV_CMDLOG_WARN,     /* warning / partial     */
} ov_cmdlog_level_t;

typedef struct
{
    struct timespec   ts;
    char              msg[OV_CMDLOG_MSG];
    ov_cmdlog_level_t level;
} OV_CMDLOG_ENTRY;

typedef struct
{
    OV_CMDLOG_ENTRY entries[OV_CMDLOG_MAX];
    int             head;  /* next write position     */
    int             count; /* entries currently stored */
} OV_CMDLOG;

void ov_cmdlog_push(OV_CMDLOG *log, ov_cmdlog_level_t level, const char *fmt, ...);
int  ov_cmdlog_drain_fd(OV_CMDLOG *log, int fd);


#include "overview_overlay.h" /* needs OV_RECT */
#include "overview_panel.h"

/* Layout state */
typedef struct
{
    OV_PANEL_STATE  panel[OV_NB_LIST_PANELS]; /* streams, processes, FPS (ov_focus_t index) */
    int             term_rows;
    int             term_cols;
    ov_view_t       view;
    ov_focus_t      focus;
    int             sel_graph;
    int             sel_loop;
    int             scroll_graph;
    int             scroll_loop;
    int             scroll_detail;
    int             loop_filter_active;
    OV_RENAME_STATE rename; /* loop rename prompt */
    int             detail_total_lines;
    OV_HELP_STATE   help; /* help overlay */
    int             paused;
    char            filter[64];
    ov_focus_t      filter_panel;   /* panel currently being edited/filtered */
    int             filter_active;  /* 1 = any panel filter active, 0 = none */
    int             filter_editing; /* 1 = typing filter */
    int             filter_cursor;  /* cursor pos in filter */
    int             filter_jump;    /* 1 = jump-to-match mode */
    /* Header filter badge hit rects / positions */
    int        r_filter_start[4];
    int        r_filter_width[4];
    ov_focus_t r_filter_panel[4];
    int        r_filter_count;
    /* Multi-select state for FPS batch ops (#8) */
    OV_TARGET  multi_sel_fps[OV_MSEL_MAX]; /* FPS batch marks, by name */
    int        multi_sel_count;
    OV_CONFIRM confirm; /* y/N modal for destructive and batch verbs */
    /* Compact mode (#13): hide extra columns */
    int compact_mode;
    /* Dashboard panel rects */
    OV_RECT        r_header;
    OV_RECT        r_tabs;
    OV_RECT        r_streams;
    OV_RECT        r_procs;
    OV_RECT        r_fps;
    OV_RECT        r_graph;
    OV_RECT        r_cmdlog;
    OV_RECT        r_status;
    OV_RECT        r_exit_btn, r_theme_btn; /* status-bar buttons as drawn; width 0 = not shown */
    OV_THEME_POPUP theme_popup;             /* theme selector popup */
    /* Command log */
    OV_CMDLOG cmdlog;
    int       cmdlog_rows; /* 0=hidden, default=4 */
    /* Control mode */
    int ctrl_mode;
    int ctrl_blink;
    /* Mouse hover track */
    int  mouse_hover;
    int  hover_view;         /* Logical focus enum (e.g. OV_FOCUS_STREAMS) or -1 */
    int  hover_idx;          /* Index of item hovered */
    int  hover_is_header;    /* 1 if hovering header */
    int  hover_col_logical;  /* Logical column index (0, 1, 2...) */
    char hover_tooltip[256]; /* Text to display in tooltip pass */
    /* Graph panel tab mode: 0=CONNECTIONS, 1=LOOPS, 2=DETAILS, 3=RESOURCES */
    int graph_tab_mode;
    int sort_pending;
    /* Freeze selection: preview + cross-highlights
     * stay locked while navigation continues */
    int        freeze;
    ov_focus_t freeze_focus;
    ov_focus_t
        graph_root_focus; /* list panel whose selection roots CONNECTIONS while GRAPH has focus */
    /* Lineage tracking mode: 0 = Trigger, 1 = Input */
    int lineage_mode;
    /* FPS parameter navigation (detail panel) */
    int  param_sel;     /* selected param (-1=none) */
    int  param_scroll;  /* scroll offset */
    int  param_editing; /* 1 = inline edit active */
    char param_edit_buf[200];
    int  param_edit_pos; /* cursor in edit buffer */
    /* FPS parameter tree state (F5 full-screen view) */
    int  fps_param_focus;     /* 0=FPS list, 1=param tree */
    char fps_param_path[200]; /* Current tree path, e.g. "conf" or "conf.sub" */
    int  fps_param_sel;       /* selected row in param tree */
    int  fps_param_scroll;    /* scroll offset in param tree */
    OV_INPUT_HISTORY
    *hist; /* F5 tree positions per FPS and directory; heap, owned by input history */
    /* F5 view split rects */
    OV_RECT r_fps_list;   /* left: FPS list  */
    OV_RECT r_fps_params; /* right: param tree */
    /* Preview-bar action buttons (row 2) */
    struct
    {
        int col;   /* 1-based start column (0 = unused) */
        int width; /* visible width in columns */
        int id;    /* action ID: OV_BTN_* */
    } preview_btns[6];
    int nb_preview_btns;
    /* F5 view drag state */
    float fps_split_ratio;
    int   fps_split_dragging;
    int   fps_split_hover;
    /* F2 dashboard view drag state */
    float dash_split_v_ratio;
    float dash_split_h_ratio;
    int   dash_split_v_dragging;
    int   dash_split_h_dragging;
    int   cmdlog_dragging;
    int   dash_split_v_hover;
    int   dash_split_h_hover;
    int   cmdlog_split_hover;
} OV_LAYOUT;
_Static_assert(sizeof(OV_LAYOUT) < 64 * 1024,
               "OV_LAYOUT is a stack local: keep large state on the heap");

void ov_input_history_free(OV_LAYOUT *lay); /* input/overview_input_history.c */

void ov_layout_compute(OV_LAYOUT *lay);
int  ov_view_has_panel(ov_view_t v, ov_focus_t f);
void ov_set_view(OV_LAYOUT *lay, ov_view_t v);

static inline int ov_get_num_cols(const OV_LAYOUT *lay, ov_focus_t focus)
{
    if (focus == OV_FOCUS_STREAMS)
    {
        return lay->compact_mode ? 9 : 12;
    }
    else if (focus == OV_FOCUS_PROCS)
    {
        return lay->compact_mode ? 11 : 16;
    }
    else if (focus == OV_FOCUS_FPS)
    {
        return lay->compact_mode ? 7 : 8;
    }
    return 1;
}

static inline int ov_get_logical_col_stream(int vis_col, int compact)
{
    if (!compact)
    {
        return vis_col;
    }
    if (vis_col <= 5)
    {
        return vis_col;
    }
    if (vis_col == 6)
    {
        return 7;
    }
    if (vis_col == 7)
    {
        return 10;
    }
    if (vis_col == 8)
    {
        return 11;
    }
    return vis_col;
}

static inline int ov_get_logical_col_fps(int vis_col, int compact)
{
    (void) compact;
    return vis_col;
}

static inline int ov_get_logical_col_proc(int vis_col, int compact)
{
    if (!compact)
    {
        return vis_col;
    }
    if (vis_col <= 6)
    {
        return vis_col;
    }
    if (vis_col == 7)
    {
        return 11;
    }
    if (vis_col == 8)
    {
        return 12;
    }
    if (vis_col == 9)
    {
        return 13;
    }
    if (vis_col == 10)
    {
        return 15;
    }
    return vis_col;
}

typedef struct
{
    int logical_col;
    int sort_key;
    int width;
} OV_COL_LAYOUT;

/**
 * @brief Populate column layout for STREAMS table.
 *
 * @param[in]  compact  1 if compact mode is enabled, 0 otherwise
 * @param[out] cols     Array to store column layout entries (min size 12)
 * @return Number of columns populated
 */
static inline int ov_get_stream_col_layout(int compact, OV_COL_LAYOUT *cols)
{
    int n     = 0;
    cols[n++] = (OV_COL_LAYOUT) { 0, 7, 3 };  /* A */
    cols[n++] = (OV_COL_LAYOUT) { 1, 0, 14 }; /* NAME */
    cols[n++] = (OV_COL_LAYOUT) { 2, 1, 4 };  /* TYP */
    cols[n++] = (OV_COL_LAYOUT) { 3, 2, 11 }; /* SIZE */
    cols[n++] = (OV_COL_LAYOUT) { 4, 3, 6 };  /* Hz */
    cols[n++] = (OV_COL_LAYOUT) { 5, 4, 7 };  /* MB/s */
    if (!compact)
    {
        cols[n++] = (OV_COL_LAYOUT) { 6, 5, 10 }; /* INODE */
    }
    cols[n++] = (OV_COL_LAYOUT) { 7, -1, 7 }; /* OWNER */
    if (!compact)
    {
        cols[n++] = (OV_COL_LAYOUT) { 8, 6, 10 };  /* COUNT */
        cols[n++] = (OV_COL_LAYOUT) { 9, -1, 10 }; /* SEMS */
    }
    cols[n++] = (OV_COL_LAYOUT) { 10, -1, 7 }; /* WPID */
    cols[n++] = (OV_COL_LAYOUT) { 11, -1, 7 }; /* RPID */
    return n;
}

/**
 * @brief Populate column layout for PROCS table.
 *
 * @param[in]  compact  1 if compact mode is enabled, 0 otherwise
 * @param[out] cols     Array to store column layout entries (min size 16)
 * @return Number of columns populated
 */
static inline int ov_get_proc_col_layout(int compact, OV_COL_LAYOUT *cols)
{
    int n     = 0;
    cols[n++] = (OV_COL_LAYOUT) { 0, 5, 3 };  /* A */
    cols[n++] = (OV_COL_LAYOUT) { 1, 0, 14 }; /* NAME */
    cols[n++] = (OV_COL_LAYOUT) { 2, 1, 7 };  /* PID */
    cols[n++] = (OV_COL_LAYOUT) { 3, 6, 4 };  /* PRIO */
    cols[n++] = (OV_COL_LAYOUT) { 4, 2, 5 };  /* STAT */
    cols[n++] = (OV_COL_LAYOUT) { 5, 3, 6 };  /* Hz */
    cols[n++] = (OV_COL_LAYOUT) { 6, 7, 6 };  /* UPTIME */
    if (!compact)
    {
        cols[n++] = (OV_COL_LAYOUT) { 7, -1, 3 };  /* TRG */
        cols[n++] = (OV_COL_LAYOUT) { 8, -1, 10 }; /* trig-strm */
        cols[n++] = (OV_COL_LAYOUT) { 9, -1, 8 };  /* exec */
        cols[n++] = (OV_COL_LAYOUT) { 10, 10, 5 }; /* DUTY */
    }
    cols[n++] = (OV_COL_LAYOUT) { 11, 8, 10 }; /* CPU% */
    cols[n++] = (OV_COL_LAYOUT) { 12, 9, 10 }; /* LOOPCNT */
    cols[n++] = (OV_COL_LAYOUT) { 13, 4, 5 };  /* MEM */
    if (!compact)
    {
        cols[n++] = (OV_COL_LAYOUT) { 14, -1, 10 }; /* MISSED */
    }
    cols[n++] = (OV_COL_LAYOUT) { 15, -1, 200 }; /* MSG */
    return n;
}

/**
 * @brief Populate column layout for FPS table.
 *
 * @param[in]  compact  1 if compact mode is enabled, 0 otherwise
 * @param[in]  view     Current overview view (OV_VIEW_FPS or other)
 * @param[out] cols     Array to store column layout entries (min size 8)
 * @return Number of columns populated
 */
static inline int ov_get_fps_col_layout(int compact, int view, OV_COL_LAYOUT *cols)
{
    int n      = 0;
    int desc_w = (view == OV_VIEW_FPS) ? 30 : 20;
    cols[n++]  = (OV_COL_LAYOUT) { 0, 3, 3 };  /* A */
    cols[n++]  = (OV_COL_LAYOUT) { 1, 0, 18 }; /* NAME */
    cols[n++]  = (OV_COL_LAYOUT) { 2, 5, 3 };  /* TMX */
    cols[n++]  = (OV_COL_LAYOUT) { 3, 1, 7 };  /* CPID */
    cols[n++]  = (OV_COL_LAYOUT) { 4, 4, 7 };  /* RPID */
    cols[n++]  = (OV_COL_LAYOUT) { 5, 6, 3 };  /* STR */
    cols[n++]  = (OV_COL_LAYOUT) { 6, 2, 5 };  /* MEM */
    if (!compact)
    {
        cols[n++] = (OV_COL_LAYOUT) { 7, -1, desc_w }; /* DESCRIPTION */
    }
    return n;
}

/**
 * @brief Hit-test a table column header given horizontal table offset.
 *
 * @param[in] cols            Array of column specifications
 * @param[in] num_cols        Number of columns
 * @param[in] collapsed_mask  Bitmask of collapsed logical columns
 * @param[in] table_x         0-based horizontal character offset in table data
 * @return Sort key index of clicked column, or -1 if none or non-sortable
 */
static inline int ov_header_hittest_sort_key(const OV_COL_LAYOUT *cols,
                                             int                  num_cols,
                                             uint32_t             collapsed_mask,
                                             int                  table_x)
{
    if (table_x < 0)
    {
        return -1;
    }

    int cur_x = 0;
    for (int c = 0; c < num_cols; c++)
    {
        int is_coll = (collapsed_mask & (1U << cols[c].logical_col)) != 0;
        int col_w   = is_coll ? 1 : cols[c].width;
        int sep_w   = (c < num_cols - 1 && !is_coll) ? 1 : 0;

        if (table_x >= cur_x && table_x < cur_x + col_w + sep_w)
        {
            return cols[c].sort_key;
        }
        cur_x += col_w + sep_w;
    }

    return -1;
}

#endif /* OVERVIEW_LAYOUT_H */
