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

#include "overview_view.h"

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
    ov_subtab_t     subtab;               /* sub-tab of the bottom-right panel */
    OV_SUBTAB_STATE sub[OV_SUBTAB_COUNT]; /* list cursor of each sub-tab */
    int             scroll_detail;        /* first line of DETAILS / RESOURCES text */
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
    int  sort_pending;
    /* Freeze selection: preview + cross-highlights
     * stay locked while navigation continues */
    int        freeze;
    ov_focus_t freeze_focus;
    ov_focus_t
        graph_root_focus; /* list panel whose selection roots CONNECTIONS while GRAPH has focus */
    /* Lineage tracking mode: 0 = Trigger, 1 = Input */
    int lineage_mode;
    /* FPS parameter navigation (detail panel) */
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

#endif /* OVERVIEW_LAYOUT_H */
