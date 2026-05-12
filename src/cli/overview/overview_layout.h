/**
 * @file overview_layout.h
 * @brief Panel layout definitions for milkCTRL
 */

#ifndef OVERVIEW_LAYOUT_H
#define OVERVIEW_LAYOUT_H

#include <stdint.h>
#include <time.h>

/* View modes */
typedef enum {
    OV_VIEW_DASHBOARD = 0,
    OV_VIEW_STREAMS,
    OV_VIEW_PROCS,
    OV_VIEW_FPS,
    OV_VIEW_GRAPH,
    OV_VIEW_COUNT,
} ov_view_t;

/* Panel rectangle */
typedef struct {
    int row, col, height, width;
} OV_RECT;

/* Panel focus */
typedef enum {
    OV_FOCUS_STREAMS = 0,
    OV_FOCUS_PROCS,
    OV_FOCUS_FPS,
    OV_FOCUS_GRAPH,
    OV_FOCUS_COUNT,
} ov_focus_t;

/* ---- Command Log ---- */
#define OV_CMDLOG_MAX  32  /* ring buffer capacity */
#define OV_CMDLOG_MSG  96  /* max message length   */

typedef enum {
    OV_CMDLOG_INFO = 0, /* neutral informational */
    OV_CMDLOG_OK,       /* action succeeded      */
    OV_CMDLOG_FAIL,     /* action failed         */
    OV_CMDLOG_WARN,     /* warning / partial     */
} ov_cmdlog_level_t;

typedef struct {
    struct timespec    ts;
    char               msg[OV_CMDLOG_MSG];
    ov_cmdlog_level_t  level;
} OV_CMDLOG_ENTRY;

typedef struct {
    OV_CMDLOG_ENTRY entries[OV_CMDLOG_MAX];
    int  head;   /* next write position     */
    int  count;  /* entries currently stored */
} OV_CMDLOG;

void ov_cmdlog_push(
    OV_CMDLOG          *log,
    ov_cmdlog_level_t   level,
    const char         *fmt, ...);

/* Layout state */
typedef struct {
    int          term_rows;
    int          term_cols;
    ov_view_t    view;
    ov_focus_t   focus;
    int          sel_stream;
    int          sel_proc;
    int          sel_fps;
    int          sel_graph;
    int          scroll_stream;
    int          scroll_proc;
    int          scroll_fps;
    int          scroll_graph;
    int          show_help;
    int          help_sel;     /* cursor row in help */
    uint32_t     help_expand;  /* bitmask: 1=expanded */
    int          paused;
    char         filter[64];
    /* Per-panel regex filter strings */
    char         filter_stream[64];
    char         filter_proc[64];
    char         filter_fps[64];
    int          filter_editing; /* 1 = typing filter */
    int          filter_cursor;  /* cursor pos in filter */
    /* Dashboard panel rects */
    OV_RECT      r_header;
    OV_RECT      r_streams;
    OV_RECT      r_procs;
    OV_RECT      r_fps;
    OV_RECT      r_graph;
    OV_RECT      r_cmdlog;
    OV_RECT      r_status;
    /* Command log */
    OV_CMDLOG    cmdlog;
    int          cmdlog_rows; /* 0=hidden, default=4 */
    /* Control mode */
    int          ctrl_mode;
    int          ctrl_blink;
    /* Graph panel tab mode: 0=CONNECTIONS, 1=DETAILS, 2=RESOURCES */
    int          graph_tab_mode;
    /* Horizontal scroll per panel */
    int          hscroll_stream;
    int          hscroll_proc;
    int          hscroll_fps;
    /* Sort state per panel: 0=name, 1..N=column-specific */
    int          sort_key_stream;
    int          sort_key_proc;
    int          sort_key_fps;
    /* Sort direction per panel: 0=ascending, 1=descending */
    int          sort_dir_stream;
    int          sort_dir_proc;
    int          sort_dir_fps;
    int          sort_pending;
    /* Freeze selection: preview + cross-highlights
     * stay locked while navigation continues */
    int          freeze;
    ov_focus_t   freeze_focus;
    int          freeze_sel_stream;
    int          freeze_sel_proc;
    int          freeze_sel_fps;
    /* Lineage tracking mode: 0 = Trigger, 1 = Input */
    int          lineage_mode;
    /* Track selected names to handle external removals */
    char         sel_name_stream[80];
    char         sel_name_proc[80];
    char         sel_name_fps[80];
} OV_LAYOUT;

void ov_layout_compute(OV_LAYOUT *lay);

#endif /* OVERVIEW_LAYOUT_H */
