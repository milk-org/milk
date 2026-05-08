/**
 * @file overview_layout.h
 * @brief Panel layout definitions for milkCTRL
 */

#ifndef OVERVIEW_LAYOUT_H
#define OVERVIEW_LAYOUT_H

/* View modes */
typedef enum {
    OV_VIEW_DASHBOARD = 0,
    OV_VIEW_GRAPH,
    OV_VIEW_STREAMS,
    OV_VIEW_PROCS,
    OV_VIEW_FPS,
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
    OV_RECT      r_status;
    /* Control mode */
    int          ctrl_mode;
    int          ctrl_blink;
    /* Detail pane: replaces CONNECTIONS when item selected */
    int          detail_mode;
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
} OV_LAYOUT;

void ov_layout_compute(OV_LAYOUT *lay);

#endif /* OVERVIEW_LAYOUT_H */
