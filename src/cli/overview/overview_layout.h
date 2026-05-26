/**
 * @file overview_layout.h
 * @brief Panel layout definitions for milk-CTRL
 */

#ifndef OVERVIEW_LAYOUT_H
#define OVERVIEW_LAYOUT_H

#include <stdint.h>
#include <time.h>

/* View modes */
typedef enum
{
    OV_VIEW_DASHBOARD = 0,
    OV_VIEW_STREAMS,
    OV_VIEW_PROCS,
    OV_VIEW_FPS,
    OV_VIEW_GRAPH,
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

/* Layout state */
typedef struct
{
    int        term_rows;
    int        term_cols;
    ov_view_t  view;
    ov_focus_t focus;
    int        sel_stream;
    int        sel_proc;
    int        sel_fps;
    int        sel_graph;
    int        scroll_stream;
    int        scroll_proc;
    int        scroll_fps;
    int        scroll_graph;
    int        scroll_detail;
    int        detail_total_lines;
    int        show_help;
    int        help_sel;    /* cursor row in help */
    uint32_t   help_expand; /* bitmask: 1=expanded */
    int        paused;
    char       filter[64];
    /* Per-panel regex filter strings */
    char filter_stream[64];
    char filter_proc[64];
    char filter_fps[64];
    int  filter_editing; /* 1 = typing filter */
    int  filter_cursor;  /* cursor pos in filter */
    int  filter_jump;    /* 1 = jump-to-match mode */
    /* Multi-select state for FPS batch ops (#8) */
    uint8_t multi_sel_fps[200]; /* per-FPS select */
    int     multi_sel_count;    /* count of selected */
    /* Compact mode (#13): hide extra columns */
    int compact_mode;
    /* Dashboard panel rects */
    OV_RECT r_header;
    OV_RECT r_streams;
    OV_RECT r_procs;
    OV_RECT r_fps;
    OV_RECT r_graph;
    OV_RECT r_cmdlog;
    OV_RECT r_status;
    /* Command log */
    OV_CMDLOG cmdlog;
    int       cmdlog_rows; /* 0=hidden, default=4 */
    /* Control mode */
    int ctrl_mode;
    int ctrl_blink;
    /* Mouse hover track */
    int  mouse_hover;
    int  hover_view;          /* Logical focus enum (e.g. OV_FOCUS_STREAMS) or -1 */
    int  hover_idx;           /* Index of item hovered */
    int  hover_is_header;     /* 1 if hovering header */
    int  hover_col_logical;   /* Logical column index (0, 1, 2...) */
    char hover_tooltip[256];  /* Text to display in tooltip pass */
    int  hover_global_stream; /* Global stream index hovered (-1 if none) */
    int  hover_global_proc;   /* Global proc index hovered (-1 if none) */
    int  hover_global_fps;    /* Global fps index hovered (-1 if none) */
    /* Graph panel tab mode: 0=CONNECTIONS, 1=DETAILS, 2=RESOURCES */
    int graph_tab_mode;
    /* Horizontal scroll per panel */
    int hscroll_stream;
    int hscroll_proc;
    int hscroll_fps;
    /* Sort state per panel: 0=name, 1..N=column-specific */
    int sort_key_stream;
    int sort_key_proc;
    int sort_key_fps;
    /* Sort direction per panel: 0=ascending, 1=descending */
    int sort_dir_stream;
    int sort_dir_proc;
    int sort_dir_fps;
    int sort_pending;
    /* Freeze selection: preview + cross-highlights
     * stay locked while navigation continues */
    int        freeze;
    ov_focus_t freeze_focus;
    int        freeze_sel_stream;
    int        freeze_sel_proc;
    int        freeze_sel_fps;
    /* Lineage tracking mode: 0 = Trigger, 1 = Input */
    int lineage_mode;
    /* Track selected names to handle external removals */
    char  sel_name_stream[80];
    char  sel_name_proc[80];
    pid_t sel_pid_proc;
    char  sel_name_fps[80];
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
    /* FPS parameter tree history */
    struct
    {
        char fps_name[80];
        char path[200];
    } fps_last_path[200];
    int nb_fps_last_path;

    struct
    {
        char fps_name[80];
        char path[200];
        int  sel;
        int  scroll;
    } fps_dir_history[1000];
    int nb_fps_dir_history;
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

void ov_layout_compute(OV_LAYOUT *lay);

#endif /* OVERVIEW_LAYOUT_H */
