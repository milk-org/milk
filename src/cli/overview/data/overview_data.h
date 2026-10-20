// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef OVERVIEW_DATA_H
#define OVERVIEW_DATA_H

#include "overview_data_types.h"
#include "overview_data_index.h"
#include "processinfo_labels.h"

/* =========================================================
 * Graph node
 * ========================================================= */

typedef struct
{
    ov_node_type_t type;
    int            index;
    char           name[100];
    int            active;
    /* layout coords (for graph view) */
    int gx;
    int gy;
} OV_NODE;


/* =========================================================
 * Graph edge
 * ========================================================= */

typedef struct
{
    int            src_node;
    int            tgt_node;
    ov_edge_type_t type;
    unsigned       evidence; /* OV_EV_* */
} OV_EDGE;


/* =========================================================
 * Loop / Cycle info
 * ========================================================= */

#define OV_MAX_LOOPS 32
#define OV_MAX_LOOP_NODES 32
#define OV_LOOP_NAME_LEN 48

typedef struct
{
    int      loop_id;                       /* 1-based loop ID: 1, 2, ... */
    uint64_t signature_hash;                /* 64-bit hash of canonical cycle */
    char     signature[256];                /* Canonical signature string */
    char     name[OV_LOOP_NAME_LEN];        /* Display name (custom or auto) */
    char     auto_name[OV_LOOP_NAME_LEN];   /* Generated name: "wfs_tt->dmcomb" */
    char     custom_name[OV_LOOP_NAME_LEN]; /* User-assigned custom name */
    int      has_custom_name;

    /* Cycle nodes in topological sequence (alternate stream <-> proc/fps) */
    int nb_nodes;
    int node_indices[OV_MAX_LOOP_NODES];

    /* Membership: model indices (the UI never reorders the model; display order is separate) */
    int nb_streams;
    int stream_indices[OV_MAX_LOOP_NODES / 2];
    int nb_procs;
    int proc_indices[OV_MAX_LOOP_NODES / 2];
    int nb_fps;
    int fps_indices[OV_MAX_LOOP_NODES / 2];
    unsigned
        fps_via_run; /* bit i: fps_indices[i] is in the loop through the process it runs, not on the cycle */

    /* Overlap metadata */
    uint32_t overlap_mask;       /* Bitmask of other loop IDs (1 << (id-1)) */
    int      nb_shared_nodes;    /* Count of nodes shared with other loops */
    int      nb_exclusive_nodes; /* Count of nodes private to this loop */

    /* Telemetry & Health */
    double min_hz; /* Bottleneck loop frequency */
    double max_hz;
    int    is_running; /* 1 if every process (and FPS run process) is RUN */
    int    is_paused;  /* 1 if any process in loop is PAUS */
    int    is_stale;   /* 1 if any running process has an unchanging loopcnt */
    int    is_error;   /* 1 if any process is ERR or CRSH */
} OV_LOOP;

/** ov_loop_units - processes and FPS of loop @lp, an FPS and the process it runs counting once. */
static inline int ov_loop_units(const OV_LOOP *lp)
{
    return lp->nb_procs + lp->nb_fps - __builtin_popcount(lp->fps_via_run);
}


/* =========================================================
 * Complete system model
 * ========================================================= */

typedef struct
{
    /* data arrays */
    OV_STREAM    streams[OV_MAX_STREAMS];
    int          nb_streams;
    int          streams_pending; /* listed streams not mapped yet: the scan comes back soon */
    int          streams_capped;  /* more *.im.shm than OV_MAX_STREAMS */
    OV_TRACE_HOP trace[OV_MAX_TRACE];
    int          nb_trace;

    OV_FPS fps[OV_MAX_FPS];
    int    nb_fps;

    OV_PROC procs[OV_MAX_PROCS];
    int     nb_procs;
    int     nb_procs_hidden; /* registry entries beyond OV_MAX_PROCS (stale ones go first) */

    /* graph */
    OV_NODE nodes[OV_MAX_NODES];
    int     nb_nodes;

    OV_EDGE edges[OV_MAX_EDGES];
    int     nb_edges;

    /* adjacency: the edges out of / into node v are adj_out[adj_out_off[v] .. adj_out_off[v + 1]) (and
     * adj_in likewise), in edge order; built by ov_model_adjacency */
    int adj_out_off[OV_MAX_NODES + 1];
    int adj_out[OV_MAX_EDGES];
    int adj_in_off[OV_MAX_NODES + 1];
    int adj_in[OV_MAX_EDGES];

    /* detected loops */
    OV_LOOP loops[OV_MAX_LOOPS];
    int     nb_loops;
    int     loops_truncated; /* cycle search hit its limit: more loops may exist */

    /* lookup indexes for ov_find_stream_by_name/_by_inode and ov_find_proc_by_pid, built with the graph */
    OV_INDEX_DEF(32768) ix_stream_name;
    OV_INDEX_DEF(32768) ix_stream_inode;
    OV_INDEX_DEF(1024) ix_proc_pid; /* one entry per PID: the live process, else the newest */
    int indexed;                    /* 0: the ov_find_* helpers scan the arrays */

    /* scan metadata */
    unsigned tmux_polls; /* tmux spawns since start (ov_scan_tmux_sessions) */
    double   scan_time_ms;
    uint64_t scan_count;
    uint64_t
        seq; /* publication number from the scan thread: unique per model, 0 = never published */
    struct timespec last_scan_time;
} OV_MODEL;


/* =========================================================
 * Scan API
 * ========================================================= */

/**
 * ov_scan_streams - scan SHM dir for streams.
 * @model: model to populate
 *
 * Scans SHAREDSHMDIR for *.im.shm files and populates
 * model->streams[].
 */
void ov_scan_streams(OV_MODEL *model);

/**
 * ov_scan_fps - scan SHM dir for FPS entries.
 * @model: model to populate
 *
 * Scans SHAREDSHMDIR for *.fps.shm files and reads
 * metadata + stream-type parameters.
 */
void ov_scan_fps(OV_MODEL *model);

/**
 * ov_scan_procs - scan processinfo list.
 * @model: model to populate
 *
 * Maps the processinfo list and reads active processes.
 */
void ov_scan_procs(OV_MODEL *model);

/**
 * ov_scan_tmux_sessions - check for live FPS tmux sessions.
 * @model: model to update
 *
 * Parses `tmux list-windows` to update the OV_TMUX_* flags
 * on the scanned FPS entries.
 */
void ov_scan_tmux_sessions(OV_MODEL *model);

/**
 * ov_build_graph - build node/edge graph from scan data.
 * @model: model to process
 *
 * Cross-references streams, FPS, and processes to build
 * the directed connection graph.
 */
void ov_build_graph(OV_MODEL *model);

/** ov_model_index - build the lookup indexes of @model (ov_build_graph does it first). */
void ov_model_index(OV_MODEL *model);
void ov_model_adjacency(OV_MODEL *model);

/* Timing of hop t of a stream's trigger chain (ov_trace_hop); -1 = unknown */
typedef struct
{
    int64_t compute_ns; /* trigger to write, when the process measures timing */
    int64_t handoff_ns; /* upstream write to this trigger */
    int64_t lag;        /* frames the trigger stream moved on since the write */
} OV_HOP;

int     ov_trace_hop(const OV_MODEL *m, const OV_STREAM *s, int t, OV_HOP *h);
int64_t ov_trace_end_to_end_ns(const OV_MODEL *m, const OV_STREAM *s);

/* for (int k = OV_ADJ_B(m, out, v); k < OV_ADJ_E(m, out, v); k++) e = &m->edges[m->adj_out[k]]:
 * the edges out of (out) or into (in) node v; an invalid v gives an empty range */
#define OV_ADJ_OK(m, v) ((v) >= 0 && (v) < (m)->nb_nodes)
#define OV_ADJ_B(m, d, v) (OV_ADJ_OK(m, v) ? (m)->adj_##d##_off[(v)] : 0)
#define OV_ADJ_E(m, d, v) (OV_ADJ_OK(m, v) ? (m)->adj_##d##_off[(v) + 1] : 0)

/**
 * ov_model_full_scan - run all four steps in sequence.
 * @model: model to populate
 */
void ov_model_full_scan(OV_MODEL *model);

/**
 * ov_scan_start - launch the background scan thread.
 * Return: 0 on success, -1 on failure.
 */
int ov_scan_start(void);

/**
 * ov_scan_stop - signal the scan thread to stop and join.
 */
void ov_scan_stop(void);

/**
 * ov_scan_get_model - pick up the latest complete model.
 * Return: pointer to the current display model.
 */
const OV_MODEL *ov_scan_get_model(void);

/**
 * ov_scan_get_event_fd - get eventfd notified on new scan data.
 * Return: eventfd file descriptor, or -1 if not initialized.
 */
int ov_scan_get_event_fd(void);

/**
 * ov_scan_has_new_data - check if the first scan has completed.
 * Return: 1 if new data is ready, 0 otherwise.
 */
int ov_scan_has_new_data(void);

/**
 * ov_scan_force_update - interrupt sleep to force an immediate scan.
 */
void ov_scan_force_update(void);

/* UI hints to the scan thread (ov_scan_set_hints) */
#define OV_HINT_PAUSED 0x1U      /* display paused: do not scan */
#define OV_HINT_FPS_VISIBLE 0x2U /* an FPS panel is on screen: tmux windows are worth polling */

void     ov_scan_set_hints(unsigned hints);
void     ov_scan_tmux_refresh(void);
unsigned ov_scan_hints(void);
unsigned ov_scan_hints_swap(unsigned hints);

/** ov_scan_get_interval / ov_scan_set_interval - scan period in seconds (set clamps to 0.1-10). */
float ov_scan_get_interval(void);
void  ov_scan_set_interval(float s);

/**
 * ov_scan_drain_event_fd - drain pending wakeups from scan eventfd.
 */
void ov_scan_drain_event_fd(void);


/**
 * ov_scan_cache_cleanup - release all persistent
 * SHM mappings held by the scan caches.
 *
 * Must be called when the scan thread stops to
 * avoid leaking file descriptors and mappings.
 */
void ov_scan_cache_cleanup(void);


/* =========================================================
 * Node / edge lookup helpers
 * ========================================================= */

/**
 * ov_find_stream_by_inode - find stream index by inode.
 * @model: model to search
 * @inode: inode value
 *
 * Return: stream index, or -1 if not found.
 */
int ov_find_stream_by_inode(const OV_MODEL *model, ino_t inode);

/**
 * ov_find_stream_by_name - find stream index by name.
 * @model: model to search
 * @name:  stream name
 *
 * Return: stream index, or -1 if not found.
 */
int ov_find_stream_by_name(const OV_MODEL *model, const char *name);

/**
 * ov_find_proc_by_pid - find process index by PID.
 * @model: model to search
 * @pid:   process PID
 *
 * Return: process index, or -1 if not found.
 */
int ov_find_proc_by_pid(const OV_MODEL *model, pid_t pid);

/**
 * ov_add_edge - add an edge to the graph if not duplicate.
 * @model: model to modify
 * @src:   source node index
 * @tgt:   target node index
 * @type:  edge type
 * @label: human-readable label
 */
void ov_add_edge(OV_MODEL *model, int src, int tgt, ov_edge_type_t type, unsigned evidence);

/* =========================================================
 * Sorting helpers
 * ========================================================= */

/**
 * ov_sort_set_depths - set depths array for ancestry sorting.
 */
void ov_sort_set_depths(const int8_t *depths);

/**
 * ov_sort_streams - fill an index array with the streams in sort order.
 * @model: model whose streams to sort
 * @key:   0=name, 1=type, 2=size, 3=Hz, 4=inode, 5=count
 * @dir:   0=ascending, 1=descending
 */
void ov_sort_streams(const OV_MODEL *model, int key, int dir, int *order);

/**
 * ov_sort_procs - fill an index array with the processes in sort order.
 * @model: model whose procs to sort
 * @key:   0=name, 1=PID, 2=status, 3=Hz
 * @dir:   0=ascending, 1=descending
 */
void ov_sort_procs(const OV_MODEL *model, int key, int dir, int *order);

/**
 * ov_sort_fps - fill an index array with the FPS in sort order.
 * @model: model whose FPS entries to sort
 * @key:   0=name, 1=conf+run alive status
 * @dir:   0=ascending, 1=descending
 */
void ov_sort_fps(const OV_MODEL *model, int key, int dir, int *order);

/* Display order of the list panels (panel numbers equal the UI's ov_focus_t list panels) */
#define OV_ORDER_STREAMS 0
#define OV_ORDER_PROCS 1
#define OV_ORDER_FPS 2
#define OV_ORDER_NB 3
void       ov_order_sort(const OV_MODEL *m, int panel, int key, int dir);
const int *ov_order_get(const OV_MODEL *m, int panel, int *n);
uint32_t   ov_order_gen(int panel);
void       ov_order_reset(void);

/**
 * ov_filter_build - build filtered index array.
 * @pattern:  regex pattern string (empty = match all)
 * @names:    array of name pointers
 * @count:    total item count
 * @out:      output index array (caller-allocated)
 * @max_out:  capacity of @out
 *
 * Return: number of matching indices written to @out.
 */
int ov_filter_build(const char *pattern, const char **names, int count, int *out, int max_out);
unsigned long ov_filter_compile_count(void);
int           ov_model_export_snapshot(const OV_MODEL *m, char *path, size_t pathsz);

/* States of processes and loops: the first seven follow processinfo's PROCESSINFO_LOOPSTAT_* */
typedef enum
{
    OV_ST_INIT,
    OV_ST_RUN,
    OV_ST_PAUS,
    OV_ST_STOP,
    OV_ST_ERR,
    OV_ST_SPIN,
    OV_ST_CRSH,
    OV_ST_STAL, /* running, counter not moving */
    OV_ST_IDLE, /* loop with no running member */
    OV_ST_UNKNOWN,
    OV_ST_COUNT
} ov_state_t;

/* Help legend of the states; a unit test checks it names every state */
#define OV_STATE_LEGEND                                                                            \
    "INIT, RUN, PAUS (paused), SPIN, STOP, ERR, CRSH (crashed), STAL (running, counter stopped); " \
    "loops IDLE "                                                                                  \
    "when no member runs"

const char *ov_state_name(ov_state_t st, int long_form);
ov_state_t  ov_state_of_loopstat(int loopstat);
const char *ov_loopstat_label(int loopstat, int long_form);

/** ov_stream_hops - the process-trace hops of stream @s (nb_proctrace of them). */
static inline const OV_TRACE_HOP *ov_stream_hops(const OV_MODEL *m, const OV_STREAM *s)
{
    return &m->trace[s->trace_off];
}

OV_TRACE_HOP *ov_stream_add_hop(OV_MODEL *m, OV_STREAM *s);

#endif /* OVERVIEW_DATA_H */
