// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef OVERVIEW_DATA_H
#define OVERVIEW_DATA_H

#include "overview_data_types.h"

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
    char           label[32];
    int            active;
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

    /* Membership indices */
    int nb_streams;
    int stream_indices[OV_MAX_LOOP_NODES / 2];
    int nb_procs;
    int proc_indices[OV_MAX_LOOP_NODES / 2];
    int nb_fps;
    int fps_indices[OV_MAX_LOOP_NODES / 2];

    /* Overlap metadata */
    uint32_t overlap_mask;       /* Bitmask of other loop IDs (1 << (id-1)) */
    int      nb_shared_nodes;    /* Count of nodes shared with other loops */
    int      nb_exclusive_nodes; /* Count of nodes private to this loop */

    /* Telemetry & Health */
    double min_hz; /* Bottleneck loop frequency */
    double max_hz;
    int    is_running; /* 1 if all processes in loop are RUN */
    int    is_paused;  /* 1 if any process in loop is PAUS */
    int    is_stale;   /* 1 if any process has unchanging loopcnt */
    int    is_error;   /* 1 if any process is ERR */
} OV_LOOP;


/* =========================================================
 * Complete system model
 * ========================================================= */

typedef struct
{
    /* data arrays */
    OV_STREAM streams[OV_MAX_STREAMS];
    int       nb_streams;

    OV_FPS fps[OV_MAX_FPS];
    int    nb_fps;

    OV_PROC procs[OV_MAX_PROCS];
    int     nb_procs;

    /* graph */
    OV_NODE nodes[OV_MAX_NODES];
    int     nb_nodes;

    OV_EDGE edges[OV_MAX_EDGES];
    int     nb_edges;

    /* detected loops */
    OV_LOOP loops[OV_MAX_LOOPS];
    int     nb_loops;

    /* scan metadata */
    double          scan_time_ms;
    uint64_t        scan_count;
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
void ov_add_edge(OV_MODEL *model, int src, int tgt, ov_edge_type_t type, const char *label);

/* =========================================================
 * Sorting helpers
 * ========================================================= */

/**
 * ov_sort_set_depths - set depths array for ancestry sorting.
 */
void ov_sort_set_depths(const int8_t *depths);

/**
 * ov_sort_streams - sort streams array in-place.
 * @model: model whose streams to sort
 * @key:   0=name, 1=type, 2=size, 3=Hz, 4=inode, 5=count
 * @dir:   0=ascending, 1=descending
 */
void ov_sort_streams(OV_MODEL *model, int key, int dir);

/**
 * ov_sort_procs - sort procs array in-place.
 * @model: model whose procs to sort
 * @key:   0=name, 1=PID, 2=status, 3=Hz
 * @dir:   0=ascending, 1=descending
 */
void ov_sort_procs(OV_MODEL *model, int key, int dir);

/**
 * ov_sort_fps - sort FPS array in-place.
 * @model: model whose FPS entries to sort
 * @key:   0=name, 1=conf+run alive status
 * @dir:   0=ascending, 1=descending
 */
void ov_sort_fps(OV_MODEL *model, int key, int dir);

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
/**
 * ov_model_export_snapshot - dump model to a text file.
 * @m: model to export
 *
 * Writes a timestamped snapshot to /tmp/milk-CTRL_snapshot_*.txt
 * containing all streams, processes, and FPS entries.
 */
void ov_model_export_snapshot(const OV_MODEL *m);

#endif /* OVERVIEW_DATA_H */
