/**
 * @file overview_data.h
 * @brief Unified data model for milkCTRL
 *
 * Aggregates stream, FPS, and processinfo data into
 * a single graph model with nodes and directed edges.
 * The model is built by scanning shared memory and is
 * double-buffered for lock-free display.
 */

#ifndef OVERVIEW_DATA_H
#define OVERVIEW_DATA_H

#include <stdint.h>
#include <sys/types.h>
#include <pthread.h>

#include "ImageStreamIO/ImageStruct.h"
#include "processinfo.h"
#include "fps_types.h"

/* =========================================================
 * Limits
 * ========================================================= */

#define OV_MAX_STREAMS     2000
#define OV_MAX_FPS          200
#define OV_MAX_PROCS        500
#define OV_MAX_NODES       2700
#define OV_MAX_EDGES       5000

#define OV_SPARKLINE_LEN     40

/* =========================================================
 * PID status (used for uniform PID coloring)
 * ========================================================= */

typedef enum
{
    OV_PID_DEAD   = 0,
    OV_PID_ALIVE  = 1,
    OV_PID_ZOMBIE = 2,
} ov_pid_status_t;

/**
 * pid_get_status - check PID liveness and zombie state.
 */
ov_pid_status_t pid_get_status(pid_t pid);

/* =========================================================
 * Node types
 * ========================================================= */

typedef enum
{
    OV_NODE_STREAM = 0,
    OV_NODE_FPS    = 1,
    OV_NODE_PROC   = 2,
} ov_node_type_t;

/* =========================================================
 * Edge relationship labels
 * ========================================================= */

typedef enum
{
    OV_EDGE_PROC_WRITES_STREAM = 0,
    OV_EDGE_STREAM_TRIGGERS_PROC,
    OV_EDGE_FPS_RUNS_PROC,
    OV_EDGE_FPS_INPUT_STREAM,
    OV_EDGE_FPS_OUTPUT_STREAM,
    OV_EDGE_PROC_TRIGGER_STREAM,
    OV_EDGE_STREAM_READ_BY_PROC,
} ov_edge_type_t;

/* =========================================================
 * Stream info (aggregated from IMAGE_METADATA)
 * ========================================================= */

typedef struct
{
    char     name[STRINGMAXLEN_IMAGE_NAME];
    int      valid;
    int      active;

    /* geometry */
    uint8_t  datatype;
    uint8_t  naxis;
    uint32_t size[3];
    uint64_t nelement;

    /* counters & timing */
    uint64_t cnt0;
    uint64_t cnt0_prev;
    double   update_hz;
    int      cnt_active;  /**< cnt0 changed since last scan */

    /* ownership */
    pid_t    creatorPID;
    pid_t    ownerPID;
    ino_t    inode;

    /* semaphores */
    int      nb_sem;
    int      semval[10];

    /* Write / read PIDs */
    pid_t    write_pid;       /**< writer PID (from proc trace) */
    int      nb_read_pids;
    pid_t    read_pids[IMAGE_NB_SEMAPHORE];

    /* process trace (from STREAM_PROC_TRACE) */
    int      nb_proctrace;
    pid_t    proctrace_pid[IMAGE_NB_PROCTRACE];
    ino_t    proctrace_inode[IMAGE_NB_PROCTRACE];
    int      proctrace_trigmode[IMAGE_NB_PROCTRACE];
    int      proctrace_status[IMAGE_NB_PROCTRACE];

    /* static string cache */
    char     size_str[32];

    /* sparkline history */
    float    spark_rate[OV_SPARKLINE_LEN];
    int      spark_idx;

    /* graph node index (-1 if not in graph) */
    int      node_idx;
} OV_STREAM;


/* =========================================================
 * FPS info (aggregated from FPS)
 * ========================================================= */

#define OV_FPS_MAX_STREAM_PARAMS 24

typedef struct
{
    char     name[STRINGMAXLEN_FPS_NAME];
    char     description[200];
    int      valid;

    /* status */
    uint32_t md_status;
    pid_t    confpid;
    pid_t    runpid;
    
    long     mem_rss_kb;
    int      conf_alive;
    int      run_alive;

    /* stream-type parameters (for edges) */
    int      nb_stream_params;
    char     stream_param_name[OV_FPS_MAX_STREAM_PARAMS]
             [FUNCTION_PARAMETER_STRMAXLEN];
    char     stream_param_value[OV_FPS_MAX_STREAM_PARAMS]
             [FUNCTION_PARAMETER_STRMAXLEN];
    uint64_t stream_param_flags[OV_FPS_MAX_STREAM_PARAMS];

#define OV_FPS_MAX_DISP_PARAMS 100
    /* display parameters (read-only list) */
    int      nb_disp_params;
    char     disp_param_name[OV_FPS_MAX_DISP_PARAMS]
             [FUNCTION_PARAMETER_STRMAXLEN];
    char     disp_param_value[OV_FPS_MAX_DISP_PARAMS]
             [FUNCTION_PARAMETER_STRMAXLEN];

    /* graph node index */
    int      node_idx;
} OV_FPS;


/* =========================================================
 * Process info (aggregated from PROCESSINFO)
 * ========================================================= */

typedef struct
{
    char     name[40];
    pid_t    PID;
    int      valid;
    int      active;

    /* status */
    int      loopstat;
    int      CTRLval;

    /* counters */
    int64_t  loopcnt;
    int      cnt_active;  /**< loopcnt changed since last scan */

    /* timing */
    long     dtmedian_iter_ns;
    long     dtmedian_exec_ns;
    double   loop_hz;

    /* trigger */
    char     trigstreamname[200];
    int      triggermode;
    int      triggersem;
    int      triggermissed;
    uint64_t triggermissed_cumul;
    int      MeasureTiming;

    /* CPU & Memory */
    int      rt_priority;
    float    cpu_used;
    long     mem_rss_kb;

    /* sparkline history */
    float    spark_cpu[OV_SPARKLINE_LEN];
    int      spark_idx;

    /* graph node index */
    int      node_idx;
} OV_PROC;


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
    int            gx;
    int            gy;
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
 * Complete system model
 * ========================================================= */

typedef struct
{
    /* data arrays */
    OV_STREAM streams[OV_MAX_STREAMS];
    int       nb_streams;

    OV_FPS    fps[OV_MAX_FPS];
    int       nb_fps;

    OV_PROC   procs[OV_MAX_PROCS];
    int       nb_procs;

    /* graph */
    OV_NODE   nodes[OV_MAX_NODES];
    int       nb_nodes;

    OV_EDGE   edges[OV_MAX_EDGES];
    int       nb_edges;

    /* scan metadata */
    double    scan_time_ms;
    uint64_t  scan_count;
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
 * ov_scan_has_new_data - check if the first scan has completed.
 * Return: 1 if new data is ready, 0 otherwise.
 */
int ov_scan_has_new_data(void);

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
int ov_find_stream_by_inode(
    const OV_MODEL *model,
    ino_t inode);

/**
 * ov_find_stream_by_name - find stream index by name.
 * @model: model to search
 * @name:  stream name
 *
 * Return: stream index, or -1 if not found.
 */
int ov_find_stream_by_name(
    const OV_MODEL *model,
    const char *name);

/**
 * ov_find_proc_by_pid - find process index by PID.
 * @model: model to search
 * @pid:   process PID
 *
 * Return: process index, or -1 if not found.
 */
int ov_find_proc_by_pid(
    const OV_MODEL *model,
    pid_t pid);

/**
 * ov_add_edge - add an edge to the graph if not duplicate.
 * @model: model to modify
 * @src:   source node index
 * @tgt:   target node index
 * @type:  edge type
 * @label: human-readable label
 */
void ov_add_edge(
    OV_MODEL *model,
    int src,
    int tgt,
    ov_edge_type_t type,
    const char *label);

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
void ov_sort_streams(
    OV_MODEL *model, int key, int dir);

/**
 * ov_sort_procs - sort procs array in-place.
 * @model: model whose procs to sort
 * @key:   0=name, 1=PID, 2=status, 3=Hz
 * @dir:   0=ascending, 1=descending
 */
void ov_sort_procs(
    OV_MODEL *model, int key, int dir);

/**
 * ov_sort_fps - sort FPS array in-place.
 * @model: model whose FPS entries to sort
 * @key:   0=name, 1=conf+run alive status
 * @dir:   0=ascending, 1=descending
 */
void ov_sort_fps(
    OV_MODEL *model, int key, int dir);

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
int ov_filter_build(
    const char  *pattern,
    const char **names,
    int          count,
    int         *out,
    int          max_out);
/**
 * ov_model_export_snapshot - dump model to a text file.
 * @m: model to export
 *
 * Writes a timestamped snapshot to /tmp/milkCTRL_snapshot_*.txt
 * containing all streams, processes, and FPS entries.
 */
void ov_model_export_snapshot(const OV_MODEL *m);

#endif /* OVERVIEW_DATA_H */
