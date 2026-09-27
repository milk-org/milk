// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file overview_data.h
 * @brief Unified data model for milk-CTRL
 *
 * Aggregates stream, FPS, and processinfo data into
 * a single graph model with nodes and directed edges.
 * The model is built by scanning shared memory and is
 * double-buffered for lock-free display.
 */

#ifndef OVERVIEW_DATA_TYPES_H
#define OVERVIEW_DATA_TYPES_H

#include <stdint.h>
#include <sys/types.h>
#include <pthread.h>

#include "ImageStreamIO/ImageStruct.h"
#include "processinfo.h"
#include "fps_types.h"

/* =========================================================
 * Limits
 * ========================================================= */

#define OV_MAX_STREAMS 2000
#define OV_MAX_FPS 200
#define OV_MAX_PROCS 500
#define OV_MAX_NODES 2700
#define OV_MAX_EDGES 5000

#define OV_SPARKLINE_LEN 40

/* =========================================================
 * Node and edge types
 * ========================================================= */

typedef enum
{
    OV_NODE_STREAM = 0,
    OV_NODE_FPS    = 1,
    OV_NODE_PROC   = 2,
} ov_node_type_t;

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

/**
 * pid_get_core_utilization - get cores actively utilized by a process's threads.
 * @pid: process ID
 * @cores: output array of core IDs
 * @max_cores: maximum number of cores to read
 *
 * Return: number of cores written to @cores.
 */
int pid_get_core_utilization(pid_t pid, int *cores, int max_cores);

typedef struct
{
    uint64_t minflt;
    uint64_t majflt;
    uint64_t threads;
    uint64_t vol_ctxt;
    uint64_t nonvol_ctxt;
    uint64_t migrations;
} ov_advanced_stats_t;

/**
 * pid_get_advanced_stats - get scheduling, memory faults, and thread counts
 */
int pid_get_advanced_stats(pid_t pid, ov_advanced_stats_t *out);

typedef struct
{
    uint64_t instructions;
    uint64_t cache_misses;
    uint64_t branch_misses;
    uint64_t l1d_misses;
    uint64_t llc_misses;
    uint64_t dtlb_misses;

    double inst_per_loop;
    double cache_miss_per_loop;
    double branch_miss_per_loop;
    double l1d_miss_per_loop;
    double llc_miss_per_loop;
    double dtlb_miss_per_loop;
} ov_perf_counters_t;

/**
 * pid_read_perf_counters - get hardware metrics via perf_event_open
 * @loopcnt: current loop iteration count of the process, used to calculate per-loop rates.
 * Requires CAP_PERFMON or root, or perf_event_paranoid <= 2
 */
int pid_read_perf_counters(pid_t pid, int64_t loopcnt, ov_perf_counters_t *out);

/* =========================================================
 * Stream info (aggregated from IMAGE_METADATA)
 * ========================================================= */

typedef struct
{
    char name[STRINGMAXLEN_IMAGE_NAME];
    int  valid;
    int  active;

    /* geometry */
    uint8_t  datatype;
    uint8_t  naxis;
    uint32_t size[3];
    uint64_t nelement;

    /* counters & timing */
    uint64_t cnt0;
    uint64_t cnt0_prev;
    double   update_hz;
    int      cnt_active; /**< cnt0 changed since last scan */

    /* ownership */
    pid_t creatorPID;
    pid_t ownerPID;
    ino_t inode;

    /* semaphores */
    int nb_sem;
    int semval[10];

    /* Write / read PIDs */
    pid_t write_pid; /**< writer PID (from proc trace) */
    int   nb_read_pids;
    pid_t read_pids[IMAGE_NB_SEMAPHORE];

    /* process trace (from STREAM_PROC_TRACE) */
    int   nb_proctrace;
    pid_t proctrace_pid[IMAGE_NB_PROCTRACE];
    ino_t proctrace_inode[IMAGE_NB_PROCTRACE];
    int   proctrace_trigmode[IMAGE_NB_PROCTRACE];
    int   proctrace_status[IMAGE_NB_PROCTRACE];

    /* static string cache */
    char size_str[32];

    /* sparkline history */
    float spark_rate[OV_SPARKLINE_LEN];
    int   spark_idx;

    /* graph node index (-1 if not in graph) */
    int node_idx;

    /* loop membership */
    uint32_t loop_mask;
    int      nb_loops;
    int      primary_loop_id;

    /* new-item flash counter (frames remaining) */
    int is_new;
} OV_STREAM;


/* =========================================================
 * FPS info (aggregated from FPS)
 * ========================================================= */

#define OV_FPS_MAX_STREAM_PARAMS 24
#define OV_FPS_MAX_DISP_PARAMS 100

typedef struct
{
    int      nb_disp_params;
    char     disp_param_name[OV_FPS_MAX_DISP_PARAMS][FUNCTION_PARAMETER_STRMAXLEN];
    char     disp_param_value[OV_FPS_MAX_DISP_PARAMS][FUNCTION_PARAMETER_STRMAXLEN];
    char     disp_param_descr[OV_FPS_MAX_DISP_PARAMS][FUNCTION_PARAMETER_DESCR_STRMAXLEN];
    char     disp_param_min[OV_FPS_MAX_DISP_PARAMS][FUNCTION_PARAMETER_STRMAXLEN];
    char     disp_param_max[OV_FPS_MAX_DISP_PARAMS][FUNCTION_PARAMETER_STRMAXLEN];
    uint8_t  disp_param_has_min[OV_FPS_MAX_DISP_PARAMS];
    uint8_t  disp_param_has_max[OV_FPS_MAX_DISP_PARAMS];
    uint32_t disp_param_type[OV_FPS_MAX_DISP_PARAMS];
    uint64_t disp_param_flags[OV_FPS_MAX_DISP_PARAMS];
} OV_FPS_PARAMS;

/**
 * ov_fps_get_params - fetch display parameters for an FPS on-demand.
 * @fps_name: name of the FPS
 *
 * Return: pointer to thread-safe static parameters struct, or NULL.
 */
const OV_FPS_PARAMS *ov_fps_get_params(const char *fps_name);

typedef struct
{
    char name[STRINGMAXLEN_FPS_NAME];
    char description[200];
    int  valid;

    /* status */
    uint32_t md_status;
    pid_t    confpid;
    pid_t    runpid;

    int64_t mem_rss_kb;
    int     conf_alive;
    int     run_alive;

    /* tmux tracking */
    uint8_t tmux_flags;
#define OV_TMUX_CTRL 0x01
#define OV_TMUX_CONF 0x02
#define OV_TMUX_RUN 0x04

    /* stream-type parameters (for edges) */
    int      nb_stream_params;
    char     stream_param_name[OV_FPS_MAX_STREAM_PARAMS][FUNCTION_PARAMETER_STRMAXLEN];
    char     stream_param_value[OV_FPS_MAX_STREAM_PARAMS][FUNCTION_PARAMETER_STRMAXLEN];
    uint64_t stream_param_flags[OV_FPS_MAX_STREAM_PARAMS];

    /* display parameters count */
    int nb_disp_params;

    /* graph node index */
    int node_idx;

    /* sparkline history: run-process Hz */
    float hz_hist[OV_SPARKLINE_LEN];
    int   hz_hist_idx;

    /* loop membership */
    uint32_t loop_mask;
    int      nb_loops;
    int      primary_loop_id;

    /* new-item flash counter (frames remaining) */
    int is_new;
} OV_FPS;

typedef struct
{
    char name[80];
    int  is_dir;
    int  param_idx;
} fps_tree_item_t;

int ov_get_fps_tree_items(const OV_FPS    *fps,
                          const char      *path,
                          fps_tree_item_t *items,
                          int              max_items);


/* =========================================================
 * Process info (aggregated from PROCESSINFO)
 * ========================================================= */

typedef struct
{
    char  name[40];
    pid_t PID;
    int   valid;
    int   active;

    /* status */
    int loopstat;
    int CTRLval;

    /* counters */
    int64_t loopcnt;
    int     cnt_active; /**< loopcnt changed since last scan */

    /* timing */
    int64_t dtmedian_iter_ns;
    int64_t dtmedian_exec_ns;
    double  loop_hz;

    /* trigger */
    char     trigstreamname[200];
    int      triggermode;
    int      triggersem;
    int      triggermissed;
    uint64_t triggermissed_cumul;
    int      MeasureTiming;

    /* CPU & Memory */
    int     rt_priority;
    float   cpu_used;
    int64_t mem_rss_kb;

    /* sparkline history */
    float spark_cpu[OV_SPARKLINE_LEN];
    int   spark_idx;

    /* graph node index */
    int node_idx;

    /* process start time (seconds since boot) */
    int64_t start_time_sec;

    /* stale detection: number of consecutive scans
     * where alive but loopcnt unchanged */
    int stale_count;

    /* new-item flash counter (frames remaining) */
    int is_new;

    /* loop membership */
    uint32_t loop_mask;
    int      nb_loops;
    int      primary_loop_id;

    /* status message / log */
    char statusmsg[200];
} OV_PROC;

#endif /* OVERVIEW_DATA_TYPES_H */
