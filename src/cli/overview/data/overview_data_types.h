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

#define OV_MAX_STREAMS 10000
#define OV_MAX_FPS 200
#define OV_MAX_PROCS 500
#define OV_MAX_NODES (OV_MAX_STREAMS + OV_MAX_FPS + OV_MAX_PROCS)
#define OV_MAX_EDGES 16000
#define OV_MAX_TRACE 16384  /* process-trace hops of all streams (OV_MODEL.trace) */
#define OV_MAP_PER_SCAN 500 /* new stream mappings per scan; the rest wait (OV_STREAM.pending) */

#define OV_SPARKLINE_LEN 40
#define OV_STALE_SCANS 3 /* scans without a loopcnt change before an active process is STALE */
#define OV_NEW_SCANS 4   /* scans a newly appeared item stays highlighted */

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
    OV_EDGE_STREAM_READ_BY_PROC,
    OV_EDGE_TYPE_COUNT,
} ov_edge_type_t;

/* Where an edge was seen (OV_EDGE.evidence); one edge may be seen several ways */
#define OV_EV_PROCTRACE 0x01     /* stream proctrace: writer PID, trigger stream inode */
#define OV_EV_SEM_READ 0x02      /* semReadPID: a process waits on the stream */
#define OV_EV_SEM_WRITE 0x04     /* semWritePID: a process posted the stream */
#define OV_EV_PINFO_TRIGGER 0x08 /* processinfo trigger stream name */
#define OV_EV_FPS_PARAM 0x10     /* FPS stream-type parameter */
#define OV_EV_FPS_RUNPID 0x20    /* FPS run PID */

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
int  pid_read_perf_counters(pid_t pid, int64_t loopcnt, ov_perf_counters_t *out);
void pid_perf_close(void);

/* Why a listed *.im.shm cannot be used */
typedef enum
{
    OV_SERR_NONE = 0,
    OV_SERR_OPEN,     /* cannot open (permission, vanished) */
    OV_SERR_FILE,     /* not a regular file, or too small for a stream */
    OV_SERR_VERSION,  /* written by another ImageStreamIO version */
    OV_SERR_DAMAGED,  /* sizes inconsistent with the file length, not shared */
    OV_SERR_DANGLING, /* symbolic link to nothing */
} ov_stream_err_t;

/* One hop of a stream's process trace: who wrote it, triggered by which stream */
typedef struct
{
    pid_t    pid;
    ino_t    inode; /**< trigger stream inode */
    int      trigmode;
    int      status;
    int64_t  tstart_ns;  /**< hop triggered (CLOCK_MILK ns), 0 = unknown */
    int64_t  tupdate_ns; /**< hop wrote its output (CLOCK_MILK ns) */
    uint64_t cnt0;       /**< the hop's trigger stream cnt0 at that write */
} OV_TRACE_HOP;

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
    double   update_hz;      /**< cnt0 rate averaged over the last block of at least 1 s */
    double   update_hz_inst; /**< cnt0 rate since the previous scan */
    int      cnt_active;     /**< cnt0 changed since last scan */
    int      cnt_flash;      /**< cnt0 changed within the last second (COUNT highlight) */

    /* ownership */
    pid_t creatorPID;
    pid_t ownerPID;
    ino_t inode;

    /* semaphores */
    int   nb_sem;
    int   semval[10];
    pid_t sem_read_pid[10]; /**< semReadPID per semaphore, aligned with semval */

    /* Write / read PIDs */
    pid_t write_pid; /**< writer PID (from proc trace) */
    int   nb_read_pids;
    pid_t read_pids[IMAGE_NB_SEMAPHORE];
    int   nb_write_pids; /**< live processes that posted the semaphores (semWritePID), unique */
    pid_t write_pids[IMAGE_NB_SEMAPHORE];

    /* process trace (STREAM_PROC_TRACE): hops trace_off .. trace_off + nb_proctrace - 1 of OV_MODEL.trace */
    int nb_proctrace;
    int trace_off;

    int    pending; /**< listed, not mapped yet (OV_MAP_PER_SCAN): only name and inode are set */
    int8_t err; /**< ov_stream_err_t: the file is listed but cannot be used (name and inode only) */
    int8_t is_link;     /**< the *.im.shm is a symbolic link to link_target */
    int8_t alias;       /**< a link whose target is listed too: shown, kept out of the graph */
    int16_t nb_holders; /**< processes mapping the file at the last HOLDERS pass, -1 = unknown */
    char    link_target[STRINGMAXLEN_IMAGE_NAME];

    /* static string cache */
    char size_str[32];

    /* graph node index (-1 if not in graph) */
    int node_idx;

    /* loop membership */
    uint32_t loop_mask;
    int      nb_loops;
    int      primary_loop_id;

    /* new-item flash counter (scans remaining) */
    int is_new;
} OV_STREAM;


/* Everything known about the one stream under inspection (F3 inspector, DETAILS): ov_scan_set_inspect */
#define OV_INSP_MAX_KW 64
#define OV_MAX_HOLDERS 16 /* processes listed per stream as mapping it */
typedef struct
{
    char            name[STRINGMAXLEN_IMAGE_NAME]; /* "" = none */
    int             valid;                         /* mapped and filled */
    char            version[32];
    uint64_t        imagetype, flag, memsize;
    int8_t          location; /* -1 CPU, >= 0 GPU device */
    uint8_t         shared, status, logflag, write;
    struct timespec creationtime, lastaccesstime, atime, writetime;
    uint64_t        cnt0, cnt1, cnt2;
    uint32_t        cb_size, cb_index;
    uint64_t        cb_cycle;
    int             nb_sem;
    int             semval[IMAGE_NB_SEMAPHORE];
    pid_t           sem_read[IMAGE_NB_SEMAPHORE], sem_write[IMAGE_NB_SEMAPHORE];
    uint8_t         sem_read_alive[IMAGE_NB_SEMAPHORE], sem_write_alive[IMAGE_NB_SEMAPHORE];
    int             nb_kw, nb_kw_listed; /* keywords in the stream, copied below */
    int             nb_holders; /* processes mapping the file at the last pass, -1 = unknown */
    struct
    {
        pid_t pid;
        char  comm[16];
    } holder[OV_MAX_HOLDERS];
    struct
    {
        char name[KEYWORD_MAX_STRING];
        char type; /* L long, D double, S string */
        char value[32];
        char comment[KEYWORD_MAX_COMMENT];
    } kw[OV_INSP_MAX_KW];
} OV_STREAM_DETAIL;

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

    int64_t  mem_rss_kb;
    int      conf_alive;
    int      run_alive;
    uint64_t conf_start_ticks; /* /proc start times: identity for signals */
    uint64_t run_start_ticks;

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

    /* new-item flash counter (scans remaining) */
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
    char     name[40];
    pid_t    PID;
    long     pindex;      /**< processinfo registry slot */
    double   createtime;  /**< registry createtime[pindex] */
    uint64_t start_ticks; /**< /proc start time when first seen alive: identity for signals */
    int      valid;
    int      active;

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

    /* new-item flash counter (scans remaining) */
    int is_new;

    /* loop membership */
    uint32_t loop_mask;
    int      nb_loops;
    int      primary_loop_id;

    /* status message / log */
    char statusmsg[200];
} OV_PROC;

#endif /* OVERVIEW_DATA_TYPES_H */
