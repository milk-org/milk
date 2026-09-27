// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef OVERVIEW_DATA_INTERNAL_H
#define OVERVIEW_DATA_INTERNAL_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include "overview_defs.h"
#include "overview_data.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "fps_shmdirname.h"

long fps_connect(const char *name, FPS *fps, int fpsconnectmode);
int  fps_disconnect(FPS *fps);

#define OV_SHMDIR_MAXLEN STRINGMAXLEN_FPS_DIRNAME

// Cache structures
typedef struct
{
    char     name[STRINGMAXLEN_IMAGE_NAME];
    ino_t    inode;
    IMAGE    img;
    int      in_use;
    uint64_t prev_cnt0;
    int      has_prev;
    float    spark_max;
    float    spark_rate[OV_SPARKLINE_LEN];
    int      spark_idx;
} ov_stream_cache_t;

extern ov_stream_cache_t s_scache[OV_MAX_STREAMS];
extern int               s_scache_nb;

typedef struct
{
    char fname[STRINGMAXLEN_FPS_NAME];
    FPS  fps;
    int  in_use;

    int  sparam_idx[OV_FPS_MAX_STREAM_PARAMS];
    char sparam_key[OV_FPS_MAX_STREAM_PARAMS][FUNCTION_PARAMETER_STRMAXLEN];
    int  sparam_nb;

    int  dparam_idx[OV_FPS_MAX_DISP_PARAMS];
    char dparam_key[OV_FPS_MAX_DISP_PARAMS][FUNCTION_PARAMETER_STRMAXLEN];
    int  dparam_nb;

    int sparam_cached;
} ov_fps_cache_t;

extern ov_fps_cache_t s_fcache[OV_MAX_FPS];
extern int            s_fcache_nb;

typedef struct
{
    pid_t        pid;
    PROCESSINFO *pinfo;
    int          fd;
    int          in_use;

    uint64_t prev_utime;
    uint64_t prev_stime;
    int      has_prev_cpu;
    float    cpu_pct;

    int64_t prev_loopcnt;
    int     has_prev_loop;

    int64_t start_time;
    int     has_start_time;
} ov_proc_cache_t;

extern pthread_mutex_t s_fcache_mutex;
void                   fcache_build_params(ov_fps_cache_t *ce);

extern ov_proc_cache_t s_pcache[OV_MAX_PROCS];
extern int             s_pcache_nb;
extern double          s_scan_dt_sec;

// Function declarations
void        pid_cache_reset(void);
int         pid_check_zombie(pid_t pid);
int64_t     pid_get_rss_kb(pid_t pid);
int         pid_is_alive(pid_t pid);
int         pid_get_cpu_ticks(pid_t pid, uint64_t *utime, uint64_t *stime);
const char *ov_datatype_name(uint8_t dt);

int  scache_find(const char *name);
void scache_evict(int ci);
int  fcache_find(const char *name);
void fcache_evict(int ci);
void fcache_evict_locked(int ci);
int  pcache_find_pid(pid_t pid);
void pcache_evict(int ci);

typedef struct
{
    char     keyword[FUNCTION_PARAMETER_STRMAXLEN];
    char     display_kw[FUNCTION_PARAMETER_STRMAXLEN];
    char     valstr[FUNCTION_PARAMETER_STRMAXLEN];
    uint32_t type;
    uint64_t fpflag;
    int      is_writable;
} ov_fps_param_info_t;

/** Fetch parameter metadata safely under cache lock */
int ov_fcache_get_param_info(const char *fps_name, int disp_idx, ov_fps_param_info_t *info);

/** Toggle an ONOFF parameter under cache lock */
int ov_fcache_toggle_param(const char *fps_name,
                           int         disp_idx,
                           char       *out_keyword,
                           size_t      kw_size,
                           int        *out_newval);

/** Set an FPS parameter value string under cache lock */
int ov_fcache_set_param_value(const char *fps_name, int disp_idx, const char *valstr);

/** Post-scan enrichment: sparklines, uptime, stale, new-item */
void ov_post_scan_enrich(OV_MODEL *model);

/** System metrics and telemetry */
double ov_sys_get_cpu_usage(void);
double ov_sys_get_bandwidth_usage(void);

/** Ordering and rank sorting */
void ov_sort_freeze_snapshot(const OV_MODEL *mm);
void ov_sort_apply_ranks(OV_MODEL *mm);

#endif // OVERVIEW_DATA_INTERNAL_H
