// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef OVERVIEW_DATA_INTERNAL_H
#define OVERVIEW_DATA_INTERNAL_H

#define OV_FILE_RECHECK_S 10.0 /* period (s) of the per-file check of cached FPS mappings */

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
#include <pthread.h>
#include "overview_defs.h"
#include "overview_data.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include "fps_shmdirname.h"
#include "processinfo.h"
#include "processinfo_procdirname.h"

long fps_connect(const char *name, FPS *fps, int fpsconnectmode);
int  fps_disconnect(FPS *fps);

#define OV_SHMDIR_MAXLEN STRINGMAXLEN_FPS_DIRNAME

/* A read-only (or on-demand read-write) mapping of a whole shm file, with its identity */
typedef struct
{
    void  *addr;
    size_t len;
    dev_t  dev;
    ino_t  ino;
} ov_shm_map_t;

// Cache structures
typedef struct
{
    char     name[STRINGMAXLEN_IMAGE_NAME];
    ino_t    inode;
    IMAGE    img;
    int      in_use;
    uint64_t prev_cnt0;
    int      has_prev;
    int      is_new;       /* scans left to highlight a new stream */
    int      mapped;       /* img is mapped (else listed only, waiting for OV_MAP_PER_SCAN) */
    char     size_str[32]; /* "WxH": the geometry of a mapping does not change */
    int      err; /* ov_stream_err_t of the last mapping attempt; retried when the file changes */
    struct timespec err_mtime; /* the file's mtime when it failed */
    int             is_link, alias;
    char            link_target[STRINGMAXLEN_IMAGE_NAME];
} ov_stream_cache_t;

extern ov_stream_cache_t s_scache[OV_MAX_STREAMS];

struct dirent;
int        ov_stream_dirent(const char          *dir,
                            const struct dirent *ep,
                            ino_t               *ino,
                            int                 *is_link,
                            char                *target,
                            size_t               n);
int        ov_stream_map_err(int rc, int version_ok);
extern int s_scache_nb;

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

    ov_shm_map_t map;         /* read-only mapping of <fpsdir>/<fname>.fps.shm */
    pid_t        tk_conf_pid; /* start ticks of the conf/run PIDs, refreshed when a PID changes */
    pid_t        tk_run_pid;
    uint64_t     tk_conf;
    uint64_t     tk_run;
    long         nbparam_fit; /* parameters that fit in the mapped length */

    float hz_hist[OV_SPARKLINE_LEN]; /* run-process Hz history, kept across scans */
    int   hz_hist_idx;
    int   is_new;
} ov_fps_cache_t;

extern ov_fps_cache_t s_fcache[OV_MAX_FPS];
extern int            s_fcache_nb;


int  ov_shm_map(ov_shm_map_t *m, const char *path, int rw, size_t minsize);
void ov_shm_unmap(ov_shm_map_t *m);
int  ov_shm_same_file(const ov_shm_map_t *m, const struct stat *st);

extern ov_shm_map_t s_plist_map;

typedef struct
{
    pid_t              pid;
    long               pindex; /* registry slot */
    const PROCESSINFO *pinfo;  /* NULL: file absent or unmappable */
    ov_shm_map_t       map;
    int                seen_dead; /* once dead, a reused PID never revives this entry */
    int                in_use;
    int                stale_count; /* consecutive scans active without a loopcnt change */
    uint64_t           start_ticks; /* /proc start time of the process when first seen alive */
    int                is_new;

    uint64_t prev_utime;
    uint64_t prev_stime;
    int      has_prev_cpu;
    float    cpu_pct;

    int64_t prev_loopcnt;
    int     has_prev_loop;

    int64_t start_time;
    int     has_start_time;
} ov_proc_cache_t;

extern pthread_mutex_t s_scache_mutex;
extern pthread_mutex_t s_fcache_mutex;
void                   fcache_build_params(ov_fps_cache_t *ce);

extern ov_proc_cache_t s_pcache[OV_MAX_PROCS];
extern int             s_pcache_nb;
extern double          s_scan_dt_sec;
extern struct timespec s_shm_mtime;
extern struct timespec s_fps_mtime;
extern struct timespec s_proc_mtime;

// Function declarations
void        pid_cache_reset(void);
const char *pid_stat_fields(pid_t pid, char *buf, size_t sz);
const char *pid_stat_skip(const char *p, int n);
uint64_t    pid_get_start_ticks(pid_t pid);
int         pid_same_process(pid_t pid, uint64_t ticks);
int         pid_check_zombie(pid_t pid);
int64_t     pid_get_rss_kb(pid_t pid);
int         pid_is_alive(pid_t pid);
int         pid_get_cpu_ticks(pid_t pid, uint64_t *utime, uint64_t *stime);
const char *ov_datatype_name(uint8_t dt);

int  scache_find(const char *name);
void scache_evict(int ci);
void scache_evict_locked(int ci);
void scache_evict_by_name(const char *name);
int  fcache_find(const char *name);
void fcache_evict(int ci);
void fcache_evict_locked(int ci);
int  pcache_find(long pindex, pid_t pid);
void pcache_evict(int ci);

typedef struct
{
    char     keyword[FUNCTION_PARAMETER_STRMAXLEN];
    char     display_kw[FUNCTION_PARAMETER_STRMAXLEN];
    char     valstr[FUNCTION_PARAMETER_STRMAXLEN];
    uint32_t type;
    uint64_t fpflag;
    int      is_writable;
    int      has_min, has_max; /* FPFLAG_MINLIMIT / MAXLIMIT on a numeric type */
    double   vmin, vmax;
} ov_fps_param_info_t;

/* Parameter write refusals, beyond -1 (not found / value rejected by libfps) */
#define OV_PARAM_MOVED -2   /* FPS re-created or parameter moved since it was shown */
#define OV_PARAM_RO_FILE -3 /* shm file not writable by this user */
#define OV_PARAM_LOCKED -4  /* WRITESTATUS clear: not writable in the FPS's current state */

/** Fetch parameter metadata safely under cache lock */
int ov_fcache_get_param_info(const char *fps_name, int disp_idx, ov_fps_param_info_t *info);

/** Toggle an ONOFF parameter under cache lock */
int ov_fcache_toggle_param(const char *fps_name,
                           int         disp_idx,
                           char       *out_keyword,
                           size_t      kw_size,
                           int        *out_newval);

/** Set an FPS parameter value string under cache lock */
int ov_fcache_set_param_value(const char *fps_name,
                              int         disp_idx,
                              const char *kw,
                              const char *valstr);

/** Post-scan enrichment: sparklines, uptime, stale, new-item */
void ov_post_scan_enrich(OV_MODEL *model);

/** System metrics and telemetry */
double ov_sys_get_cpu_usage(void);

/** Ordering and rank sorting */
void ov_sort_index(const void *base,
                   size_t      elsz,
                   int         n,
                   int (*cmp)(const void *, const void *),
                   int (*tie)(const void *, const void *),
                   int *order);

#endif // OVERVIEW_DATA_INTERNAL_H
