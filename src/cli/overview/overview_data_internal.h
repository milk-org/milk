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
int fps_disconnect(FPS *fps);

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
    FPS fps;
    int  in_use;

    int  sparam_idx[OV_FPS_MAX_STREAM_PARAMS];
    char sparam_key[OV_FPS_MAX_STREAM_PARAMS][FUNCTION_PARAMETER_STRMAXLEN];
    int  sparam_nb;
    
    int  dparam_idx[OV_FPS_MAX_DISP_PARAMS];
    char dparam_key[OV_FPS_MAX_DISP_PARAMS][FUNCTION_PARAMETER_STRMAXLEN];
    int  dparam_nb;
    
    int  sparam_cached;
} ov_fps_cache_t;

extern ov_fps_cache_t s_fcache[OV_MAX_FPS];
extern int            s_fcache_nb;

typedef struct
{
    pid_t         pid;
    PROCESSINFO  *pinfo;
    int           fd;
    int           in_use;

    uint64_t prev_utime;
    uint64_t prev_stime;
    int           has_prev_cpu;
    float         cpu_pct;

    int64_t       prev_loopcnt;
    int           has_prev_loop;
} ov_proc_cache_t;

extern ov_proc_cache_t s_pcache[OV_MAX_PROCS];
extern int             s_pcache_nb;
extern double s_scan_dt_sec;

// Function declarations
void pid_cache_reset(void);
int pid_check_zombie(pid_t pid);
int64_t pid_get_rss_kb(pid_t pid);
int pid_is_alive(pid_t pid);
int pid_get_cpu_ticks(pid_t pid, uint64_t *utime, uint64_t *stime);
const char *ov_datatype_name(uint8_t dt);

int scache_find(const char *name);
void scache_evict(int ci);
int fcache_find(const char *name);
void fcache_evict(int ci);
int pcache_find_pid(pid_t pid);
void pcache_evict(int ci);

/** Get cached FPS pointer for direct parameter access */
FPS *ov_fcache_get_fps(const char *name);

/** Get raw parameter index by display index */
int ov_fcache_get_param_index(
    const char *fps_name,
    int         disp_idx);
/** Post-scan enrichment: sparklines, uptime, stale, new-item */
void ov_post_scan_enrich(OV_MODEL *model);

#endif // OVERVIEW_DATA_INTERNAL_H
