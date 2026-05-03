/**
 * @file overview_data.c
 * @brief Scan and graph-building for milkCTRL
 *
 * Implements the three scanners (streams, FPS, processes)
 * and the cross-referencing graph builder.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>

#include "overview_defs.h"
#include "overview_data.h"

#include "ImageStreamIO/ImageStreamIO.h"

/* fps_shmdirname.h for function_parameter_struct_shmdirname */
#include "fps_shmdirname.h"

/* Forward-declare FPS connect/disconnect to avoid
 * pulling in fps.h (which conflicts with our defs) */
long function_parameter_struct_connect(
    const char *name,
    FUNCTION_PARAMETER_STRUCT *fps,
    int fpsconnectmode);
int function_parameter_struct_disconnect(
    FUNCTION_PARAMETER_STRUCT *fps);

/* STRINGMAXLEN_FPS_DIRNAME defined in fps_types.h */
#define OV_SHMDIR_MAXLEN STRINGMAXLEN_FPS_DIRNAME


/* =========================================================
 * Helpers
 * ========================================================= */

/**
 * pid_get_status - check PID liveness and zombie state.
 *
 * Uses a per-tick cache to avoid redundant syscalls
 * when many streams share the same PID.
 *
 * Returns OV_PID_DEAD, OV_PID_ALIVE, or OV_PID_ZOMBIE.
 */

#define PID_CACHE_MAX 512

static struct
{
    pid_t          pid;
    ov_pid_status_t status;
} s_pid_cache[PID_CACHE_MAX];

static int s_pid_cache_nb = 0;

/**
 * pid_cache_reset - clear PID cache.
 *
 * Must be called once at the start of each scan tick.
 */
static void pid_cache_reset(void)
{
    s_pid_cache_nb = 0;
}

/**
 * pid_check_zombie - test if PID is zombie via /proc.
 *
 * Reads the third field of /proc/<pid>/stat. If it is
 * 'Z', the process is a zombie.
 */
static int pid_check_zombie(pid_t pid)
{
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/stat",
             (int) pid);
    FILE *fp = fopen(path, "r");
    if (fp == NULL)
    {
        return 0;
    }
    /* Skip PID and (comm), then read state */
    char state = '\0';
    if (fscanf(fp, "%*d %*s %c", &state) != 1)
    {
        state = '\0';
    }
    fclose(fp);
    return (state == 'Z');
}

ov_pid_status_t pid_get_status(pid_t pid)
{
    if (pid <= 0)
    {
        return OV_PID_DEAD;
    }

    /* Search cache */
    for (int i = 0; i < s_pid_cache_nb; i++)
    {
        if (s_pid_cache[i].pid == pid)
        {
            return s_pid_cache[i].status;
        }
    }

    /* Cache miss — do the syscall(s) */
    ov_pid_status_t st = OV_PID_DEAD;
    if (kill(pid, 0) == 0)
    {
        st = pid_check_zombie(pid)
             ? OV_PID_ZOMBIE : OV_PID_ALIVE;
    }

    if (s_pid_cache_nb < PID_CACHE_MAX)
    {
        s_pid_cache[s_pid_cache_nb].pid    = pid;
        s_pid_cache[s_pid_cache_nb].status = st;
        s_pid_cache_nb++;
    }

    return st;
}

/**
 * pid_is_alive - convenience wrapper (backward compat).
 */
static int pid_is_alive(pid_t pid)
{
    return (pid_get_status(pid) != OV_PID_DEAD);
}


/**
 * ov_datatype_name - short name for data type code.
 */
static const char *ov_datatype_name(uint8_t dt)
{
    switch (dt)
    {
    case _DATATYPE_UINT8:
        return "UI8";
    case _DATATYPE_INT8:
        return "SI8";
    case _DATATYPE_UINT16:
        return "U16";
    case _DATATYPE_INT16:
        return "S16";
    case _DATATYPE_UINT32:
        return "U32";
    case _DATATYPE_INT32:
        return "S32";
    case _DATATYPE_UINT64:
        return "U64";
    case _DATATYPE_INT64:
        return "S64";
    case _DATATYPE_FLOAT:
        return "F32";
    case _DATATYPE_DOUBLE:
        return "F64";
    case _DATATYPE_COMPLEX_FLOAT:
        return "CF";
    case _DATATYPE_COMPLEX_DOUBLE:
        return "CD";
    default:
        return "???";
    }
}
/* suppress unused-function warning when header is
 * included but ov_datatype_name is only used by
 * the render code */
__attribute__((unused))
static const char *ov_datatype_name_ref =
    (const char *)(uintptr_t) ov_datatype_name;


/* =========================================================
 * Persistent SHM mapping caches
 *
 * These caches keep SHM file descriptors and mappings
 * alive across scan ticks to avoid the overhead of
 * open/mmap/munmap/close on every cycle.
 *
 * Thread safety: the cache is accessed only by the
 * background scan thread, so no locking is required.
 *
 * Staleness: each tick checks stat() inode; a mismatch
 * or missing file triggers remap/eviction.
 * ========================================================= */

/* --- Stream cache --- */

typedef struct
{
    char     name[STRINGMAXLEN_IMAGE_NAME];
    ino_t    inode;
    IMAGE    img;       /**< persistent mmap'd IMAGE */
    int      in_use;    /**< set each tick; cleared for eviction */
    uint64_t prev_cnt0; /**< cnt0 from previous scan tick */
    int      has_prev;  /**< 1 once prev_cnt0 is valid */
    float    spark_rate[OV_SPARKLINE_LEN];
    int      spark_idx; /**< ring index into spark_rate */
} ov_stream_cache_t;

static ov_stream_cache_t s_scache[OV_MAX_STREAMS];
static int               s_scache_nb = 0;

/**
 * scache_find - find stream in cache by name.
 *
 * Return: cache index, or -1 if not found.
 */
static int scache_find(const char *name)
{
    for (int i = 0; i < s_scache_nb; i++)
    {
        if (strcmp(s_scache[i].name, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

/**
 * scache_evict - close mapping and compact array.
 */
static void scache_evict(int ci)
{
    ImageStreamIO_closeIm(&s_scache[ci].img);
    s_scache_nb--;
    if (ci < s_scache_nb)
    {
        s_scache[ci] = s_scache[s_scache_nb];
    }
}

/* --- FPS cache --- */

typedef struct
{
    char fname[STRINGMAXLEN_FPS_NAME];
    FUNCTION_PARAMETER_STRUCT fps;
    int  in_use;

    /* Cached stream-type parameter indices.
     * Built once on first fill; avoids scanning
     * all ~10K params on every tick. */
    int  sparam_idx[OV_FPS_MAX_STREAM_PARAMS];
    char sparam_key[OV_FPS_MAX_STREAM_PARAMS]
                   [FUNCTION_PARAMETER_STRMAXLEN];
    int  sparam_nb;
    int  sparam_cached; /**< 1 once index scan done */
} ov_fps_cache_t;

static ov_fps_cache_t s_fcache[OV_MAX_FPS];
static int            s_fcache_nb = 0;

static int fcache_find(const char *name)
{
    for (int i = 0; i < s_fcache_nb; i++)
    {
        if (strcmp(s_fcache[i].fname, name) == 0)
        {
            return i;
        }
    }
    return -1;
}

static void fcache_evict(int ci)
{
    function_parameter_struct_disconnect(
        &s_fcache[ci].fps);
    s_fcache_nb--;
    if (ci < s_fcache_nb)
    {
        s_fcache[ci] = s_fcache[s_fcache_nb];
    }
}

/* --- Proc cache --- */

typedef struct
{
    pid_t        pid;
    PROCESSINFO *pinfo;   /**< mmap'd pointer */
    int          fd;
    int          in_use;
} ov_proc_cache_t;

static ov_proc_cache_t s_pcache[OV_MAX_PROCS];
static int             s_pcache_nb = 0;

/* Processinfo list mapping (single, persistent) */
static PROCESSINFOLIST *s_pilist     = NULL;
static int              s_pilist_fd  = -1;

static int pcache_find_pid(pid_t pid)
{
    for (int i = 0; i < s_pcache_nb; i++)
    {
        if (s_pcache[i].pid == pid)
        {
            return i;
        }
    }
    return -1;
}

static void pcache_evict(int ci)
{
    munmap(s_pcache[ci].pinfo, sizeof(PROCESSINFO));
    close(s_pcache[ci].fd);
    s_pcache_nb--;
    if (ci < s_pcache_nb)
    {
        s_pcache[ci] = s_pcache[s_pcache_nb];
    }
}


/* =========================================================
 * Stream scanning
 * ========================================================= */

/* Directory mtime for readdir skip optimization */
static struct timespec s_shm_mtime = {0, 0};

/* Scan-tick time delta (set by ov_model_full_scan
 * before calling scan functions) */
static double s_scan_dt_sec = 0.0;


/**
 * scache_rate_update - compute Hz and sparkline for
 *     a stream using cache's prev_cnt0.
 * @s:   the stream entry (already filled)
 * @ci:  cache index
 */
static void scache_rate_update(
    OV_STREAM *s,
    int        ci)
{
    ov_stream_cache_t *ce = &s_scache[ci];

    if (ce->has_prev && s_scan_dt_sec > 0.01)
    {
        uint64_t dc = s->cnt0 - ce->prev_cnt0;
        s->update_hz =
            (double) dc / s_scan_dt_sec;

        /* Update sparkline in cache */
        float sv   = (float) s->update_hz;
        float norm = sv / 10000.0f;
        if (norm > 1.0f)
        {
            norm = 1.0f;
        }
        ce->spark_rate[
            ce->spark_idx % OV_SPARKLINE_LEN]
            = norm;
        ce->spark_idx++;
    }

    /* Store current cnt0 for next tick */
    ce->prev_cnt0 = s->cnt0;
    ce->has_prev  = 1;

    /* Copy sparkline from cache to model */
    memcpy(s->spark_rate, ce->spark_rate,
           sizeof(s->spark_rate));
    s->spark_idx = ce->spark_idx;
}

/**
 * fill_stream_from_img - populate OV_STREAM from
 *     a cached IMAGE mapping.
 * @s:     stream entry to fill
 * @imgp:  persistent IMAGE pointer
 * @name:  stream name
 * @inode: inode value
 */
static void fill_stream_from_img(
    OV_STREAM   *s,
    IMAGE       *imgp,
    const char  *name,
    ino_t        inode)
{
    memset(s, 0, sizeof(OV_STREAM));
    strncpy(s->name, name,
            sizeof(s->name) - 1);
    s->valid = 1;
    s->inode = inode;

    if (imgp->md == NULL)
    {
        s->node_idx = -1;
        return;
    }

    s->datatype   = imgp->md->datatype;
    s->naxis      = imgp->md->naxis;
    s->size[0]    = imgp->md->size[0];
    s->size[1]    = imgp->md->size[1];
    s->size[2]    = imgp->md->size[2];
    s->nelement   = imgp->md->nelement;
    s->creatorPID = imgp->md->creatorPID;
    s->ownerPID   = imgp->md->ownerPID;
    s->cnt0       = imgp->md->cnt0;

    /* Process trace entries */
    int npt = imgp->md->NBproctrace;
    if (npt > IMAGE_NB_PROCTRACE)
    {
        npt = IMAGE_NB_PROCTRACE;
    }
    s->nb_proctrace = 0;

    if (imgp->streamproctrace != NULL)
    {
        for (int t = 0; t < npt; t++)
        {
            STREAM_PROC_TRACE *spt =
                &imgp->streamproctrace[t];
            if (spt->procwrite_PID > 0)
            {
                int ti = s->nb_proctrace;
                s->proctrace_pid[ti] =
                    spt->procwrite_PID;
                s->proctrace_inode[ti] =
                    spt->trigger_inode;
                s->proctrace_trigmode[ti] =
                    spt->triggermode;
                s->proctrace_status[ti] =
                    spt->triggerstatus;
                s->nb_proctrace++;
            }
        }
    }

    s->active = pid_is_alive(s->ownerPID)
                || pid_is_alive(s->creatorPID);

    s->nb_sem = imgp->md->sem;
    if (s->nb_sem > 10)
    {
        s->nb_sem = 10;
    }
    for (int sm = 0; sm < s->nb_sem; sm++)
    {
        s->semval[sm] =
            ImageStreamIO_semvalue(imgp, sm);
    }

    /* Writer PID: first active proc trace entry */
    s->write_pid = 0;
    if (s->nb_proctrace > 0)
    {
        s->write_pid = s->proctrace_pid[0];
    }

    /* Reader PIDs from semReadPID array */
    s->nb_read_pids = 0;
    if (imgp->semReadPID != NULL)
    {
        for (int sm = 0;
             sm < s->nb_sem
             && s->nb_read_pids < IMAGE_NB_SEMAPHORE;
             sm++)
        {
            pid_t rpid = imgp->semReadPID[sm];
            if (rpid > 0 && pid_is_alive(rpid))
            {
                /* Avoid duplicates */
                int dup = 0;
                for (int k = 0;
                     k < s->nb_read_pids; k++)
                {
                    if (s->read_pids[k] == rpid)
                    {
                        dup = 1;
                        break;
                    }
                }
                if (!dup)
                {
                    s->read_pids[
                        s->nb_read_pids] = rpid;
                    s->nb_read_pids++;
                }
            }
        }
    }

    s->node_idx = -1;
}

void ov_scan_streams(OV_MODEL *model)
{
    const char *shmdir = SHAREDSHMDIR;

    /* Check directory mtime to skip readdir */
    struct stat dirstat;
    if (stat(shmdir, &dirstat) != 0)
    {
        model->nb_streams = 0;
        return;
    }

    int dir_changed =
        (dirstat.st_mtim.tv_sec
            != s_shm_mtime.tv_sec)
        || (dirstat.st_mtim.tv_nsec
            != s_shm_mtime.tv_nsec);

    if (!dir_changed && s_scache_nb > 0)
    {
        /* Fast path: no files added/removed.
         * Just re-read metadata from cache. */
        int idx = 0;
        for (int ci = 0;
             ci < s_scache_nb
             && idx < OV_MAX_STREAMS;
             ci++)
        {
            fill_stream_from_img(
                &model->streams[idx],
                &s_scache[ci].img,
                s_scache[ci].name,
                s_scache[ci].inode);
            scache_rate_update(
                &model->streams[idx], ci);
            idx++;
        }
        model->nb_streams = idx;
        return;
    }

    /* Full path: directory changed */
    s_shm_mtime = dirstat.st_mtim;

    DIR *dp = opendir(shmdir);
    if (dp == NULL)
    {
        model->nb_streams = 0;
        return;
    }

    /* Mark all cache entries as not-in-use */
    for (int i = 0; i < s_scache_nb; i++)
    {
        s_scache[i].in_use = 0;
    }

    int idx = 0;
    struct dirent *ep;

    while ((ep = readdir(dp)) != NULL
           && idx < OV_MAX_STREAMS)
    {
        /* Match *.im.shm */
        int namelen = (int) strlen(ep->d_name);
        if (namelen < 8)
        {
            continue;
        }
        if (strcmp(ep->d_name + namelen - 7,
                  ".im.shm") != 0)
        {
            continue;
        }

        /* Extract stream name */
        char sname[STRINGMAXLEN_IMAGE_NAME];
        int  snlen = namelen - 7;
        if (snlen >= (int) sizeof(sname))
        {
            snlen = (int) sizeof(sname) - 1;
        }
        memcpy(sname, ep->d_name,
               (size_t) snlen);
        sname[snlen] = '\0';

        /* stat() for inode */
        char fpath[1024];
        snprintf(fpath, sizeof(fpath),
                 "%s/%s", shmdir, ep->d_name);
        struct stat st;
        if (stat(fpath, &st) != 0)
        {
            continue;
        }

        /* Look up in cache */
        int ci = scache_find(sname);
        IMAGE *imgp = NULL;

        if (ci >= 0
            && s_scache[ci].inode == st.st_ino)
        {
            /* Cache hit — reuse mapping */
            s_scache[ci].in_use = 1;
            imgp = &s_scache[ci].img;
        }
        else
        {
            /* Cache miss or inode changed */
            if (ci >= 0)
            {
                scache_evict(ci);
                ci = -1;
            }

            if (s_scache_nb >= OV_MAX_STREAMS)
            {
                continue;
            }

            ci = s_scache_nb;
            memset(&s_scache[ci], 0,
                   sizeof(ov_stream_cache_t));

            if (ImageStreamIO_read_sharedmem_image_toIMAGE(
                    sname,
                    &s_scache[ci].img)
                != IMAGESTREAMIO_SUCCESS)
            {
                continue;
            }

            strncpy(s_scache[ci].name, sname,
                    sizeof(s_scache[ci].name)
                    - 1);
            s_scache[ci].inode  = st.st_ino;
            s_scache[ci].in_use = 1;
            s_scache_nb++;
            imgp = &s_scache[ci].img;
        }

        fill_stream_from_img(
            &model->streams[idx],
            imgp, sname, st.st_ino);
        scache_rate_update(
            &model->streams[idx], ci);
        idx++;
    }

    closedir(dp);
    model->nb_streams = idx;

    /* Evict stale cache entries */
    for (int i = s_scache_nb - 1; i >= 0; i--)
    {
        if (!s_scache[i].in_use)
        {
            scache_evict(i);
        }
    }
}


/* =========================================================
 * FPS scanning
 * ========================================================= */

/* FPS directory mtime for readdir skip */
static struct timespec s_fps_mtime = {0, 0};

/**
 * fcache_build_sparam - discover stream-type param
 *     indices and cache them. Called once per FPS.
 * @ce: FPS cache entry
 */
static void fcache_build_sparam(
    ov_fps_cache_t *ce)
{
    FUNCTION_PARAMETER_STRUCT *fpsp = &ce->fps;
    int sp = 0;

    if (fpsp->md == NULL)
    {
        ce->sparam_nb     = 0;
        ce->sparam_cached = 1;
        return;
    }

    int nb_params = fpsp->md->NBparamMAX;
    if (nb_params > 10000)
    {
        nb_params = 10000;
    }

    for (int p = 0;
         p < nb_params
         && sp < OV_FPS_MAX_STREAM_PARAMS;
         p++)
    {
        FUNCTION_PARAMETER *fp =
            &fpsp->parray[p];
        if (!(fp->fpflag & FPFLAG_ACTIVE))
        {
            continue;
        }
        if (fp->type != FPTYPE_STREAMNAME)
        {
            continue;
        }

        ce->sparam_idx[sp] = p;

        /* Build and cache keyword string */
        char *kbuf = ce->sparam_key[sp];
        kbuf[0] = '\0';
        {
            int klen = 0;
            for (int kl = 1;
                 kl < FUNCTION_PARAMETER_KEYWORD_MAXLEVEL;
                 kl++)
            {
                if (fp->keyword[kl][0] == '\0')
                {
                    break;
                }
                if (klen > 0
                    && klen < FUNCTION_PARAMETER_STRMAXLEN - 1)
                {
                    kbuf[klen++] = '.';
                    kbuf[klen]   = '\0';
                }
                int rem =
                    FUNCTION_PARAMETER_STRMAXLEN
                    - klen - 1;
                if (rem > 0)
                {
                    strncat(kbuf + klen,
                            fp->keyword[kl],
                            (size_t) rem);
                    klen = (int) strlen(kbuf);
                }
            }
            if (kbuf[0] == '\0')
            {
                strncpy(kbuf,
                        fp->keyword[0],
                        FUNCTION_PARAMETER_STRMAXLEN
                        - 1);
            }
        }
        sp++;
    }

    ce->sparam_nb     = sp;
    ce->sparam_cached = 1;
}

/**
 * fill_fps_from_struct - populate OV_FPS from
 *     a cached FPS mapping.
 * @f:   FPS entry to fill
 * @ce:  FPS cache entry (includes param index cache)
 */
static void fill_fps_from_struct(
    OV_FPS          *f,
    ov_fps_cache_t  *ce)
{
    FUNCTION_PARAMETER_STRUCT *fpsp = &ce->fps;

    memset(f, 0, sizeof(OV_FPS));
    strncpy(f->name, ce->fname,
            sizeof(f->name) - 1);
    f->valid = 1;

    if (fpsp->md == NULL)
    {
        f->node_idx = -1;
        return;
    }

    f->md_status  = fpsp->md->status;
    f->confpid    = fpsp->md->confpid;
    f->runpid     = fpsp->md->runpid;
    f->conf_alive = pid_is_alive(f->confpid);
    f->run_alive  = pid_is_alive(f->runpid);

    /* Build param index cache on first use */
    if (!ce->sparam_cached)
    {
        fcache_build_sparam(ce);
    }

    /* Read only the cached stream-type params */
    for (int sp = 0; sp < ce->sparam_nb; sp++)
    {
        FUNCTION_PARAMETER *fp =
            &fpsp->parray[ce->sparam_idx[sp]];
        strncpy(f->stream_param_name[sp],
                ce->sparam_key[sp],
                FUNCTION_PARAMETER_STRMAXLEN
                - 1);
        strncpy(f->stream_param_value[sp],
                fp->val.string[0],
                FUNCTION_PARAMETER_STRMAXLEN
                - 1);
        f->stream_param_flags[sp] = fp->fpflag;
    }
    f->nb_stream_params = ce->sparam_nb;

    /* Read description from md */
    if (fpsp->md->description[0] != '\0')
    {
        strncpy(f->description,
                fpsp->md->description,
                sizeof(f->description) - 1);
    }

    f->node_idx = -1;
}

void ov_scan_fps(OV_MODEL *model)
{
    char shmdir[OV_SHMDIR_MAXLEN];
    function_parameter_struct_shmdirname(shmdir);

    /* Check directory mtime to skip readdir */
    struct stat dirstat;
    if (stat(shmdir, &dirstat) != 0)
    {
        model->nb_fps = 0;
        return;
    }

    int dir_changed =
        (dirstat.st_mtim.tv_sec
            != s_fps_mtime.tv_sec)
        || (dirstat.st_mtim.tv_nsec
            != s_fps_mtime.tv_nsec);

    if (!dir_changed && s_fcache_nb > 0)
    {
        /* Fast path: no files added/removed */
        int idx = 0;
        for (int ci = 0;
             ci < s_fcache_nb
             && idx < OV_MAX_FPS;
             ci++)
        {
            fill_fps_from_struct(
                &model->fps[idx],
                &s_fcache[ci]);
            idx++;
        }
        model->nb_fps = idx;
        return;
    }

    /* Full path: directory changed */
    s_fps_mtime = dirstat.st_mtim;

    DIR *dp = opendir(shmdir);
    if (dp == NULL)
    {
        model->nb_fps = 0;
        return;
    }

    /* Mark all FPS cache entries as not-in-use */
    for (int i = 0; i < s_fcache_nb; i++)
    {
        s_fcache[i].in_use = 0;
    }

    int idx = 0;
    struct dirent *ep;

    while ((ep = readdir(dp)) != NULL
           && idx < OV_MAX_FPS)
    {
        /* Match *.fps.shm */
        int namelen = (int) strlen(ep->d_name);
        if (namelen < 9)
        {
            continue;
        }
        if (strcmp(ep->d_name + namelen - 8,
                  ".fps.shm") != 0)
        {
            continue;
        }

        /* Extract FPS name */
        char fname[STRINGMAXLEN_FPS_NAME];
        int fnlen = namelen - 8;
        if (fnlen >= (int) sizeof(fname))
        {
            fnlen = (int) sizeof(fname) - 1;
        }
        memcpy(fname, ep->d_name,
               (size_t) fnlen);
        fname[fnlen] = '\0';

        /* Look up in cache */
        int ci = fcache_find(fname);

        if (ci >= 0)
        {
            /* Cache hit */
            s_fcache[ci].in_use = 1;
        }
        else
        {
            /* Cache miss */
            if (s_fcache_nb >= OV_MAX_FPS)
            {
                continue;
            }

            ci = s_fcache_nb;
            memset(&s_fcache[ci], 0,
                   sizeof(ov_fps_cache_t));
            s_fcache[ci].fps.SMfd = -1;

            long fpsID =
                function_parameter_struct_connect(
                    fname,
                    &s_fcache[ci].fps,
                    FPSCONNECT_SIMPLE);
            if (fpsID < 0)
            {
                continue;
            }

            strncpy(s_fcache[ci].fname, fname,
                    sizeof(s_fcache[ci].fname)
                    - 1);
            s_fcache[ci].in_use = 1;
            s_fcache_nb++;
        }

        fill_fps_from_struct(
            &model->fps[idx], &s_fcache[ci]);
        idx++;
    }

    closedir(dp);
    model->nb_fps = idx;

    /* Evict stale FPS cache entries */
    for (int i = s_fcache_nb - 1; i >= 0; i--)
    {
        if (!s_fcache[i].in_use)
        {
            fcache_evict(i);
        }
    }
}


/* =========================================================
 * Process scanning
 * ========================================================= */

void ov_scan_procs(OV_MODEL *model)
{
    char shmdir[OV_SHMDIR_MAXLEN];
    function_parameter_struct_shmdirname(shmdir);

    /* Persistent processinfo list mapping */
    if (s_pilist == NULL)
    {
        char pilistfname[1024];
        snprintf(pilistfname, sizeof(pilistfname),
                 "%s/processinfo.list.shm", shmdir);

        s_pilist_fd = open(pilistfname, O_RDONLY);
        if (s_pilist_fd == -1)
        {
            model->nb_procs = 0;
            return;
        }

        s_pilist =
            (PROCESSINFOLIST *) mmap(
                NULL,
                sizeof(PROCESSINFOLIST),
                PROT_READ,
                MAP_SHARED,
                s_pilist_fd,
                0);

        if (s_pilist == MAP_FAILED)
        {
            close(s_pilist_fd);
            s_pilist    = NULL;
            s_pilist_fd = -1;
            model->nb_procs = 0;
            return;
        }
    }

    /* Mark all proc cache entries as not-in-use */
    for (int i = 0; i < s_pcache_nb; i++)
    {
        s_pcache[i].in_use = 0;
    }

    int idx = 0;

    /* Track high-water mark of active indices to
     * avoid scanning all 50K entries every tick.
     * Margin of 256 catches newly-added entries. */
    static int s_pilist_hwm = 256;
    int scan_limit = s_pilist_hwm + 256;
    if (scan_limit > PROCESSINFOLISTSIZE)
    {
        scan_limit = PROCESSINFOLISTSIZE;
    }

    int hwm_this_tick = 0;

    for (int i = 0;
         i < scan_limit
         && idx < OV_MAX_PROCS;
         i++)
    {
        if (s_pilist->active[i] == 0)
        {
            continue;
        }

        hwm_this_tick = i;

        pid_t pid = s_pilist->PIDarray[i];
        if (!pid_is_alive(pid))
        {
            continue;
        }

        /* Look up in proc cache */
        int ci = pcache_find_pid(pid);
        PROCESSINFO *pinfo = NULL;

        if (ci >= 0)
        {
            /* Cache hit */
            s_pcache[ci].in_use = 1;
            pinfo = s_pcache[ci].pinfo;
        }
        else
        {
            /* Cache miss — map new */
            if (s_pcache_nb >= OV_MAX_PROCS)
            {
                continue;
            }

            char pinfofname[1024];
            snprintf(pinfofname, sizeof(pinfofname),
                     "%s/proc.%s.%06d.shm",
                     shmdir,
                     s_pilist->pnamearray[i],
                     (int) pid);

            int pfd = open(pinfofname, O_RDONLY);
            if (pfd == -1)
            {
                continue;
            }

            PROCESSINFO *pm =
                (PROCESSINFO *) mmap(
                    NULL,
                    sizeof(PROCESSINFO),
                    PROT_READ,
                    MAP_SHARED,
                    pfd,
                    0);

            if (pm == MAP_FAILED)
            {
                close(pfd);
                continue;
            }

            ci = s_pcache_nb;
            s_pcache[ci].pid    = pid;
            s_pcache[ci].pinfo  = pm;
            s_pcache[ci].fd     = pfd;
            s_pcache[ci].in_use = 1;
            s_pcache_nb++;
            pinfo = pm;
        }

        OV_PROC *p = &model->procs[idx];
        memset(p, 0, sizeof(OV_PROC));
        strncpy(p->name,
                s_pilist->pnamearray[i],
                sizeof(p->name) - 1);
        p->PID    = pid;
        p->valid  = 1;
        p->active = 1;

        p->loopstat = pinfo->loopstat;
        p->CTRLval  = pinfo->CTRLval;
        p->loopcnt  = pinfo->loopcnt;

        /* Timing stats */
        p->dtmedian_iter_ns =
            pinfo->dtmedian_iter_ns;
        p->dtmedian_exec_ns =
            pinfo->dtmedian_exec_ns;
        if (p->dtmedian_iter_ns > 0)
        {
            p->loop_hz =
                1.0e9
                / (double) p->dtmedian_iter_ns;
        }

        /* Trigger info */
        strncpy(p->trigstreamname,
                pinfo->triggerstreamname,
                sizeof(p->trigstreamname) - 1);
        p->triggermode = pinfo->triggermode;
        p->triggersem  = pinfo->triggersem;
        p->triggermissed =
            pinfo->triggermissedframe;
        p->triggermissed_cumul =
            pinfo->triggermissedframe_cumul;
        p->MeasureTiming = pinfo->MeasureTiming;

        p->rt_priority = pinfo->RT_priority;

        p->node_idx = -1;
        idx++;
    }

    model->nb_procs = idx;

    /* Update high-water mark */
    if (hwm_this_tick > s_pilist_hwm)
    {
        s_pilist_hwm = hwm_this_tick;
    }

    /* Evict stale proc cache entries */
    for (int i = s_pcache_nb - 1; i >= 0; i--)
    {
        if (!s_pcache[i].in_use)
        {
            pcache_evict(i);
        }
    }
}


/* =========================================================
 * Cache cleanup (called from ov_scan_stop)
 * ========================================================= */

void ov_scan_cache_cleanup(void)
{
    /* Close all cached stream mappings */
    for (int i = s_scache_nb - 1; i >= 0; i--)
    {
        ImageStreamIO_closeIm(&s_scache[i].img);
    }
    s_scache_nb = 0;

    /* Disconnect all cached FPS mappings */
    for (int i = s_fcache_nb - 1; i >= 0; i--)
    {
        function_parameter_struct_disconnect(
            &s_fcache[i].fps);
    }
    s_fcache_nb = 0;

    /* Unmap all cached proc mappings */
    for (int i = s_pcache_nb - 1; i >= 0; i--)
    {
        munmap(s_pcache[i].pinfo,
               sizeof(PROCESSINFO));
        close(s_pcache[i].fd);
    }
    s_pcache_nb = 0;

    /* Unmap the processinfo list */
    if (s_pilist != NULL)
    {
        munmap(s_pilist, sizeof(PROCESSINFOLIST));
        close(s_pilist_fd);
        s_pilist    = NULL;
        s_pilist_fd = -1;
    }

    /* Reset directory mtime trackers */
    memset(&s_shm_mtime, 0,
           sizeof(s_shm_mtime));
    memset(&s_fps_mtime, 0,
           sizeof(s_fps_mtime));

    /* Reset PID liveness cache */
    pid_cache_reset();

    /* Reset graph and rate state */
    s_scan_dt_sec  = 0.0;
}


/* =========================================================
 * Lookup helpers
 * ========================================================= */

int ov_find_stream_by_inode(
    const OV_MODEL *model,
    ino_t inode)
{
    if (inode == 0)
    {
        return -1;
    }
    for (int i = 0; i < model->nb_streams; i++)
    {
        if (model->streams[i].valid
            && model->streams[i].inode == inode)
        {
            return i;
        }
    }
    return -1;
}

int ov_find_stream_by_name(
    const OV_MODEL *model,
    const char *name)
{
    if (name == NULL || name[0] == '\0')
    {
        return -1;
    }
    for (int i = 0; i < model->nb_streams; i++)
    {
        if (model->streams[i].valid
            && strcmp(model->streams[i].name,
                     name) == 0)
        {
            return i;
        }
    }
    return -1;
}

int ov_find_proc_by_pid(
    const OV_MODEL *model,
    pid_t pid)
{
    if (pid <= 0)
    {
        return -1;
    }
    for (int i = 0; i < model->nb_procs; i++)
    {
        if (model->procs[i].valid
            && model->procs[i].PID == pid)
        {
            return i;
        }
    }
    return -1;
}


/* =========================================================
 * Edge management
 * ========================================================= */

void ov_add_edge(
    OV_MODEL *model,
    int src,
    int tgt,
    ov_edge_type_t type,
    const char *label)
{
    if (src < 0 || tgt < 0 || src == tgt)
    {
        return;
    }
    if (model->nb_edges >= OV_MAX_EDGES)
    {
        return;
    }

    /* Check for duplicate */
    for (int i = 0; i < model->nb_edges; i++)
    {
        if (model->edges[i].src_node == src
            && model->edges[i].tgt_node == tgt
            && model->edges[i].type == type)
        {
            return;
        }
    }

    OV_EDGE *e = &model->edges[model->nb_edges];
    e->src_node = src;
    e->tgt_node = tgt;
    e->type     = type;
    e->active   = 1;
    strncpy(e->label, label,
            sizeof(e->label) - 1);
    e->label[sizeof(e->label) - 1] = '\0';
    model->nb_edges++;
}


/* =========================================================
 * Graph builder
 * ========================================================= */

/**
 * add_node - add a node to the graph.
 * @model: model
 * @type:  node type
 * @index: type-specific index
 * @name:  display name
 * @active: whether the node is active
 *
 * Return: node index in model->nodes[].
 */
static int add_node(
    OV_MODEL *model,
    ov_node_type_t type,
    int index,
    const char *name,
    int active)
{
    if (model->nb_nodes >= OV_MAX_NODES)
    {
        return -1;
    }
    int ni = model->nb_nodes;
    OV_NODE *n = &model->nodes[ni];
    n->type   = type;
    n->index  = index;
    n->active = active;
    n->gx     = 0;
    n->gy     = 0;
    strncpy(n->name, name,
            sizeof(n->name) - 1);
    n->name[sizeof(n->name) - 1] = '\0';
    model->nb_nodes++;
    return ni;
}


void ov_build_graph(OV_MODEL *model)
{
    model->nb_nodes = 0;
    model->nb_edges = 0;

    /* ---- Create nodes for all entities ---- */

    for (int i = 0; i < model->nb_streams; i++)
    {
        if (!model->streams[i].valid)
        {
            continue;
        }
        model->streams[i].node_idx = add_node(
                                         model, OV_NODE_STREAM, i,
                                         model->streams[i].name,
                                         model->streams[i].active);
    }

    for (int i = 0; i < model->nb_fps; i++)
    {
        if (!model->fps[i].valid)
        {
            continue;
        }
        model->fps[i].node_idx = add_node(
                                     model, OV_NODE_FPS, i,
                                     model->fps[i].name,
                                     model->fps[i].run_alive);
    }

    for (int i = 0; i < model->nb_procs; i++)
    {
        if (!model->procs[i].valid)
        {
            continue;
        }
        model->procs[i].node_idx = add_node(
                                       model, OV_NODE_PROC, i,
                                       model->procs[i].name,
                                       model->procs[i].active);
    }

    /* ---- Build edges from process trace ---- */

    for (int si = 0; si < model->nb_streams; si++)
    {
        OV_STREAM *s = &model->streams[si];
        if (!s->valid || s->node_idx < 0)
        {
            continue;
        }

        for (int t = 0; t < s->nb_proctrace; t++)
        {
            pid_t wpid = s->proctrace_pid[t];
            if (wpid <= 0)
            {
                continue;
            }

            /* Find the process that writes
             * this stream */
            int pi = ov_find_proc_by_pid(model, wpid);
            if (pi >= 0 && model->procs[pi].node_idx >= 0)
            {
                /* proc → stream (writes) */
                ov_add_edge(
                    model,
                    model->procs[pi].node_idx,
                    s->node_idx,
                    OV_EDGE_PROC_WRITES_STREAM,
                    "writes");
            }

            /* Find the trigger stream */
            ino_t trig_inode =
                s->proctrace_inode[t];
            if (trig_inode != 0
                && pi >= 0
                && model->procs[pi].node_idx >= 0)
            {
                int tsi = ov_find_stream_by_inode(
                              model, trig_inode);
                if (tsi >= 0
                    && model->streams[tsi].node_idx
                    >= 0)
                {
                    /* stream → proc (triggers) */
                    ov_add_edge(
                        model,
                        model->streams[tsi].node_idx,
                        model->procs[pi].node_idx,
                        OV_EDGE_STREAM_TRIGGERS_PROC,
                        "triggers");
                }
            }
        }
    }

    /* ---- Build edges from FPS ---- */

    for (int fi = 0; fi < model->nb_fps; fi++)
    {
        OV_FPS *f = &model->fps[fi];
        if (!f->valid || f->node_idx < 0)
        {
            continue;
        }

        /* FPS → Process (via runpid match) */
        if (f->run_alive && f->runpid > 0)
        {
            int pi = ov_find_proc_by_pid(
                         model, f->runpid);
            if (pi >= 0
                && model->procs[pi].node_idx >= 0)
            {
                ov_add_edge(
                    model,
                    f->node_idx,
                    model->procs[pi].node_idx,
                    OV_EDGE_FPS_RUNS_PROC,
                    "runs");
            }
        }

        /* FPS stream params → Stream nodes */
        for (int sp = 0;
             sp < f->nb_stream_params;
             sp++)
        {
            const char *sval =
                f->stream_param_value[sp];
            if (sval[0] == '\0')
            {
                continue;
            }

            int si = ov_find_stream_by_name(
                         model, sval);
            if (si < 0
                || model->streams[si].node_idx < 0)
            {
                continue;
            }

            uint64_t flags =
                f->stream_param_flags[sp];
            const char *pname =
                f->stream_param_name[sp];

            int is_out = 0;
            if (flags & FPFLAG_WRITE)
            {
                is_out = 1;
            }
            else if (strstr(pname, "out") || strstr(pname, "OUT"))
            {
                is_out = 1;
            }

            if (is_out)
            {
                /* FPS → stream (output) */
                ov_add_edge(
                    model,
                    f->node_idx,
                    model->streams[si].node_idx,
                    OV_EDGE_FPS_OUTPUT_STREAM,
                    "output");
            }
            else
            {
                /* stream → FPS (input) */
                ov_add_edge(
                    model,
                    model->streams[si].node_idx,
                    f->node_idx,
                    OV_EDGE_FPS_INPUT_STREAM,
                    "input");
            }
        }
    }

    /* ---- Build edges from process trigger ---- */

    for (int pi = 0; pi < model->nb_procs; pi++)
    {
        OV_PROC *p = &model->procs[pi];
        if (!p->valid || p->node_idx < 0)
        {
            continue;
        }

        if (p->trigstreamname[0] != '\0')
        {
            int si = ov_find_stream_by_name(
                         model, p->trigstreamname);
            if (si >= 0
                && model->streams[si].node_idx >= 0)
            {
                ov_add_edge(
                    model,
                    model->streams[si].node_idx,
                    p->node_idx,
                    OV_EDGE_PROC_TRIGGER_STREAM,
                    "trigger");
            }
        }
    }
}


/* =========================================================
 * Full scan
 * ========================================================= */

void ov_model_full_scan(OV_MODEL *model)
{
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    /* Reset per-tick PID liveness cache */
    pid_cache_reset();

    /* Compute time delta for rate estimation
     * (used by scache_rate_update inside scan) */
    if (model->scan_count > 0)
    {
        struct timespec dt_ts = ov_timespec_diff(
                                    model->last_scan_time, t0);
        s_scan_dt_sec =
            (double) dt_ts.tv_sec
            + (double) dt_ts.tv_nsec * 1.0e-9;
    }
    else
    {
        s_scan_dt_sec = 0.0;
    }

    /* Clear dynamic parts */
    model->nb_streams = 0;
    model->nb_fps     = 0;
    model->nb_procs   = 0;
    model->nb_nodes   = 0;
    model->nb_edges   = 0;

    ov_scan_streams(model);
    ov_scan_fps(model);
    ov_scan_procs(model);

    ov_build_graph(model);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    struct timespec dt = ov_timespec_diff(t0, t1);
    model->scan_time_ms =
        (double) dt.tv_sec * 1000.0
        + (double) dt.tv_nsec * 1.0e-6;
    model->last_scan_time = t0;
    model->scan_count++;
}


/* =========================================================
 * Sorting helpers
 *
 * Key assignments match visual column order
 * (left-to-right) so that ] cycles naturally.
 *
 * Streams: 0=NAME 1=TYP 2=SIZE 3=Hz
 *          4=INODE 5=COUNT
 * Procs:   0=NAME 1=PID 2=STAT 3=Hz
 * FPS:     0=NAME 1=C(alive)
 * ========================================================= */

/**
 * Sort direction multiplier: +1 for ascending,
 * -1 for descending. Set before each qsort call.
 */
static int ov_sort_dir_mul = 1;

/* ----- Stream comparators ----- */

static int sort_stream_by_name(
    const void *a, const void *b)
{
    return ov_sort_dir_mul * strcmp(
        ((const OV_STREAM *) a)->name,
        ((const OV_STREAM *) b)->name);
}

static int sort_stream_by_type(
    const void *a, const void *b)
{
    int ta = ((const OV_STREAM *) a)->datatype;
    int tb = ((const OV_STREAM *) b)->datatype;
    if (ta < tb) { return -ov_sort_dir_mul; }
    if (ta > tb) { return ov_sort_dir_mul; }
    return 0;
}

static int sort_stream_by_size(
    const void *a, const void *b)
{
    const OV_STREAM *sa = (const OV_STREAM *) a;
    const OV_STREAM *sb = (const OV_STREAM *) b;
    uint64_t na = sa->nelement;
    uint64_t nb = sb->nelement;
    if (na < nb) { return -ov_sort_dir_mul; }
    if (na > nb) { return ov_sort_dir_mul; }
    return 0;
}

static int sort_stream_by_hz(
    const void *a, const void *b)
{
    double ha = ((const OV_STREAM *) a)->update_hz;
    double hb = ((const OV_STREAM *) b)->update_hz;
    if (ha < hb) { return -ov_sort_dir_mul; }
    if (ha > hb) { return ov_sort_dir_mul; }
    return 0;
}

static int sort_stream_by_inode(
    const void *a, const void *b)
{
    ino_t ia = ((const OV_STREAM *) a)->inode;
    ino_t ib = ((const OV_STREAM *) b)->inode;
    if (ia < ib) { return -ov_sort_dir_mul; }
    if (ia > ib) { return ov_sort_dir_mul; }
    return 0;
}

static int sort_stream_by_count(
    const void *a, const void *b)
{
    uint64_t ca = ((const OV_STREAM *) a)->cnt0;
    uint64_t cb = ((const OV_STREAM *) b)->cnt0;
    if (ca < cb) { return -ov_sort_dir_mul; }
    if (ca > cb) { return ov_sort_dir_mul; }
    return 0;
}

/** Number of sortable stream columns. */
#define OV_STREAM_SORT_NCOL 6

void ov_sort_streams(
    OV_MODEL *model, int key, int dir)
{
    if (model->nb_streams < 2)
    {
        return;
    }
    ov_sort_dir_mul = dir ? -1 : 1;
    int (*cmp)(const void *, const void *);
    switch (key)
    {
    case 1:  cmp = sort_stream_by_type;  break;
    case 2:  cmp = sort_stream_by_size;  break;
    case 3:  cmp = sort_stream_by_hz;    break;
    case 4:  cmp = sort_stream_by_inode; break;
    case 5:  cmp = sort_stream_by_count; break;
    default: cmp = sort_stream_by_name;  break;
    }
    qsort(model->streams,
          (size_t) model->nb_streams,
          sizeof(OV_STREAM), cmp);
}


/* ----- Process comparators ----- */

static int sort_proc_by_name(
    const void *a, const void *b)
{
    return ov_sort_dir_mul * strcmp(
        ((const OV_PROC *) a)->name,
        ((const OV_PROC *) b)->name);
}

static int sort_proc_by_pid(
    const void *a, const void *b)
{
    pid_t pa = ((const OV_PROC *) a)->PID;
    pid_t pb = ((const OV_PROC *) b)->PID;
    if (pa < pb) { return -ov_sort_dir_mul; }
    if (pa > pb) { return ov_sort_dir_mul; }
    return 0;
}

static int sort_proc_by_stat(
    const void *a, const void *b)
{
    int sa = ((const OV_PROC *) a)->loopstat;
    int sb = ((const OV_PROC *) b)->loopstat;
    if (sa < sb) { return -ov_sort_dir_mul; }
    if (sa > sb) { return ov_sort_dir_mul; }
    return 0;
}

static int sort_proc_by_hz(
    const void *a, const void *b)
{
    double ha = ((const OV_PROC *) a)->loop_hz;
    double hb = ((const OV_PROC *) b)->loop_hz;
    if (ha < hb) { return -ov_sort_dir_mul; }
    if (ha > hb) { return ov_sort_dir_mul; }
    return 0;
}

/** Number of sortable proc columns. */
#define OV_PROC_SORT_NCOL 4

void ov_sort_procs(
    OV_MODEL *model, int key, int dir)
{
    if (model->nb_procs < 2)
    {
        return;
    }
    ov_sort_dir_mul = dir ? -1 : 1;
    int (*cmp)(const void *, const void *);
    switch (key)
    {
    case 1:  cmp = sort_proc_by_pid;  break;
    case 2:  cmp = sort_proc_by_stat; break;
    case 3:  cmp = sort_proc_by_hz;   break;
    default: cmp = sort_proc_by_name; break;
    }
    qsort(model->procs,
          (size_t) model->nb_procs,
          sizeof(OV_PROC), cmp);
}


/* ----- FPS comparators ----- */

static int sort_fps_by_name(
    const void *a, const void *b)
{
    return ov_sort_dir_mul * strcmp(
        ((const OV_FPS *) a)->name,
        ((const OV_FPS *) b)->name);
}

static int sort_fps_by_alive(
    const void *a, const void *b)
{
    const OV_FPS *fa = (const OV_FPS *) a;
    const OV_FPS *fb = (const OV_FPS *) b;
    int aa = fa->conf_alive + fa->run_alive;
    int ab = fb->conf_alive + fb->run_alive;
    if (aa < ab) { return -ov_sort_dir_mul; }
    if (aa > ab) { return ov_sort_dir_mul; }
    return 0;
}

/** Number of sortable FPS columns. */
#define OV_FPS_SORT_NCOL 2

void ov_sort_fps(
    OV_MODEL *model, int key, int dir)
{
    if (model->nb_fps < 2)
    {
        return;
    }
    ov_sort_dir_mul = dir ? -1 : 1;
    int (*cmp)(const void *, const void *);
    switch (key)
    {
    case 1:  cmp = sort_fps_by_alive; break;
    default: cmp = sort_fps_by_name;  break;
    }
    qsort(model->fps,
          (size_t) model->nb_fps,
          sizeof(OV_FPS), cmp);
}
