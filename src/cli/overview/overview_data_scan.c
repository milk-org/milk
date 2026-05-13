#include "overview_data_internal.h"

/* =========================================================
 * Stream scanning
 * ========================================================= */

/* Directory mtime for readdir skip optimization */
static struct timespec s_shm_mtime = {0, 0};

/* Scan-tick time delta (set by ov_model_full_scan
 * before calling scan functions) */
double s_scan_dt_sec = 0.0;


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

    s->cnt_active = 0;
    if (ce->has_prev && s_scan_dt_sec > 0.01)
    {
        uint64_t dc = s->cnt0 - ce->prev_cnt0;
        s->update_hz =
            (double) dc / s_scan_dt_sec;

        if (dc > 0)
        {
            s->cnt_active = 1;
        }

        /* Update sparkline in cache (auto-scale) */
        float sv = (float) s->update_hz;
        if (sv > ce->spark_max)
        {
            ce->spark_max = sv;
        }
        /* Decay max slowly so sparkline adapts */
        ce->spark_max *= 0.999f;
        if (ce->spark_max < 1.0f)
        {
            ce->spark_max = 1.0f;
        }
        float norm = sv / ce->spark_max;
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

    if (s->naxis == 1) {
        snprintf(s->size_str, sizeof(s->size_str), "%u", (unsigned)s->size[0]);
    } else if (s->naxis == 2) {
        snprintf(s->size_str, sizeof(s->size_str), "%ux%u", (unsigned)s->size[0], (unsigned)s->size[1]);
    } else {
        snprintf(s->size_str, sizeof(s->size_str), "%ux%ux%u", (unsigned)s->size[0], (unsigned)s->size[1], (unsigned)s->size[2]);
    }

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
 * fcache_build_params - discover param indices and cache them. 
 * Called once per FPS.
 * @ce: FPS cache entry
 */
static void fcache_build_params(
    ov_fps_cache_t *ce)
{
    FPS *fpsp = &ce->fps;
    int sp = 0;
    int dp = 0;

    if (fpsp->md == NULL)
    {
        ce->sparam_nb     = 0;
        ce->dparam_nb     = 0;
        ce->sparam_cached = 1;
        return;
    }

    int nb_params = fpsp->md->NBparamMAX;
    if (nb_params > 10000)
    {
        nb_params = 10000;
    }

    for (int p = 0; p < nb_params && (sp < OV_FPS_MAX_STREAM_PARAMS || dp < OV_FPS_MAX_DISP_PARAMS); p++)
    {
        FPS_PARAM *fp =
            &fpsp->parray[p];
        if (!(fp->fpflag & FPFLAG_ACTIVE))
        {
            continue;
        }
        char kbuf[FUNCTION_PARAMETER_STRMAXLEN];
        kbuf[0] = '\0';
        int klen = 0;
        for (int kl = 1; kl < FUNCTION_PARAMETER_KEYWORD_MAXLEVEL; kl++)
        {
            if (fp->keyword[kl][0] == '\0')
            {
                break;
            }
            if (klen > 0 && klen < FUNCTION_PARAMETER_STRMAXLEN - 1)
            {
                kbuf[klen++] = '.';
                kbuf[klen]   = '\0';
            }
            int rem = FUNCTION_PARAMETER_STRMAXLEN - klen - 1;
            if (rem > 0)
            {
                strncat(kbuf + klen, fp->keyword[kl], (size_t) rem);
                klen = (int) strlen(kbuf);
            }
        }
        if (kbuf[0] == '\0')
        {
            strncpy(kbuf, fp->keyword[0], FUNCTION_PARAMETER_STRMAXLEN - 1);
        }

        /* If there's room in dparam cache, add it */
        if (dp < OV_FPS_MAX_DISP_PARAMS)
        {
            ce->dparam_idx[dp] = p;
            strncpy(ce->dparam_key[dp], kbuf, FUNCTION_PARAMETER_STRMAXLEN - 1);
            dp++;
        }

        if (fp->type == FPTYPE_STREAMNAME && sp < OV_FPS_MAX_STREAM_PARAMS)
        {
            ce->sparam_idx[sp] = p;
            strncpy(ce->sparam_key[sp], kbuf, FUNCTION_PARAMETER_STRMAXLEN - 1);
            sp++;
        }
    }

    ce->sparam_nb     = sp;
    ce->dparam_nb     = dp;
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
    FPS *fpsp = &ce->fps;

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
    f->mem_rss_kb = (f->runpid > 0) ? pid_get_rss_kb(f->runpid) : 0;
    f->conf_alive = (pid_get_status(f->confpid) == OV_PID_ALIVE);
    f->run_alive  = pid_is_alive(f->runpid);

    if (!ce->sparam_cached)
    {
        fcache_build_params(ce);
    }

    /* Read only the cached stream-type params */
    for (int sp = 0; sp < ce->sparam_nb; sp++)
    {
        FPS_PARAM *fp =
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

    /* Read the cached display params */
    for (int dp = 0; dp < ce->dparam_nb; dp++)
    {
        FPS_PARAM *fp = &fpsp->parray[ce->dparam_idx[dp]];
        strncpy(f->disp_param_name[dp], ce->dparam_key[dp], FUNCTION_PARAMETER_STRMAXLEN - 1);
        
        /* Format value to string based on type */
        char valstr[FUNCTION_PARAMETER_STRMAXLEN] = {0};
        switch (fp->type)
        {
        case FPTYPE_INT64:
            snprintf(valstr, sizeof(valstr), "%" PRIi64, fp->val.i64[0]);
            break;
        case FPTYPE_FLOAT64:
            snprintf(valstr, sizeof(valstr), "%g", fp->val.f64[0]);
            break;
        case FPTYPE_FLOAT32:
            snprintf(valstr, sizeof(valstr), "%g", fp->val.f32[0]);
            break;
        case FPTYPE_PID:
            snprintf(valstr, sizeof(valstr), "%d", (int)fp->val.pid[0]);
            break;
        case FPTYPE_ONOFF:
            snprintf(valstr, sizeof(valstr), "%s", fp->val.i64[0] ? "ON" : "OFF");
            break;
        case FPTYPE_FPSNAME:
        case FPTYPE_STREAMNAME:
        case FPTYPE_STRING:
        case FPTYPE_DIRNAME:
        case FPTYPE_FILENAME:
        case FPTYPE_EXECFILENAME:
            strncpy(valstr, fp->val.string[0], sizeof(valstr) - 1);
            break;
        default:
            snprintf(valstr, sizeof(valstr), "[Type %d]", fp->type);
            break;
        }
        strncpy(f->disp_param_value[dp], valstr, FUNCTION_PARAMETER_STRMAXLEN - 1);
    }
    f->nb_disp_params = ce->dparam_nb;

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
                fps_connect(
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


/* Directory mtime for proc readdir skip */
static struct timespec s_proc_mtime = {0, 0};

void ov_scan_procs(OV_MODEL *model)
{
    char shmdir[OV_SHMDIR_MAXLEN];
    function_parameter_struct_shmdirname(shmdir);

    /* Check directory mtime to skip readdir when
     * no proc.*.shm files have been added or removed. */
    struct stat dirstat;
    if (stat(shmdir, &dirstat) != 0)
    {
        model->nb_procs = 0;
        return;
    }

    int dir_changed =
        (dirstat.st_mtim.tv_sec
            != s_proc_mtime.tv_sec)
        || (dirstat.st_mtim.tv_nsec
            != s_proc_mtime.tv_nsec);

    if (!dir_changed && s_pcache_nb > 0)
    {
        /* Fast path: refresh data from existing mappings */
        int idx = 0;
        for (int ci = 0;
             ci < s_pcache_nb && idx < OV_MAX_PROCS;
             ci++)
        {
            OV_PROC   *p     = &model->procs[idx];
            PROCESSINFO *pinfo = s_pcache[ci].pinfo;
            pid_t pid = s_pcache[ci].pid;

            memset(p, 0, sizeof(OV_PROC));
            strncpy(p->name, pinfo->name,
                    sizeof(p->name) - 1);
            p->PID    = pid;
            p->valid  = 1;
            p->active = 1;
            p->loopstat = pinfo->loopstat;

            int alive = pid_is_alive(pid);
            if (!alive)
            {
                p->loopstat =
                    (pinfo->loopstat == PROCESSINFO_LOOPSTAT_STOP)
                    ? PROCESSINFO_LOOPSTAT_STOP
                    : PROCESSINFO_LOOPSTAT_CRASHED;
            }

            p->CTRLval  = pinfo->CTRLval;
            p->loopcnt  = pinfo->loopcnt;
            p->mem_rss_kb = pid_get_rss_kb(pid);
            p->dtmedian_iter_ns = pinfo->dtmedian_iter_ns;
            p->dtmedian_exec_ns = pinfo->dtmedian_exec_ns;
            if (p->dtmedian_iter_ns > 0)
            {
                p->loop_hz =
                    1.0e9 / (double) p->dtmedian_iter_ns;
            }
            strncpy(p->trigstreamname,
                    pinfo->triggerstreamname,
                    sizeof(p->trigstreamname) - 1);
            p->triggermode    = pinfo->triggermode;
            p->triggersem     = pinfo->triggersem;
            p->triggermissed  = pinfo->triggermissedframe;
            p->triggermissed_cumul =
                pinfo->triggermissedframe_cumul;
            p->MeasureTiming  = pinfo->MeasureTiming;
            p->rt_priority    = pinfo->RT_priority;

            {
                ov_proc_cache_t *ce = &s_pcache[ci];

                /* Hz from loopcnt delta (fallback) */
                if (p->loop_hz < 0.1
                    && ce->has_prev_loop
                    && s_scan_dt_sec > 0.01)
                {
                    int64_t dlc =
                        p->loopcnt - ce->prev_loopcnt;
                    if (dlc > 0)
                    {
                        p->loop_hz =
                            (double) dlc / s_scan_dt_sec;
                    }
                }
                /* Flag whether counter is advancing */
                p->cnt_active =
                    (ce->has_prev_loop
                     && p->loopcnt != ce->prev_loopcnt);
                ce->prev_loopcnt  = p->loopcnt;
                ce->has_prev_loop = 1;

                /* CPU percent from tick delta */
                unsigned long ut = 0, st = 0;
                if (pid_get_cpu_ticks(pid, &ut, &st) == 0)
                {
                    if (ce->has_prev_cpu
                        && s_scan_dt_sec > 0.01)
                    {
                        long clk = sysconf(_SC_CLK_TCK);
                        unsigned long dticks =
                            (ut - ce->prev_utime)
                            + (st - ce->prev_stime);
                        ce->cpu_pct = (float)(
                            (double) dticks
                            / ((double) clk
                                * s_scan_dt_sec)
                            * 100.0);
                    }
                    ce->prev_utime   = ut;
                    ce->prev_stime   = st;
                    ce->has_prev_cpu = 1;
                }
                p->cpu_used = ce->cpu_pct;
            }

            p->node_idx = -1;
            idx++;
        }
        model->nb_procs = idx;
        return;
    }

    /* Full path: directory changed — rescan via readdir */
    s_proc_mtime = dirstat.st_mtim;

    /* Mark all proc cache entries as not-in-use */
    for (int i = 0; i < s_pcache_nb; i++)
    {
        s_pcache[i].in_use = 0;
    }

    DIR *dp = opendir(shmdir);
    if (dp == NULL)
    {
        model->nb_procs = 0;
        return;
    }

    int idx = 0;
    struct dirent *ep;

    while ((ep = readdir(dp)) != NULL
           && idx < OV_MAX_PROCS)
    {
        /* Match proc.*.shm — at minimum "proc.a.1.shm" */
        const char *fname = ep->d_name;
        if (strncmp(fname, "proc.", 5) != 0)
        {
            continue;
        }
        int flen = (int) strlen(fname);
        if (flen < 10)
        {
            continue;
        }
        if (strcmp(fname + flen - 4, ".shm") != 0)
        {
            continue;
        }
        /* proc.NAME.PID.shm — find last dot before .shm
         * to split NAME and PID. */
        /* Find second-to-last '.' */
        const char *p_sfx = fname + flen - 4; /* points to .shm */
        const char *p_pid_end = p_sfx;        /* one past PID digits */
        /* Walk backward to the '.' before PID */
        const char *q = p_pid_end - 1;
        while (q > fname + 5 && *q != '.')
        {
            q--;
        }
        if (*q != '.')
        {
            continue;
        }
        /* Parse PID */
        pid_t pid = (pid_t) atoi(q + 1);
        if (pid <= 0)
        {
            continue;
        }
        /* Extract process name: fname+5 .. q-1 */
        int pname_len = (int)(q - (fname + 5));
        if (pname_len <= 0 ||
            pname_len >= (int) sizeof(((OV_PROC *)0)->name))
        {
            continue;
        }

        /* Full path for mmap */
        char fpath[1024];
        snprintf(fpath, sizeof(fpath),
                 "%s/%s", shmdir, fname);

        /* Look up in proc cache by PID */
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
            /* Cache miss — open and mmap */
            if (s_pcache_nb >= OV_MAX_PROCS)
            {
                continue;
            }

            int pfd = open(fpath, O_RDONLY);
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
        } // cache miss

        OV_PROC *p = &model->procs[idx];
        memset(p, 0, sizeof(OV_PROC));

        /* Use name from PROCESSINFO struct if available,
         * otherwise fall back to the name from the filename. */
        if (pinfo != NULL && pinfo->name[0] != '\0')
        {
            strncpy(p->name, pinfo->name,
                    sizeof(p->name) - 1);
        }
        else
        {
            memcpy(p->name, fname + 5, (size_t) pname_len);
            p->name[pname_len] = '\0';
        }

        p->PID    = pid;
        p->valid  = 1;
        p->active = 1;

        if (pinfo != NULL)
        {
            p->loopstat = pinfo->loopstat;
        }

        int alive = pid_is_alive(pid);
        if (!alive)
        {
            p->loopstat =
                (pinfo != NULL &&
                 pinfo->loopstat == PROCESSINFO_LOOPSTAT_STOP)
                ? PROCESSINFO_LOOPSTAT_STOP
                : PROCESSINFO_LOOPSTAT_CRASHED;
        }

        if (pinfo != NULL)
        {
            p->CTRLval  = pinfo->CTRLval;
            p->loopcnt  = pinfo->loopcnt;
            p->mem_rss_kb = pid_get_rss_kb(pinfo->PID);

            /* Timing stats */
            p->dtmedian_iter_ns = pinfo->dtmedian_iter_ns;
            p->dtmedian_exec_ns = pinfo->dtmedian_exec_ns;
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
            p->triggermode    = pinfo->triggermode;
            p->triggersem     = pinfo->triggersem;
            p->triggermissed  = pinfo->triggermissedframe;
            p->triggermissed_cumul =
                pinfo->triggermissedframe_cumul;
            p->MeasureTiming  = pinfo->MeasureTiming;
            p->rt_priority    = pinfo->RT_priority;

            /* CPU% and Hz tracking via cache delta */
            {
                ov_proc_cache_t *ce = &s_pcache[ci];

                /* Hz from loopcnt delta (fallback) */
                if (p->loop_hz < 0.1
                    && ce->has_prev_loop
                    && s_scan_dt_sec > 0.01)
                {
                    int64_t dlc =
                        p->loopcnt - ce->prev_loopcnt;
                    if (dlc > 0)
                    {
                        p->loop_hz =
                            (double) dlc / s_scan_dt_sec;
                    }
                }
                /* Flag whether counter is advancing */
                p->cnt_active =
                    (ce->has_prev_loop
                     && p->loopcnt != ce->prev_loopcnt);
                ce->prev_loopcnt  = p->loopcnt;
                ce->has_prev_loop = 1;

                /* CPU percent from tick delta */
                unsigned long ut = 0, st = 0;
                if (pid_get_cpu_ticks(pid, &ut, &st) == 0)
                {
                    if (ce->has_prev_cpu
                        && s_scan_dt_sec > 0.01)
                    {
                        long clk = sysconf(
                            _SC_CLK_TCK);
                        unsigned long dticks =
                            (ut - ce->prev_utime)
                            + (st - ce->prev_stime);
                        ce->cpu_pct = (float)(
                            (double) dticks
                            / ((double) clk
                                * s_scan_dt_sec)
                            * 100.0);
                    }
                    ce->prev_utime   = ut;
                    ce->prev_stime   = st;
                    ce->has_prev_cpu = 1;
                }
                p->cpu_used = ce->cpu_pct;
            }
        } // if (pinfo != NULL)

        p->node_idx = -1;
        idx++;
    } // while readdir

    closedir(dp);
    model->nb_procs = idx;

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
        fps_disconnect(
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

    /* Reset directory mtime trackers */
    memset(&s_shm_mtime, 0,
           sizeof(s_shm_mtime));
    memset(&s_fps_mtime, 0,
           sizeof(s_fps_mtime));
    memset(&s_proc_mtime, 0,
           sizeof(s_proc_mtime));

    /* Reset PID liveness cache */
    pid_cache_reset();

    /* Reset graph and rate state */
    s_scan_dt_sec  = 0.0;
}


