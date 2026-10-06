// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"


/**
 * fcache_build_params - discover param indices and cache them.
 * Called once per FPS.
 * @ce: FPS cache entry
 */
void fcache_build_params(ov_fps_cache_t *ce)
{
    FPS *fpsp = &ce->fps;
    int  sp   = 0;
    int  dp   = 0;

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

    for (int p = 0; p < nb_params && (sp < OV_FPS_MAX_STREAM_PARAMS || dp < OV_FPS_MAX_DISP_PARAMS);
         p++)
    {
        FPS_PARAM *fp = &fpsp->parray[p];
        if (!(fp->fpflag & FPFLAG_ACTIVE))
        {
            continue;
        }
        char kbuf[FUNCTION_PARAMETER_STRMAXLEN];
        kbuf[0]  = '\0';
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
static void fill_fps_from_struct(OV_FPS *f, ov_fps_cache_t *ce)
{
    FPS *fpsp = &ce->fps;

    memset(f, 0, sizeof(OV_FPS));
    strncpy(f->name, ce->fname, sizeof(f->name) - 1);
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
        FPS_PARAM *fp = &fpsp->parray[ce->sparam_idx[sp]];
        strncpy(f->stream_param_name[sp], ce->sparam_key[sp], FUNCTION_PARAMETER_STRMAXLEN - 1);
        strncpy(f->stream_param_value[sp], fp->val.string[0], FUNCTION_PARAMETER_STRMAXLEN - 1);
        f->stream_param_flags[sp] = fp->fpflag;
    }
    f->nb_stream_params = ce->sparam_nb;

    /* Display parameters are queried on demand via ov_fps_get_params() */
    f->nb_disp_params = ce->dparam_nb;

    /* Read description from md */
    if (fpsp->md->description[0] != '\0')
    {
        strncpy(f->description, fpsp->md->description, sizeof(f->description) - 1);
    }

    f->node_idx = -1;
}

/**
 * @brief Scan shared memory for active FPS instances.
 */
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

    int dir_changed = (dirstat.st_mtim.tv_sec != s_fps_mtime.tv_sec) ||
                      (dirstat.st_mtim.tv_nsec != s_fps_mtime.tv_nsec);

    if (!dir_changed && s_fcache_nb > 0)
    {
        /* Fast path: no files added/removed */
        pthread_mutex_lock(&s_fcache_mutex);
        int idx = 0;
        for (int ci = 0; ci < s_fcache_nb && idx < OV_MAX_FPS; ci++)
        {
            fill_fps_from_struct(&model->fps[idx], &s_fcache[ci]);
            idx++;
        }
        pthread_mutex_unlock(&s_fcache_mutex);
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

    /* Mark all FPS cache entries as not-in-use under cache mutex */
    pthread_mutex_lock(&s_fcache_mutex);
    for (int i = 0; i < s_fcache_nb; i++)
    {
        s_fcache[i].in_use = 0;
    }

    int            idx = 0;
    struct dirent *ep;

    while ((ep = readdir(dp)) != NULL && idx < OV_MAX_FPS)
    {
        /* Match *.fps.shm */
        int namelen = (int) strlen(ep->d_name);
        if (namelen < 9)
        {
            continue;
        }
        if (strcmp(ep->d_name + namelen - 8, ".fps.shm") != 0)
        {
            continue;
        }

        /* Extract FPS name */
        char fname[STRINGMAXLEN_FPS_NAME];
        int  fnlen = namelen - 8;
        if (fnlen >= (int) sizeof(fname))
        {
            fnlen = (int) sizeof(fname) - 1;
        }
        memcpy(fname, ep->d_name, (size_t) fnlen);
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
            memset(&s_fcache[ci], 0, sizeof(ov_fps_cache_t));
            s_fcache[ci].fps.SMfd = -1;

            long fpsID = fps_connect(fname, &s_fcache[ci].fps, FPSCONNECT_SIMPLE);
            if (fpsID < 0)
            {
                continue;
            }

            strncpy(s_fcache[ci].fname, fname, sizeof(s_fcache[ci].fname) - 1);
            s_fcache[ci].in_use = 1;
            s_fcache_nb++;
        }

        fill_fps_from_struct(&model->fps[idx], &s_fcache[ci]);
        idx++;
    }

    closedir(dp);
    model->nb_fps = idx;

    /* Evict stale FPS cache entries */
    for (int i = s_fcache_nb - 1; i >= 0; i--)
    {
        if (!s_fcache[i].in_use)
        {
            fcache_evict_locked(i);
        }
    }
    pthread_mutex_unlock(&s_fcache_mutex);
}
