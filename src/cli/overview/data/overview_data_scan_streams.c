// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"
#include "ImageStreamIO/ImageStreamIO_config.h" /* IMAGESTRUCT_VERSION */


/* Exported by libImageStreamIO but not declared in ImageStreamIO.h (upstream follow-up) */
errno_t ImageStreamIO_image_sizing(IMAGE *image, uint8_t *map);

/**
 * scache_map_ro - map a stream file read-only: metadata, keywords, semaphores and traces.
 *
 * Unlike ImageStreamIO_read_sharedmem_image_toIMAGE() this needs no write access, does not
 * MAP_POPULATE or mlock the image data, keeps no fd, and rejects half-created files silently.
 * Release with ImageStreamIO_closeIm(). Return: 0, or -1 with nothing mapped.
 */
static int scache_map_ro(IMAGE *img, const char *path)
{
    ov_shm_map_t m;
    memset(img, 0, sizeof(*img));
    if (ov_shm_map(&m, path, 0, sizeof(IMAGE_METADATA) + 1) != 0)
    {
        return -1;
    }
    img->md = (IMAGE_METADATA *) m.addr;
    if (strncmp(img->md->version, IMAGESTRUCT_VERSION, sizeof(img->md->version)) != 0 ||
        img->md->shared != 1 ||
        ImageStreamIO_image_sizing(img, (uint8_t *) m.addr) != IMAGESTREAMIO_SUCCESS ||
        img->memsize != m.len)
    {
        ov_shm_unmap(&m);
        memset(img, 0, sizeof(*img));
        return -1;
    }
    img->semptr = (sem_t **) calloc((size_t) img->md->sem + 1, sizeof(sem_t *));
    if (img->semptr == NULL)
    {
        ov_shm_unmap(&m);
        memset(img, 0, sizeof(*img));
        return -1;
    }
    for (long si = 0; si < img->md->sem; si++)
    {
        img->semptr[si] = &img->semfile[si].semdata;
    }
    img->shmfd = -1;
    img->used  = 1;
    return 0;
}

/**
 * scache_rate_update - compute Hz and sparkline for
 *     a stream using cache's prev_cnt0.
 * @s:   the stream entry (already filled)
 * @ci:  cache index
 */
static void scache_rate_update(OV_STREAM *s, int ci)
{
    ov_stream_cache_t *ce = &s_scache[ci];

    s->cnt_active = 0;
    if (ce->has_prev && s_scan_dt_sec > 0.01)
    {
        uint64_t dc  = s->cnt0 - ce->prev_cnt0;
        s->update_hz = (double) dc / s_scan_dt_sec;

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
        ce->spark_rate[ce->spark_idx % OV_SPARKLINE_LEN] = norm;
        ce->spark_idx++;
    }

    /* Store current cnt0 for next tick */
    ce->prev_cnt0 = s->cnt0;
    ce->has_prev  = 1;

    /* Copy sparkline from cache to model */
    memcpy(s->spark_rate, ce->spark_rate, sizeof(s->spark_rate));
    s->spark_idx = ce->spark_idx;
    s->is_new    = ce->is_new;
    if (ce->is_new > 0)
    {
        ce->is_new--;
    }
}

/** live_unique_pids - the live PIDs among @src[0..n), each once, into @dst. Return: how many. */
static int live_unique_pids(const pid_t *src, int n, pid_t *dst)
{
    int nb = 0;
    for (int sm = 0; src != NULL && sm < n && nb < IMAGE_NB_SEMAPHORE; sm++)
    {
        int dup = 0;
        for (int k = 0; k < nb && !dup; k++)
        {
            dup = (dst[k] == src[sm]);
        }
        if (src[sm] > 0 && !dup && pid_is_alive(src[sm]))
        {
            dst[nb++] = src[sm];
        }
    }
    return nb;
}

/**
 * fill_stream_from_img - populate OV_STREAM from
 *     a cached IMAGE mapping.
 * @s:     stream entry to fill
 * @imgp:  persistent IMAGE pointer
 * @name:  stream name
 * @inode: inode value
 */
static void fill_stream_from_img(OV_STREAM *s, IMAGE *imgp, const char *name, ino_t inode)
{
    memset(s, 0, sizeof(OV_STREAM));
    strncpy(s->name, name, sizeof(s->name) - 1);
    s->valid = 1;
    s->inode = inode;

    if (imgp->md == NULL)
    {
        s->node_idx = -1;
        return;
    }

    s->datatype = imgp->md->datatype;
    s->naxis    = imgp->md->naxis;
    s->size[0]  = imgp->md->size[0];
    s->size[1]  = imgp->md->size[1];
    s->size[2]  = imgp->md->size[2];
    s->nelement = imgp->md->nelement;

    if (s->naxis == 1)
    {
        snprintf(s->size_str, sizeof(s->size_str), "%u", (unsigned) s->size[0]);
    }
    else if (s->naxis == 2)
    {
        snprintf(s->size_str, sizeof(s->size_str), "%ux%u", (unsigned) s->size[0],
                 (unsigned) s->size[1]);
    }
    else
    {
        snprintf(s->size_str, sizeof(s->size_str), "%ux%ux%u", (unsigned) s->size[0],
                 (unsigned) s->size[1], (unsigned) s->size[2]);
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
            STREAM_PROC_TRACE *spt = &imgp->streamproctrace[t];
            if (spt->procwrite_PID > 0)
            {
                int ti                    = s->nb_proctrace;
                s->proctrace_pid[ti]      = spt->procwrite_PID;
                s->proctrace_inode[ti]    = spt->trigger_inode;
                s->proctrace_trigmode[ti] = spt->triggermode;
                s->proctrace_status[ti]   = spt->triggerstatus;
                s->nb_proctrace++;
            }
        }
    }

    s->active = pid_is_alive(s->ownerPID) || pid_is_alive(s->creatorPID);

    s->nb_sem = (imgp->md != NULL) ? imgp->md->sem : 0;
    if (s->nb_sem > 10)
    {
        s->nb_sem = 10;
    }
    for (int sm = 0; sm < s->nb_sem; sm++)
    {
        if (imgp->semptr != NULL && imgp->semptr[sm] != NULL)
        {
            s->semval[sm] = ImageStreamIO_semvalue(imgp, sm);
        }
        else
        {
            s->semval[sm] = 0;
        }
        s->sem_read_pid[sm] = (imgp->semReadPID != NULL) ? imgp->semReadPID[sm] : 0;
    }

    /* Writer PID: first active proc trace entry */
    s->write_pid = 0;
    if (s->nb_proctrace > 0)
    {
        s->write_pid = s->proctrace_pid[0];
    }

    /* Readers and writers from the semaphore PID arrays: live, each once */
    s->nb_read_pids  = live_unique_pids(imgp->semReadPID, s->nb_sem, s->read_pids);
    s->nb_write_pids = live_unique_pids(imgp->semWritePID, s->nb_sem, s->write_pids);

    s->node_idx = -1;
}

/**
 * @brief Scan shared memory for active streams.
 */
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

    int dir_changed = (dirstat.st_mtim.tv_sec != s_shm_mtime.tv_sec) ||
                      (dirstat.st_mtim.tv_nsec != s_shm_mtime.tv_nsec);

    static int s_scanned; /* a full scan was done: the fast path holds even with no streams */
    if (!dir_changed && s_scanned)
    {
        /* Fast path: no files added/removed.
         * Just re-read metadata from cache. */
        pthread_mutex_lock(&s_scache_mutex);
        int idx = 0;
        for (int ci = 0; ci < s_scache_nb && idx < OV_MAX_STREAMS; ci++)
        {
            fill_stream_from_img(&model->streams[idx], &s_scache[ci].img, s_scache[ci].name,
                                 s_scache[ci].inode);
            scache_rate_update(&model->streams[idx], ci);
            idx++;
        }
        pthread_mutex_unlock(&s_scache_mutex);
        model->nb_streams = idx;
        return;
    }

    /* Full path: directory changed */
    s_shm_mtime = dirstat.st_mtim;
    s_scanned   = 1;

    DIR *dp = opendir(shmdir);
    if (dp == NULL)
    {
        model->nb_streams = 0;
        return;
    }

    pthread_mutex_lock(&s_scache_mutex);

    /* Mark all cache entries as not-in-use */
    for (int i = 0; i < s_scache_nb; i++)
    {
        s_scache[i].in_use = 0;
    }

    int            idx = 0;
    struct dirent *ep;

    while ((ep = readdir(dp)) != NULL && idx < OV_MAX_STREAMS)
    {
        /* Match *.im.shm */
        int namelen = (int) strlen(ep->d_name);
        if (namelen < 8)
        {
            continue;
        }
        if (strcmp(ep->d_name + namelen - 7, ".im.shm") != 0)
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
        memcpy(sname, ep->d_name, (size_t) snlen);
        sname[snlen] = '\0';

        /* Inode from the directory entry; stat() only where the filesystem does not report it */
        char fpath[1024];
        snprintf(fpath, sizeof(fpath), "%s/%s", shmdir, ep->d_name);
        ino_t ino = ep->d_ino;
        if (ino == 0)
        {
            struct stat st;
            if (stat(fpath, &st) != 0)
            {
                continue;
            }
            ino = st.st_ino;
        }

        /* Look up in cache */
        int    ci   = scache_find(sname);
        IMAGE *imgp = NULL;

        if (ci >= 0 && s_scache[ci].inode == ino)
        {
            /* Cache hit — reuse mapping */
            s_scache[ci].in_use = 1;
            imgp                = &s_scache[ci].img;
        }
        else
        {
            /* Cache miss or inode changed */
            if (ci >= 0)
            {
                scache_evict_locked(ci);
                ci = -1;
            }

            if (s_scache_nb >= OV_MAX_STREAMS)
            {
                continue;
            }

            ci = s_scache_nb;
            memset(&s_scache[ci], 0, sizeof(ov_stream_cache_t));

            if (scache_map_ro(&s_scache[ci].img, fpath) != 0)
            {
                continue;
            }

            strncpy(s_scache[ci].name, sname, sizeof(s_scache[ci].name) - 1);
            s_scache[ci].inode  = ino;
            s_scache[ci].in_use = 1;
            s_scache[ci].is_new = (s_scan_dt_sec > 0.0) ? OV_NEW_SCANS : 0; /* none at startup */
            s_scache_nb++;
            imgp = &s_scache[ci].img;
        }

        fill_stream_from_img(&model->streams[idx], imgp, sname, ino);
        scache_rate_update(&model->streams[idx], ci);
        idx++;
    }

    closedir(dp);
    model->nb_streams = idx;

    /* Evict stale cache entries */
    for (int i = s_scache_nb - 1; i >= 0; i--)
    {
        if (!s_scache[i].in_use)
        {
            scache_evict_locked(i);
        }
    }
    pthread_mutex_unlock(&s_scache_mutex);
}
