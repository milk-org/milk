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
    }

    /* Store current cnt0 for next tick */
    ce->prev_cnt0 = s->cnt0;
    ce->has_prev  = 1;
    s->is_new     = ce->is_new;
    if (ce->is_new > 0)
    {
        ce->is_new--;
    }
}

/** ts_ns - a timespec in nanoseconds (0 stays 0: never set). */
static int64_t ts_ns(const struct timespec *ts)
{
    return (int64_t) ts->tv_sec * 1000000000LL + ts->tv_nsec;
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

/** size_string - "W", "WxH" or "WxHxD" of a mapped image into @buf. */
static void size_string(const IMAGE *img, char *buf, size_t n)
{
    const uint32_t *sz = img->md->size;
    if (img->md->naxis == 1)
    {
        snprintf(buf, n, "%u", (unsigned) sz[0]);
    }
    else if (img->md->naxis == 2)
    {
        snprintf(buf, n, "%ux%u", (unsigned) sz[0], (unsigned) sz[1]);
    }
    else
    {
        snprintf(buf, n, "%ux%ux%u", (unsigned) sz[0], (unsigned) sz[1], (unsigned) sz[2]);
    }
}

/** fill_stream_from_img - fill @s from cache entry @ce's mapping; its trace hops go to @m's pool. */
static void fill_stream_from_img(OV_MODEL *m, OV_STREAM *s, ov_stream_cache_t *ce)
{
    IMAGE      *imgp  = &ce->img;
    const char *name  = ce->name;
    ino_t       inode = ce->inode;
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

    memcpy(s->size_str, ce->size_str, sizeof(s->size_str));

    s->creatorPID = imgp->md->creatorPID;
    s->ownerPID   = imgp->md->ownerPID;
    s->cnt0       = imgp->md->cnt0;

    /* Process trace entries */
    int npt = imgp->md->NBproctrace;
    if (npt > IMAGE_NB_PROCTRACE)
    {
        npt = IMAGE_NB_PROCTRACE;
    }
    for (int t = 0; imgp->streamproctrace != NULL && t < npt; t++)
    {
        STREAM_PROC_TRACE *spt = &imgp->streamproctrace[t];
        OV_TRACE_HOP      *h   = (spt->procwrite_PID > 0) ? ov_stream_add_hop(m, s) : NULL;
        if (h != NULL)
        {
            *h = (OV_TRACE_HOP) { .pid        = spt->procwrite_PID,
                                  .inode      = spt->trigger_inode,
                                  .trigmode   = spt->triggermode,
                                  .status     = spt->triggerstatus,
                                  .tstart_ns  = ts_ns(&spt->ts_procstart),
                                  .tupdate_ns = ts_ns(&spt->ts_streamupdate),
                                  .cnt0       = spt->cnt0 };
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
    s->write_pid = (s->nb_proctrace > 0) ? ov_stream_hops(m, s)[0].pid : 0;

    /* Readers and writers from the semaphore PID arrays: live, each once */
    s->nb_read_pids  = live_unique_pids(imgp->semReadPID, s->nb_sem, s->read_pids);
    s->nb_write_pids = live_unique_pids(imgp->semWritePID, s->nb_sem, s->write_pids);

    s->node_idx = -1;
}

/** fill_stream_pending - a listed stream not mapped yet: name and inode only. */
static void fill_stream_pending(OV_STREAM *s, const char *name, ino_t inode)
{
    memset(s, 0, sizeof(*s));
    snprintf(s->name, sizeof(s->name), "%s", name);
    s->inode    = inode;
    s->pending  = 1;
    s->node_idx = -1;
}

/**
 * scache_try_map - map cache entry @ci from @path unless done, while *@budget mappings remain this scan.
 * Return: 0 mapped, 1 waiting (no budget), -1 the file cannot be mapped.
 */
static int scache_try_map(int ci, const char *path, int *budget)
{
    if (s_scache[ci].mapped)
    {
        return 0;
    }
    if (*budget <= 0)
    {
        return 1;
    }
    (*budget)--;
    if (scache_map_ro(&s_scache[ci].img, path) != 0)
    {
        return -1;
    }
    s_scache[ci].mapped = 1;
    size_string(&s_scache[ci].img, s_scache[ci].size_str, sizeof(s_scache[ci].size_str));
    return 0;
}

/** emit_stream - model row @idx from cache entry @ci, mapped or waiting. Return: 1 if it waits. */
static int emit_stream(OV_MODEL *model, int idx, int ci)
{
    if (!s_scache[ci].mapped)
    {
        fill_stream_pending(&model->streams[idx], s_scache[ci].name, s_scache[ci].inode);
        return 1;
    }
    fill_stream_from_img(model, &model->streams[idx], &s_scache[ci]);
    scache_rate_update(&model->streams[idx], ci);
    return 0;
}

/**
 * ov_scan_streams - list the *.im.shm of the shm directory (readdir only when its mtime changed) and fill the
 * model's streams. At most OV_MAP_PER_SCAN new files are mapped per scan; the others are listed as pending
 * (model->streams_pending) until a later scan maps them.
 */
void ov_scan_streams(OV_MODEL *model)
{
    const char *shmdir = SHAREDSHMDIR;
    static int  s_scanned; /* a full scan was done: the fast path holds even with no streams */
    static int  s_capped;  /* the last readdir found more streams than OV_MAX_STREAMS */
    int         budget = OV_MAP_PER_SCAN, idx = 0, waiting = 0;
    char        fpath[1024];
    struct stat dirstat;

    model->nb_trace        = 0;
    model->streams_pending = 0;
    model->streams_capped  = s_capped;
    if (stat(shmdir, &dirstat) != 0)
    {
        model->nb_streams = 0;
        return;
    }
    int dir_changed = (dirstat.st_mtim.tv_sec != s_shm_mtime.tv_sec) ||
                      (dirstat.st_mtim.tv_nsec != s_shm_mtime.tv_nsec);

    if (!dir_changed && s_scanned)
    {
        /* Fast path: no files added or removed; map what still waits, re-read the rest */
        pthread_mutex_lock(&s_scache_mutex);
        for (int ci = 0; ci < s_scache_nb && idx < OV_MAX_STREAMS;)
        {
            snprintf(fpath, sizeof(fpath), "%s/%s.im.shm", shmdir, s_scache[ci].name);
            if (scache_try_map(ci, fpath, &budget) < 0)
            {
                scache_evict_locked(ci); /* the last entry moves to ci */
                continue;
            }
            waiting += emit_stream(model, idx++, ci++);
        }
        pthread_mutex_unlock(&s_scache_mutex);
        model->nb_streams      = idx;
        model->streams_pending = waiting;
        return;
    }

    /* Full path: directory changed */
    s_shm_mtime = dirstat.st_mtim;
    s_scanned   = 1;
    s_capped    = 0;

    DIR *dp = opendir(shmdir);
    if (dp == NULL)
    {
        model->nb_streams = 0;
        return;
    }

    pthread_mutex_lock(&s_scache_mutex);
    for (int i = 0; i < s_scache_nb; i++)
    {
        s_scache[i].in_use = 0;
    }

    struct dirent *ep;
    while ((ep = readdir(dp)) != NULL)
    {
        /* Match *.im.shm */
        int namelen = (int) strlen(ep->d_name);
        if (namelen < 8 || strcmp(ep->d_name + namelen - 7, ".im.shm") != 0)
        {
            continue;
        }
        if (idx >= OV_MAX_STREAMS)
        {
            s_capped = 1;
            continue;
        }

        char sname[STRINGMAXLEN_IMAGE_NAME];
        snprintf(sname, sizeof(sname), "%.*s", namelen - 7, ep->d_name);

        /* Inode from the directory entry; stat() only where the filesystem does not report it */
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

        int ci = scache_find(sname);
        if (ci >= 0 && s_scache[ci].inode != ino)
        {
            scache_evict_locked(ci); /* recreated: map the new file */
            ci = -1;
        }
        if (ci < 0)
        {
            if (s_scache_nb >= OV_MAX_STREAMS)
            {
                s_capped = 1;
                continue;
            }
            ci = s_scache_nb++;
            memset(&s_scache[ci], 0, sizeof(ov_stream_cache_t));
            snprintf(s_scache[ci].name, sizeof(s_scache[ci].name), "%s", sname);
            s_scache[ci].inode  = ino;
            s_scache[ci].is_new = (s_scan_dt_sec > 0.0) ? OV_NEW_SCANS : 0; /* none at startup */
        }
        if (scache_try_map(ci, fpath, &budget) < 0)
        {
            scache_evict_locked(ci);
            continue;
        }
        s_scache[ci].in_use = 1;
        waiting += emit_stream(model, idx++, ci);
    }

    closedir(dp);
    model->nb_streams      = idx;
    model->streams_pending = waiting;
    model->streams_capped  = s_capped;

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
