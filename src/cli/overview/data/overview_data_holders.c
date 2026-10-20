// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    overview_data_holders.c
 * @brief   Which processes map which stream: one pass over /proc/<pid>/maps for all streams, only while the
 *          HOLDERS page is on screen and at most every 2 s. No fuser, no shell.
 */

#include "overview_data_internal.h"

#include <ctype.h>
#include <dirent.h>

#define HOLDERS_PERIOD_S 2.0

static double s_last_t = -1, s_ms, s_unreadable;

/* Monotonic seconds */
static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

/* Record that @pid maps the stream file at @path (a *.im.shm directly in @dir, @dlen chars) */
static void note_mapping(const char *path, const char *dir, size_t dlen, pid_t pid)
{
    char        name[STRINGMAXLEN_IMAGE_NAME];
    const char *e = strstr(path, ".im.shm");
    if (strncmp(path, dir, dlen) != 0 || path[dlen] != '/' || e == NULL ||
        (e[7] != '\n' && e[7] != '\0') ||
        memchr(path + dlen + 1, '/', (size_t) (e - path - (long) dlen - 1)) != NULL)
    {
        return;
    }
    snprintf(name, sizeof(name), "%.*s", (int) (e - path - (long) dlen - 1), path + dlen + 1);
    int ci = scache_find(name);
    if (ci < 0)
    {
        return;
    }
    ov_stream_cache_t *ce = &s_scache[ci];
    for (int k = 0; k < ce->nb_holders && k < OV_MAX_HOLDERS; k++)
    {
        if (ce->holder[k] == pid)
        {
            return; /* mapped twice (data and metadata) */
        }
    }
    if (ce->nb_holders < OV_MAX_HOLDERS)
    {
        ce->holder[ce->nb_holders] = pid;
    }
    ce->nb_holders++;
}

/* One pass: every readable /proc/<pid>/maps but our own */
static void holders_pass(const char *dir)
{
    size_t dlen = strlen(dir);
    pid_t  self = getpid();
    DIR   *d    = opendir("/proc");
    char   path[64], line[1024];
    s_unreadable = 0;
    for (int i = 0; i < s_scache_nb; i++)
    {
        s_scache[i].nb_holders = 0;
    }
    for (struct dirent *de; d != NULL && (de = readdir(d)) != NULL;)
    {
        if (!isdigit((unsigned char) de->d_name[0]) || atoi(de->d_name) == self)
        {
            continue;
        }
        snprintf(path, sizeof(path), "/proc/%s/maps", de->d_name);
        FILE *f = fopen(path, "re");
        if (f == NULL)
        {
            s_unreadable += (errno == EACCES);
            continue;
        }
        while (fgets(line, sizeof(line), f) != NULL)
        {
            char *p = strchr(line, '/'); /* the path column */
            if (p != NULL && strstr(p, ".im.shm") != NULL)
            {
                note_mapping(p, dir, dlen, (pid_t) atoi(de->d_name));
            }
        }
        fclose(f);
    }
    if (d != NULL)
    {
        closedir(d);
    }
}

/**
 * ov_scan_holders - while OV_HINT_HOLDERS is set, find the processes mapping each stream (every 2 s); the counts
 * stay in the stream cache for the rows and the HOLD sort, and the pass's age and cost go to @model.
 */
void ov_scan_holders(OV_MODEL *model)
{
    double t = now_s();
    if ((ov_scan_hints() & OV_HINT_HOLDERS) && (s_last_t < 0 || t - s_last_t >= HOLDERS_PERIOD_S))
    {
        char dir[1024];
        snprintf(dir, sizeof(dir), "%s", SHAREDSHMDIR);
        pthread_mutex_lock(&s_scache_mutex);
        holders_pass(dir);
        pthread_mutex_unlock(&s_scache_mutex);
        s_last_t = t;
        s_ms     = (now_s() - t) * 1e3;
    }
    model->holders_age_s      = (s_last_t < 0) ? -1 : t - s_last_t;
    model->holders_ms         = s_ms;
    model->holders_unreadable = (int) s_unreadable;
}
