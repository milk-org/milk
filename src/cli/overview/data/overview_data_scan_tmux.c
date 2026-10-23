// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE /* pipe2 */
#endif

#include "overview_data_internal.h"
#include "milk_spawn.h"
#include "overview_data_index.h"
#include <math.h>
#include <stdatomic.h>

#define OV_TMUX_PERIOD_MIN 3.0  /* s between polls while windows come and go */
#define OV_TMUX_PERIOD_MAX 15.0 /* s between polls once the window set is stable */


typedef struct
{
    char    name[STRINGMAXLEN_FPS_NAME];
    uint8_t flags;
} ov_tmux_cache_entry_t;

static ov_tmux_cache_entry_t s_tmux_cache[OV_MAX_FPS];
static int                   s_tmux_cache_cnt = 0;
static struct timespec       s_last_tmux_scan = { 0, 0 };
static double                s_tmux_period = OV_TMUX_PERIOD_MIN; /* doubles while nothing changes */
static atomic_int            s_tmux_refresh;                     /* poll at the next scan */
static unsigned              s_tmux_polls; /* spawns so far (model->tmux_polls) */

/** ov_scan_tmux_refresh - poll tmux at the next scan (after an action that may start or stop windows). */
void ov_scan_tmux_refresh(void)
{
    atomic_store(&s_tmux_refresh, 1);
}

/** tmux_cache_sig - signature of the cached session/window set, to tell whether a poll changed it. */
static uint64_t tmux_cache_sig(void)
{
    uint64_t h = 0;
    for (int j = 0; j < s_tmux_cache_cnt; j++)
    {
        h += ov_hash_str(s_tmux_cache[j].name) *
             (1U + s_tmux_cache[j].flags); /* order independent */
    }
    return h ^ (uint64_t) s_tmux_cache_cnt;
}

/** tmux_apply_cache - copy the cached flags onto the model's FPS. */
static void tmux_apply_cache(OV_MODEL *model)
{
    for (int i = 0; i < model->nb_fps; i++)
    {
        for (int j = 0; j < s_tmux_cache_cnt; j++)
        {
            if (strcmp(model->fps[i].name, s_tmux_cache[j].name) == 0)
            {
                model->fps[i].tmux_flags = s_tmux_cache[j].flags;
                break;
            }
        }
    }
}

/**
 * ov_scan_tmux_sessions - Query tmux server sessions and match against FPS entries
 * @model: Pointer to data model
 *
 * Populates tmux session flags (e.g. running, attached, ctrl window) for each
 * known FPS instance by parsing tmux list-windows output. tmux is spawned only while an FPS panel is
 * on screen (OV_HINT_FPS_VISIBLE), every 3 s while windows change, backing off to 15 s while they do
 * not; otherwise the last flags are used.
 */
void ov_scan_tmux_sessions(OV_MODEL *model)
{
    /* Reset all tmux flags */
    for (int i = 0; i < model->nb_fps; i++)
    {
        model->fps[i].tmux_flags = 0;
    }

    if (model->nb_fps == 0)
    {
        return;
    }

    /* Fast check: does the tmux socket directory exist?
     * If not, tmux is definitely not running, avoid spawning it. */
    static char tmux_dir[128] = { 0 };
    if (tmux_dir[0] == '\0')
    {
        const char *tmpdir = getenv("TMUX_TMPDIR");
        if (tmpdir != NULL && tmpdir[0] != '\0')
        {
            snprintf(tmux_dir, sizeof(tmux_dir), "%s/tmux-%u", tmpdir, (unsigned int) getuid());
        }
        else
        {
            snprintf(tmux_dir, sizeof(tmux_dir), "/tmp/tmux-%u", (unsigned int) getuid());
        }
    }

    if (access(tmux_dir, F_OK) != 0)
    {
        s_tmux_cache_cnt = 0;
        return;
    }

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed    = (double) (now.tv_sec - s_last_tmux_scan.tv_sec) +
                        (double) (now.tv_nsec - s_last_tmux_scan.tv_nsec) * 1e-9;
    int    due        = atomic_exchange(&s_tmux_refresh, 0) || s_last_tmux_scan.tv_sec == 0 ||
                        elapsed >= s_tmux_period;
    model->tmux_polls = s_tmux_polls;
    if (!due || !(ov_scan_hints() & OV_HINT_FPS_VISIBLE))
    {
        tmux_apply_cache(model);
        return;
    }

    uint64_t before   = tmux_cache_sig();
    s_last_tmux_scan  = now;
    s_tmux_cache_cnt  = 0;
    model->tmux_polls = ++s_tmux_polls;

    static const char *const argv[] = {
        "tmux", "list-windows", "-a", "-F", "#{session_name}:#{window_name}", NULL
    };
    int pfd[2];
    if (pipe2(pfd, O_CLOEXEC) != 0)
    {
        return;
    }
    pid_t pid = milkspawn_start(argv, pfd[1], -1); /* no /bin/sh; stdin and stderr: /dev/null */
    close(pfd[1]);
    FILE *fp = (pid < 0) ? NULL : fdopen(pfd[0], "r");
    if (fp == NULL)
    {
        close(pfd[0]);
        if (pid >= 0)
        {
            milkspawn_wait(pid);
        }
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), fp) != NULL)
    {
        /* Strip newline */
        int len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
        {
            line[--len] = '\0';
        }

        /* Find the colon */
        char *colon = strchr(line, ':');
        if (colon == NULL)
        {
            continue;
        }

        *colon                   = '\0';
        const char *session_name = line;
        const char *window_name  = colon + 1;

        uint8_t win_flag = 0;
        if (strcmp(window_name, "ctrl") == 0)
        {
            win_flag = OV_TMUX_CTRL;
        }
        else if (strcmp(window_name, "conf") == 0)
        {
            win_flag = OV_TMUX_CONF;
        }
        else if (strcmp(window_name, "run") == 0)
        {
            win_flag = OV_TMUX_RUN;
        }

        if (win_flag != 0)
        {
            /* Update cache entry */
            int found_cache = 0;
            for (int j = 0; j < s_tmux_cache_cnt; j++)
            {
                if (strcmp(s_tmux_cache[j].name, session_name) == 0)
                {
                    s_tmux_cache[j].flags |= win_flag;
                    found_cache = 1;
                    break;
                }
            }
            if (!found_cache && s_tmux_cache_cnt < OV_MAX_FPS)
            {
                strncpy(s_tmux_cache[s_tmux_cache_cnt].name, session_name,
                        sizeof(s_tmux_cache[s_tmux_cache_cnt].name) - 1);
                s_tmux_cache[s_tmux_cache_cnt].name[sizeof(s_tmux_cache[0].name) - 1] = '\0';
                s_tmux_cache[s_tmux_cache_cnt].flags                                  = win_flag;
                s_tmux_cache_cnt++;
            }

            /* Find matching FPS in model */
            for (int i = 0; i < model->nb_fps; i++)
            {
                if (strcmp(model->fps[i].name, session_name) == 0)
                {
                    model->fps[i].tmux_flags |= win_flag;
                    break;
                }
            }
        }
    }

    fclose(fp);
    milkspawn_wait(pid);
    s_tmux_period = (tmux_cache_sig() == before) ? fmin(2.0 * s_tmux_period, OV_TMUX_PERIOD_MAX)
                                                 : OV_TMUX_PERIOD_MIN;
}
