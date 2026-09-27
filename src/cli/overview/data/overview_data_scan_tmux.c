// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "overview_data_internal.h"


typedef struct
{
    char    name[STRINGMAXLEN_FPS_NAME];
    uint8_t flags;
} ov_tmux_cache_entry_t;

static ov_tmux_cache_entry_t s_tmux_cache[OV_MAX_FPS];
static int                   s_tmux_cache_cnt = 0;
static struct timespec       s_last_tmux_scan = { 0, 0 };

/**
 * ov_scan_tmux_sessions - Query tmux server sessions and match against FPS entries
 * @model: Pointer to data model
 *
 * Populates tmux session flags (e.g. running, attached, ctrl window) for each
 * known FPS instance by parsing tmux list-windows output.
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
     * If not, tmux is definitely not running, avoid popen fork. */
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

    /* Rate limit popen("tmux ...") to at most once every 3.0 seconds */
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed = (double) (now.tv_sec - s_last_tmux_scan.tv_sec) +
                     (double) (now.tv_nsec - s_last_tmux_scan.tv_nsec) * 1e-9;

    if (s_last_tmux_scan.tv_sec != 0 && elapsed < 3.0)
    {
        /* Apply cached tmux flags */
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
        return;
    }

    s_last_tmux_scan = now;
    s_tmux_cache_cnt = 0;

    FILE *fp = popen(
        "tmux list-windows -a -F \"#{session_name}:#{window_name}\" </dev/null 2>/dev/null", "r");
    if (fp == NULL)
    {
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

    pclose(fp);
}
