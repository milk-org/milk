// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_cmdfile.c
 * @brief The private directory of translated command files, shared by the milk-cli bash co-process and
 * the `milktranslate` builtin (part of the thin bash client).
 */

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include <dirent.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "milk_bash_engine.h"

static char  g_cmddir[256]  = { '\0' }; /* private dir holding the command files */
static pid_t g_cmddir_owner = 0;        /* forked children must not remove the parent's dir */

/** rm_dir_files() - remove the files in directory @dir, then the directory. */
static void rm_dir_files(const char *dir)
{
    DIR *dd = opendir(dir);
    if (dd != NULL)
    {
        struct dirent *de;
        while ((de = readdir(dd)) != NULL)
        {
            if (de->d_name[0] != '.')
            {
                unlinkat(dirfd(dd), de->d_name, 0);
            }
        }
        closedir(dd);
    }
    rmdir(dir);
}

/** sweep_stale_cmddirs() - remove milk-engine-<pid>-* directories of processes that are gone. */
static void sweep_stale_cmddirs(const char *base)
{
    DIR *bd = opendir(base);
    if (bd == NULL)
    {
        return;
    }
    struct dirent *de;
    while ((de = readdir(bd)) != NULL)
    {
        long pid = 0;
        if (sscanf(de->d_name, "milk-engine-%ld-", &pid) == 1 && pid > 0 &&
            kill((pid_t) pid, 0) != 0 && errno == ESRCH)
        {
            char path[512];
            snprintf(path, sizeof(path), "%s/%s", base, de->d_name);
            rm_dir_files(path);
        }
    }
    closedir(bd);
}

/** cmddir_atexit() - remove the command directory when the process exits without shutdown. */
static void cmddir_atexit(void)
{
    if (g_cmddir[0] != '\0' && getpid() == g_cmddir_owner)
    {
        rm_dir_files(g_cmddir);
        g_cmddir[0] = '\0';
    }
}

/** make_cmddir() - create the private directory (once per process). */
static int make_cmddir(void)
{
    static int atexit_done = 0;
    if (g_cmddir[0] != '\0')
    {
        return 0;
    }
    const char *base = getenv("XDG_RUNTIME_DIR");
    if (base == NULL || access(base, W_OK) != 0)
    {
        base = (access("/dev/shm", W_OK) == 0) ? "/dev/shm" : getenv("TMPDIR");
    }
    if (base == NULL || base[0] == '\0')
    {
        base = "/tmp";
    }
    sweep_stale_cmddirs(base);
    snprintf(g_cmddir, sizeof(g_cmddir), "%s/milk-engine-%ld-XXXXXX", base, (long) getpid());
    if (mkdtemp(g_cmddir) == NULL)
    {
        g_cmddir[0] = '\0';
        return -1;
    }
    g_cmddir_owner = getpid();
    if (!atexit_done)
    {
        atexit(cmddir_atexit);
        atexit_done = 1;
    }
    return 0;
}

int milk_bash_cmdfile_path(const char *label, char *out, size_t outsz)
{
    if (make_cmddir() != 0)
    {
        return -1;
    }
    snprintf(out, outsz, "%s/%s", g_cmddir, label);
    return 0;
}

int milk_bash_cmdfile_owned(const char *path)
{
    size_t n = strlen(g_cmddir);
    return n > 0 && strncmp(path, g_cmddir, n) == 0 && path[n] == '/' &&
           strstr(path + n, "/..") == NULL;
}

void milk_bash_cmdfile_cleanup(void)
{
    if (g_cmddir[0] != '\0')
    {
        rm_dir_files(g_cmddir);
        g_cmddir[0] = '\0';
    }
}
