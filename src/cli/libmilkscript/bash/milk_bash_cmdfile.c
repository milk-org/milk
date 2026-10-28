// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk_bash_cmdfile.c
 * @brief Private directory of translated command files, shared by the milk-cli bash co-process
 *        and the `milktranslate` builtin (part of the thin bash client).
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
    if (g_cmddir[0] != '\0' && getpid() == g_cmddir_owner)
    {
        return 0;
    }
    g_cmddir[0]      = '\0';
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

/**
 * milk_bash_cmdfile_path - Build the path for a translated command file
 * @label: Base filename label (e.g. script basename or "milk")
 * @out:   Destination buffer for the full command file path
 * @outsz: Size of @out in bytes
 *
 * Ensures the process-private command directory exists and formats the path
 * `<cmddir>/<label>` into @out so bash can `source` translated code while
 * preserving meaningful script names in diagnostic messages.
 *
 * Return: 0 on success, -1 if the private directory could not be created
 */
int milk_bash_cmdfile_path(const char *label, char *out, size_t outsz)
{
    if (make_cmddir() != 0)
    {
        return -1;
    }
    snprintf(out, outsz, "%s/%s", g_cmddir, label);
    return 0;
}

/**
 * milk_bash_cmdfile_owned - Check whether a file path resides in our command directory
 * @path: Filesystem path to validate
 *
 * Verifies that @path starts with this process's private `g_cmddir` prefix and
 * contains no `..` traversal components, preventing accidental deletion or
 * treatment of user files as temporary translated command files.
 *
 * Return: 1 if @path is inside the private command directory, 0 otherwise
 */
int milk_bash_cmdfile_owned(const char *path)
{
    size_t n = strlen(g_cmddir);
    return n > 0 && strncmp(path, g_cmddir, n) == 0 && path[n] == '/' &&
           strstr(path + n, "/..") == NULL;
}

/**
 * milk_bash_cmdfile_cleanup - Remove the process-private translated command directory
 *
 * Unlinks all remaining translated command files in `g_cmddir` and removes the
 * directory itself during engine shutdown when called by the owning process.
 *
 * Return: void
 */
void milk_bash_cmdfile_cleanup(void)
{
    if (g_cmddir[0] != '\0' && getpid() == g_cmddir_owner)
    {
        rm_dir_files(g_cmddir);
        g_cmddir[0] = '\0';
    }
}
