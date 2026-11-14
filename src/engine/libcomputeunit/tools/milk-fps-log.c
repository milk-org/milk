// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milk-fps-log.c
 * @brief Show the output of a unit: the last lines of its run.out or conf.out, optionally following it
 */

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "fps.h"
#include "fps_unit_state.h"

#define FL_DESC "show the output of a unit (its run.out or conf.out)"
#define FL_DESC_LONG                                                                           \
    "Print the last lines a unit wrote: its run process by default, or its conf process.\n"    \
    "Units started in the background write to <dir>/<name>.fps.d/{run,conf}.out, rotated to\n" \
    ".1 at each start and above MILK_UNIT_OUT_MAX (default 1 MiB). With -f, keep printing\n"   \
    "what the unit writes, across rotations, until interrupted."

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, FL_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%s-f%s] [%s-n N%s] %s<fpsname>%s [%srun%s|%sconf%s]\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "", mh_color ? MH_OPT : "",
           mh_color ? MH_RST : "", mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "", mh_color ? MH_ARG : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", FL_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-f, --follow", mh_color ? MH_RST : "",
           "Keep printing new output");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-n, --lines N", mh_color ? MH_RST : "",
           "Last N lines (default 20, 0: all)");
    printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    const char *see_also[] = { "milk-fps-runstart:start the run loop of a unit",
                               "milk-fps-info:inspect an FPS, --env for its environment" };
    milk_help_see_also(see_also, 2, mh_color);
}

/* Print what @fd holds from @from on; return the new end */
static off_t copy_from(int fd, off_t from)
{
    char    buf[65536];
    ssize_t n;

    while ((n = pread(fd, buf, sizeof(buf), from)) > 0)
    {
        fwrite(buf, 1, (size_t) n, stdout);
        from += n;
    }
    fflush(stdout);
    return from;
}

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv, FL_DESC, FL_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], action == MH_ACTION_HELP);
        return 0;
    }

    const char   *name   = NULL;
    FPS_UNIT_ROLE role   = FPS_UNIT_RUN;
    long          nlines = 20;
    int           follow = 0;
    char          path[PATH_MAX];

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--follow") == 0)
        {
            follow = 1;
        }
        else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--lines") == 0) && i + 1 < argc)
        {
            nlines = atol(argv[++i]);
        }
        else if (name != NULL && (strcmp(argv[i], "run") == 0 || strcmp(argv[i], "conf") == 0))
        {
            role = (argv[i][0] == 'c') ? FPS_UNIT_CONF : FPS_UNIT_RUN;
        }
        else if (name == NULL && argv[i][0] != '-')
        {
            name = argv[i];
        }
        else
        {
            fprintf(stderr, "%s: unexpected argument '%s' (-h for help)\n", argv[0], argv[i]);
            return 2;
        }
    }
    if (name == NULL)
    {
        fprintf(stderr, "%s: missing <fpsname> (-h for help)\n", argv[0]);
        return 2;
    }
    if (fps_unit_role_path(name, role, "out", path, sizeof(path)) != 0)
    {
        fprintf(stderr, "%s: name too long\n", name);
        return 1;
    }

    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0 && !follow)
    {
        fprintf(stderr, "%s: no %s output: %s\n", name, fps_unit_role_name(role), strerror(errno));
        return 1;
    }
    off_t pos = 0;
    if (fd >= 0)
    {
        off_t end = lseek(fd, 0, SEEK_END);
        pos       = copy_from(fd, fps_unit_tail_offset(fd, end, nlines));
    }
    while (follow)
    {
        struct timespec t = { 0, 200000000L };
        struct stat     cur;
        struct stat     open_st;
        nanosleep(&t, NULL);
        // a start or the unit itself rotated the file: continue with the new one
        if (stat(path, &cur) == 0 &&
            (fd < 0 ||
             (fstat(fd, &open_st) == 0 && (cur.st_ino != open_st.st_ino || cur.st_size < pos))))
        {
            if (fd >= 0)
            {
                copy_from(fd, pos); // what was written before the rotation
                close(fd);
            }
            fd  = open(path, O_RDONLY | O_CLOEXEC);
            pos = 0;
        }
        if (fd >= 0)
        {
            pos = copy_from(fd, pos);
        }
    }
    if (fd >= 0)
    {
        close(fd);
    }
    return 0;
}
