// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk-logshim-ctrl.c
 * @brief   Control a streamFITSlog logging process
 *
 * Replaces the five thin bash wrapper scripts:
 *   milk-logshim       (start:  delegates to milk-streamFITSlog pstart)
 *   milk-logshimkill   (kill:   milk-fps-rm -f)
 *   milk-logshimon     (on:     milk-fps-set)
 *   milk-logshimoff    (off:    milk-fps-set)
 *   milk-logshimstat   (stat:   reads FPS SHM params)
 *
 * on/off/offc/kill run the same milk-fps-* tools as milk-streamFITSlog,
 * without a shell (the fpsCTRL FIFO they used to write is retired).
 *
 * The 'start' action delegates to milk-streamFITSlog, which makes
 * and starts the logger unit (milk-fpsexec-mem-streamFITSlog).
 *
 * Usage:
 *   milk-logshim-ctrl <action> <stream> [options]
 *
 * Actions:
 *   start <stream> <blocksize> <dir>  Start logging process
 *   on    <stream>                    Enable saving
 *   off   <stream>                    Disable saving
 *   offc  <stream>                    Disable after cube done
 *   kill  <stream>                    Stop the logger, remove its FPS
 *   stat  <stream>                    Show FPS parameter status
 */

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "milk_help.h"
#include "milk_spawn.h"

#define LSC_DESC "control a streamFITSlog logging process"

#define LSC_DESC_LONG                                               \
    "Unified control tool for shared memory stream FITS logging.\n" \
    "Replaces the five milk-logshim* shell wrappers.\n"             \
    "\n"                                                            \
    "The 'start' action delegates to milk-streamFITSlog, which\n"   \
    "makes and starts the logger unit.\n"                           \
    "\n"                                                            \
    "on/off/offc set FPS parameters with milk-fps-set; kill runs\n" \
    "milk-fps-rm -f, which stops the unit and removes its FPS."

/** FPS name prefix for streamFITSlog instances */
#define FPS_PREFIX "streamFITSlog-"
/** SHM filename suffix */
#define SHM_SUFFIX ".im.shm"

/**
 * print_help() - Print usage information
 * @progname: argv[0]
 * @mh_color: non-zero for ANSI color output
 */
static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, LSC_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s %s<action>%s %s<stream>%s [%soptions%s]\n\n", mh_color ? MH_CMD : "", progname,
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "", mh_color ? MH_OPT : "",
           mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", LSC_DESC_LONG);
    milk_help_section("Actions", mh_color);
    printf("  %s%-38s%s %s\n", mh_color ? MH_ARG : "", "start <stream> <blocksize> <dir>",
           mh_color ? MH_RST : "", "Start FPS logging process (via milk-streamFITSlog)");
    printf("  %s%-38s%s %s\n", mh_color ? MH_ARG : "", "on    <stream>", mh_color ? MH_RST : "",
           "Enable saving to disk");
    printf("  %s%-38s%s %s\n", mh_color ? MH_ARG : "", "off   <stream>", mh_color ? MH_RST : "",
           "Disable saving to disk");
    printf("  %s%-38s%s %s\n", mh_color ? MH_ARG : "", "offc  <stream>", mh_color ? MH_RST : "",
           "Disable saving after current cube completes");
    printf("  %s%-38s%s %s\n", mh_color ? MH_ARG : "", "kill  <stream>", mh_color ? MH_RST : "",
           "Stop the logger, remove its FPS instance");
    printf("  %s%-38s%s %s\n\n", mh_color ? MH_ARG : "", "stat  <stream>", mh_color ? MH_RST : "",
           "Show saveON status from FPS shared memory");
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-c <cpuset>", mh_color ? MH_RST : "",
           "CPU set for 'start' (forwarded to milk-streamFITSlog)");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h, --help", mh_color ? MH_RST : "",
           "Show this help and exit");
    printf("  %s%-25s%s %s\n", mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n\n", mh_color ? MH_OPT : "", "-hm, --help-mono", mh_color ? MH_RST : "",
           "Full help, no ANSI color");
    milk_help_section("Examples", mh_color);
    printf("  %s$ milk-logshim-ctrl%s %sstart ircam0 10000 /mnt/log%s\n", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-logshim-ctrl%s %son   ircam0%s\n", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-logshim-ctrl%s %soff  ircam0%s\n", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-logshim-ctrl%s %soffc ircam0%s\n", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-logshim-ctrl%s %skill ircam0%s\n", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-logshim-ctrl%s %sstat ircam0%s\n\n", mh_color ? MH_CMD : "",
           mh_color ? MH_RST : "", mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    const char *see_also[] = { "milk-streamFITSlog:log stream frames to FITS files",
                               "milk-logshim:launch the logging shim daemon",
                               "milk-fpsCTRL:launch the FPS dashboard TUI" };
    milk_help_see_also(see_also, 3, mh_color);
}

/**
 * get_shmdir() - Return the shared memory directory path
 *
 * Checks MILK_SHM_DIR env, then /milk/shm, then /dev/shm.
 */
static const char *get_shmdir(void)
{
    const char *env = getenv("MILK_SHM_DIR");
    if (env != NULL)
    {
        return env;
    }

    static const char fallback[] = "/milk/shm";
    struct stat       st;

    if (stat(fallback, &st) == 0 && S_ISDIR(st.st_mode))
    {
        return fallback;
    }
    return "/dev/shm";
}

/**
 * run_tool() - run @argv (PATH lookup, no shell) with our stdout/stderr; return its exit status.
 */
static int run_tool(const char *const argv[])
{
    int rc = milkspawn_run(argv, STDOUT_FILENO, STDERR_FILENO);
    if (rc < 0)
    {
        fprintf(stderr, "\033[1;31mERROR\033[0m: cannot run %s: %s\n", argv[0], strerror(errno));
        return 1;
    }
    return rc;
}

/**
 * fps_set() - milk-fps-set streamFITSlog-<stream>.<key> <val>
 */
static int fps_set(const char *stream, const char *key, const char *val)
{
    char param[512];
    snprintf(param, sizeof(param), FPS_PREFIX "%s.%s", stream, key);
    const char *const argv[] = { "milk-fps-set", param, val, NULL };
    return run_tool(argv);
}

/**
 * action_on() - Enable FITS saving for a stream
 * @stream: stream name
 *
 * Runs: milk-fps-set streamFITSlog-<stream>.saveON ON
 */
static int action_on(const char *stream)
{
    printf("logshim-ctrl on  %s: enabling saving\n", stream);
    return fps_set(stream, "saveON", "ON");
}

/**
 * action_off() - Disable FITS saving for a stream
 * @stream: stream name
 *
 * Runs: milk-fps-set streamFITSlog-<stream>.saveON OFF
 */
static int action_off(const char *stream)
{
    printf("logshim-ctrl off %s: disabling saving\n", stream);
    return fps_set(stream, "saveON", "OFF");
}

/**
 * action_offc() - Disable saving after current cube completes
 * @stream: stream name
 *
 * Runs: milk-fps-set streamFITSlog-<stream>.lastcubeON ON
 * (lastcubeON tells the logger to write one more cube then stop)
 */
static int action_offc(const char *stream)
{
    printf("logshim-ctrl offc %s:"
           " will stop after current cube\n",
           stream);
    return fps_set(stream, "lastcubeON", "ON");
}

/**
 * action_kill() - Stop the logger unit and remove its FPS
 * @stream: stream name
 *
 * Runs milk-fps-rm -f -x streamFITSlog-<stream> (stops the CONF and RUN
 * processes, removes the FPS and its unit directory).
 *
 * Also removes the two log buffer SHM streams.
 */
static int action_kill(const char *stream)
{
    printf("logshim-ctrl kill %s: stopping...\n", stream);

    char name[512];
    snprintf(name, sizeof(name), FPS_PREFIX "%s", stream);
    const char *const rm_argv[] = { "milk-fps-rm", "-f", "-x", name, NULL };
    int               rc        = run_tool(rm_argv);

    /* Remove log buffer streams */
    const char *shmdir = get_shmdir();
    char        buf0[512], buf1[512];
    snprintf(buf0, sizeof(buf0), "%s/%s_logbuff0%s", shmdir, stream, SHM_SUFFIX);
    snprintf(buf1, sizeof(buf1), "%s/%s_logbuff1%s", shmdir, stream, SHM_SUFFIX);

    if (unlink(buf0) == 0)
    {
        printf("  removed: %s_logbuff0\n", stream);
    }
    if (unlink(buf1) == 0)
    {
        printf("  removed: %s_logbuff1\n", stream);
    }

    return rc;
}

/**
 * action_stat() - Show saveON status from FPS SHM
 * @stream: stream name
 *
 * Reads the FPS parameter file directly from SHM:
 *   $MILK_SHM_DIR/streamFITSlog-<stream>.fps.shm
 */
static int action_stat(const char *stream)
{
    const char *shmdir = get_shmdir();

    /* FPS SHM file: streamFITSlog-<stream>.fps.shm (it was looked up as fps.<name>.shm: always ABSENT) */
    char shm_path[512];
    snprintf(shm_path, sizeof(shm_path), "%s/" FPS_PREFIX "%s.fps.shm", shmdir, stream);

    struct stat st;
    int         fps_alive = (stat(shm_path, &st) == 0);

    printf("\n  Stream: %s\033[1m%s\033[0m\n\n", "", stream);

    /* FPS instance status */
    printf("  FPS SHM  : %s%s\033[0m\n", fps_alive ? "\033[1;32m[ LIVE ] " : "\033[1;31m[ABSENT] ",
           shm_path);


    /* If FPS SHM exists, parse saveON by scanning for key */
    if (fps_alive)
    {
        /* We can't link CLIcore here (standalone binary
         * must not link it).  Read raw bytes from the SHM file
         * and search for the saveON parameter name.
         *
         * The FPS SHM layout stores parameter name strings followed
         * by a value area.  We do a simple byte scan for ".saveON"
         * in the first 4 MB of the file, then read the adjacent
         * ON/OFF bytes that milk-fpsCTRL / milk-streamFITSlog write.
         *
         * For a fully robust implementation, link milkfps
         * and call fps_paramname_to_indexparam() — but this avoids
         * the extra dependency for a standalone binary.
         */
        FILE *fp = fopen(shm_path, "rb");
        if (fp != NULL)
        {
            /* Scan up to 4 MB */
            static unsigned char buf[4 * 1024 * 1024];
            size_t               nr = fread(buf, 1, sizeof(buf), fp);
            fclose(fp);

            const char *needle = ".saveON";
            size_t      nlen   = strlen(needle);
            int         found  = 0;

            for (size_t i = 0; i + nlen + 10 < nr; i++)
            {
                if (memcmp(&buf[i], needle, nlen) == 0)
                {
                    /* The ON/OFF flag is stored as a uint8
                     * in the val.u8 union member.  Walk
                     * forward past the null-terminator of
                     * the name to find the first non-zero
                     * byte that is 0 or 1. */
                    for (size_t j = i + nlen; j < i + nlen + 64 && j < nr; j++)
                    {
                        if (buf[j] == 0 || buf[j] == 1)
                        {
                            printf("  saveON   : %s%s\033[0m\n\n",
                                   buf[j] ? "\033[1;32mON " : "\033[1;31mOFF", "");
                            found = 1;
                            break;
                        }
                    }
                    if (found)
                    {
                        break;
                    }
                }
            } // for i

            if (!found)
            {
                printf("  saveON   : \033[33m(cannot parse)\033[0m\n\n");
            }
        } // if (fp)
    }
    else
    {
        printf("\n  Start logging first with:\n"
               "    milk-logshim-ctrl start %s <blocksize> <dir>\n\n",
               stream);
    }

    return 0;
}

/**
 * action_start() - Start a streamFITSlog process
 * @stream:    stream name
 * @blocksize: cube size
 * @dir:       log directory
 * @cpuset:    CPU set (or NULL for default)
 *
 * Delegates to milk-streamFITSlog, which makes and starts the
 * logger unit.
 */
static int action_start(const char *stream,
                        const char *blocksize,
                        const char *dir,
                        const char *cpuset)
{
    /* Verify stream SHM exists */
    const char *shmdir = get_shmdir();
    char        shmfile[512];
    snprintf(shmfile, sizeof(shmfile), "%s/%s%s", shmdir, stream, SHM_SUFFIX);

    struct stat st;
    if (stat(shmfile, &st) != 0)
    {
        fprintf(stderr,
                "\n\033[1;31mERROR\033[0m:"
                " stream SHM '%s' not found.\n\n",
                shmfile);
        return 1;
    }

    /* Build milk-streamFITSlog command */
    char cmd[1024];
    if (cpuset != NULL)
    {
        snprintf(cmd, sizeof(cmd),
                 "milk-streamFITSlog -cset \"%s\" -D \"%s\""
                 " -z %s %s pstart && "
                 "milk-streamFITSlog %s on",
                 cpuset, dir, blocksize, stream, stream);
    }
    else
    {
        snprintf(cmd, sizeof(cmd),
                 "milk-streamFITSlog -D \"%s\" -z %s %s pstart &&"
                 " milk-streamFITSlog %s on",
                 dir, blocksize, stream, stream);
    }

    printf("logshim-ctrl start %s: launching...\n", stream);
    printf("  Executing: %s\n", cmd);
    return system(cmd) == 0 ? 0 : 1;
}

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv, LSC_DESC, LSC_DESC_LONG);

    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }

    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    if (argc < 3)
    {
        fprintf(stderr, "\n\033[1;31mERROR\033[0m:"
                        " expected <action> <stream> [options].\n\n");
        print_help(argv[0], 1);
        return 1;
    }

    const char *act    = argv[1];
    const char *stream = argv[2];

    /* Optional -c <cpuset> (only used by 'start') */
    const char *cpuset = NULL;
    for (int i = 3; i < argc - 1; i++)
    {
        if (strcmp(argv[i], "-c") == 0)
        {
            cpuset = argv[i + 1];
            i++;
        }
    }

    /* Dispatch */
    if (strcmp(act, "on") == 0)
    {
        if (argc < 3)
        {
            goto missing_stream;
        }
        return action_on(stream);
    }

    if (strcmp(act, "off") == 0)
    {
        if (argc < 3)
        {
            goto missing_stream;
        }
        return action_off(stream);
    }

    if (strcmp(act, "offc") == 0)
    {
        if (argc < 3)
        {
            goto missing_stream;
        }
        return action_offc(stream);
    }

    if (strcmp(act, "kill") == 0)
    {
        if (argc < 3)
        {
            goto missing_stream;
        }
        return action_kill(stream);
    }

    if (strcmp(act, "stat") == 0)
    {
        if (argc < 3)
        {
            goto missing_stream;
        }
        return action_stat(stream);
    }

    if (strcmp(act, "start") == 0)
    {
        if (argc < 5)
        {
            fprintf(stderr, "\n\033[1;31mERROR\033[0m:"
                            " 'start' requires <stream> <blocksize> <dir>.\n\n");
            print_help(argv[0], 1);
            return 1;
        }
        return action_start(stream, argv[3], argv[4], cpuset);
    }

    fprintf(stderr,
            "\n\033[1;31mERROR\033[0m:"
            " unknown action '%s'.\n\n",
            act);
    print_help(argv[0], 1);
    return 1;

missing_stream:
    fprintf(stderr,
            "\n\033[1;31mERROR\033[0m:"
            " action '%s' requires <stream>.\n\n",
            act);
    print_help(argv[0], 1);
    return 1;
}
