/**
 * @file    milk-logshim-ctrl.c
 * @brief   Control a streamFITSlog logging process
 *
 * Replaces the five thin bash wrapper scripts:
 *   milk-logshim       (start:  delegates to milk-streamFITSlog pstart)
 *   milk-logshimkill   (kill:   writes FPS FIFO commands)
 *   milk-logshimon     (on:     writes FPS FIFO commands)
 *   milk-logshimoff    (off:    writes FPS FIFO commands)
 *   milk-logshimstat   (stat:   reads FPS SHM params)
 *
 * All actions that require running/stopping a process write
 * directly to the milkFITSlogger FIFO, matching the protocol
 * used by milk-streamFITSlog.
 *
 * The 'start' action delegates to milk-streamFITSlog (kept as
 * the orchestration layer for tmux + milk-cli setup).
 *
 * Usage:
 *   milk-logshim-ctrl <action> <stream> [options]
 *
 * Actions:
 *   start <stream> <blocksize> <dir>  Start logging process
 *   on    <stream>                    Enable saving
 *   off   <stream>                    Disable saving
 *   offc  <stream>                    Disable after cube done
 *   kill  <stream>                    Kill process + tmux
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

#define LSC_DESC "control a streamFITSlog logging process"

#define LSC_DESC_LONG                                               \
    "Unified control tool for shared memory stream FITS logging.\n" \
    "Replaces the five milk-logshim* shell wrappers.\n"             \
    "\n"                                                            \
    "The 'start' action delegates to milk-streamFITSlog for\n"      \
    "tmux session and FPS instance setup.\n"                        \
    "\n"                                                            \
    "All other actions write commands directly to the\n"            \
    "milkFITSlogger FIFO ($MILK_SHM_DIR/milkFITSlogger.fifo),\n"    \
    "matching the protocol used by milk-streamFITSlog."

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
           "Stop run, remove FPS instance, kill tmux");
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
 * fifo_path() - Construct the milkFITSlogger FIFO path
 * @buf:    output buffer
 * @bufsz:  buffer size
 *
 * The FIFO is at $MILK_SHM_DIR/milkFITSlogger.fifo,
 * matching the path used by milk-streamFITSlog.
 */
static void fifo_path(char *buf, size_t bufsz)
{
    snprintf(buf, bufsz, "%s/milkFITSlogger.fifo", get_shmdir());
}

/**
 * fifo_write() - Write a command line to the FPS FIFO
 * @fifo:   path to the FIFO file
 * @cmd:    null-terminated command string (no trailing newline needed)
 *
 * Opens the FIFO in non-blocking write-only mode, writes the
 * command, then closes. Returns 0 on success, 1 on error.
 */
static int fifo_write(const char *fifo, const char *cmd)
{
    /* Check FIFO exists */
    struct stat st;
    if (stat(fifo, &st) != 0 || !S_ISFIFO(st.st_mode))
    {
        fprintf(stderr,
                "\033[1;31mERROR\033[0m: FIFO '%s' not found.\n"
                "  Is the logging process running?\n"
                "  Start it first with: milk-logshim-ctrl"
                " start <stream> ...\n",
                fifo);
        return 1;
    }

    /*
     * Open in blocking mode — the reader (milk-fpsCTRL) must be
     * present to consume the write end.  Using O_WRONLY without
     * O_NONBLOCK blocks until the reader opens its end.  We open
     * with O_NONBLOCK first to detect a missing reader immediately.
     */
    int fd = open(fifo, O_WRONLY | O_NONBLOCK);
    if (fd < 0)
    {
        if (errno == ENXIO)
        {
            fprintf(stderr,
                    "\033[1;31mERROR\033[0m: No reader on"
                    " FIFO '%s' — is milk-fpsCTRL running?\n",
                    fifo);
        }
        else
        {
            fprintf(stderr, "\033[1;31mERROR\033[0m: open('%s'): %s\n", fifo, strerror(errno));
        }
        return 1;
    }

    /* Write command + newline */
    char line[512];
    int  n = snprintf(line, sizeof(line), "%s\n", cmd);

    ssize_t written = write(fd, line, (size_t) n);
    close(fd);

    if (written < 0)
    {
        fprintf(stderr, "\033[1;31mERROR\033[0m: write to FIFO: %s\n", strerror(errno));
        return 1;
    }

    return 0;
}

/**
 * action_on() - Enable FITS saving for a stream
 * @stream: stream name
 *
 * Sends: setval streamFITSlog-<stream>.saveON ON
 */
static int action_on(const char *stream)
{
    char fifo[512];
    fifo_path(fifo, sizeof(fifo));

    char cmd[256];
    snprintf(cmd, sizeof(cmd), "setval " FPS_PREFIX "%s.saveON ON", stream);

    printf("logshim-ctrl on  %s: enabling saving\n", stream);
    return fifo_write(fifo, cmd);
}

/**
 * action_off() - Disable FITS saving for a stream
 * @stream: stream name
 *
 * Sends: setval streamFITSlog-<stream>.saveON OFF
 */
static int action_off(const char *stream)
{
    char fifo[512];
    fifo_path(fifo, sizeof(fifo));

    char cmd[256];
    snprintf(cmd, sizeof(cmd), "setval " FPS_PREFIX "%s.saveON OFF", stream);

    printf("logshim-ctrl off %s: disabling saving\n", stream);
    return fifo_write(fifo, cmd);
}

/**
 * action_offc() - Disable saving after current cube completes
 * @stream: stream name
 *
 * Sends: setval streamFITSlog-<stream>.lastcubeON ON
 * (lastcubeON tells the logger to write one more cube then stop)
 */
static int action_offc(const char *stream)
{
    char fifo[512];
    fifo_path(fifo, sizeof(fifo));

    char cmd[256];
    snprintf(cmd, sizeof(cmd), "setval " FPS_PREFIX "%s.lastcubeON ON", stream);

    printf("logshim-ctrl offc %s:"
           " will stop after current cube\n",
           stream);
    return fifo_write(fifo, cmd);
}

/**
 * action_kill() - Stop the logging run, remove FPS, kill tmux
 * @stream: stream name
 *
 * Sends (in order):
 *   runstop  streamFITSlog-<stream>
 *   confstop streamFITSlog-<stream>
 *   tmuxstop streamFITSlog-<stream>
 *   fpsrm    streamFITSlog-<stream>
 *   rescan
 *
 * Also removes the two log buffer SHM streams.
 */
static int action_kill(const char *stream)
{
    char fifo[512];
    fifo_path(fifo, sizeof(fifo));

    char cmd[256];
    int  errors = 0;

    printf("logshim-ctrl kill %s: stopping...\n", stream);

    snprintf(cmd, sizeof(cmd), "runstop " FPS_PREFIX "%s", stream);
    errors += fifo_write(fifo, cmd);

    snprintf(cmd, sizeof(cmd), "confstop " FPS_PREFIX "%s", stream);
    errors += fifo_write(fifo, cmd);

    snprintf(cmd, sizeof(cmd), "tmuxstop " FPS_PREFIX "%s", stream);
    errors += fifo_write(fifo, cmd);

    snprintf(cmd, sizeof(cmd), "fpsrm " FPS_PREFIX "%s", stream);
    errors += fifo_write(fifo, cmd);

    fifo_write(fifo, "rescan"); /* best-effort */

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

    return errors > 0 ? 1 : 0;
}

/**
 * action_stat() - Show saveON status from FPS SHM
 * @stream: stream name
 *
 * Reads the FPS parameter file directly from SHM:
 *   $MILK_SHM_DIR/fps.streamFITSlog-<stream>.shm
 *
 * Falls back to listing the FIFO existence if SHM is not found.
 */
static int action_stat(const char *stream)
{
    const char *shmdir = get_shmdir();

    /* FPS SHM file: fps.streamFITSlog-<stream>.shm */
    char shm_path[512];
    snprintf(shm_path, sizeof(shm_path), "%s/fps." FPS_PREFIX "%s.shm", shmdir, stream);

    /* FIFO path */
    char fifo[512];
    fifo_path(fifo, sizeof(fifo));

    struct stat st;
    int         fps_alive  = (stat(shm_path, &st) == 0);
    int         fifo_alive = (stat(fifo, &st) == 0 && S_ISFIFO(st.st_mode));

    printf("\n  Stream: %s\033[1m%s\033[0m\n\n", "", stream);

    /* FPS instance status */
    printf("  FPS SHM  : %s%s\033[0m\n", fps_alive ? "\033[1;32m[ LIVE ] " : "\033[1;31m[ABSENT] ",
           shm_path);

    /* FIFO status */
    printf("  FIFO     : %s%s\033[0m\n", fifo_alive ? "\033[1;32m[ LIVE ] " : "\033[1;31m[ABSENT] ",
           fifo);

    /* If FPS SHM exists, parse saveON by scanning for key */
    if (fps_alive)
    {
        /* We can't include milkfpsStandalone here (standalone binary
         * must not link CLIcore).  Read raw bytes from the SHM file
         * and search for the saveON parameter name.
         *
         * The FPS SHM layout stores parameter name strings followed
         * by a value area.  We do a simple byte scan for ".saveON"
         * in the first 4 MB of the file, then read the adjacent
         * ON/OFF bytes that milk-fpsCTRL / milk-streamFITSlog write.
         *
         * For a fully robust implementation, link milkfpsStandalone
         * and call fps_paramname_to_indexparam() — but this avoids
         * the CLIcore dependency for a standalone binary.
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
 * Delegates to milk-streamFITSlog which handles tmux +
 * milk-cli orchestration.
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
