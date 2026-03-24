/**
 * @file CLIcore_script.c
 * @brief CLI scripting engine — variables, FPS access,
 *        arithmetic, flow control, user functions
 *
 * Implements bash-style scripting constructs for the
 * milk CLI:
 * - Variable assignment (VAR=val), expansion ($VAR)
 * - FPS parameter read (@fpsname.param)
 * - FPS parameter write (fpsset)
 * - Arithmetic $(( expr ))
 * - Flow control: if/then/else/fi,
 *   while/do/done, for/do/done
 * - User-defined functions:
 *   function name { body }
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>

#include "CLIcore.h"
#include "CLIcore_script.h"
#include "CLIcore_UI_execute.h"

#include <sys/mman.h>

/* processinfo functions — linked via milkprocessinfo */
extern PROCESSINFO *processinfo_shm_link(
    const char *pname, int *fd);
extern errno_t processinfo_procdirname(
    char *procdname);

/* ============================================================
 *  CLI Variable Storage
 * ============================================================
 */

CLI_VAR cli_vars[CLI_MAX_VARS];
int     cli_last_retval = 0;

/* ---- set flags ---- */
int     cli_flag_errexit = 0;  /* set -e */
int     cli_flag_xtrace  = 0;  /* set -x */

/* ---- Trap Handlers ---- */
CLI_TRAP cli_traps[CLI_TRAP_MAXSIGS];

/**
 * @brief Map signal name to number
 *
 * @param name  Signal name (EXIT, INT, etc.)
 * @return signal number, or 0 for EXIT
 */
int cli_trap_signum(const char *name)
{
    if(strcasecmp(name, "EXIT") == 0)
    {
        return 0;
    }
    if(strcasecmp(name, "INT") == 0)
    {
        return SIGINT;
    }
    if(strcasecmp(name, "TERM") == 0)
    {
        return SIGTERM;
    }
    if(strcasecmp(name, "HUP") == 0)
    {
        return SIGHUP;
    }
    if(strcasecmp(name, "USR1") == 0)
    {
        return SIGUSR1;
    }
    if(strcasecmp(name, "USR2") == 0)
    {
        return SIGUSR2;
    }
    if(strcasecmp(name, "ERR") == 0)
    {
        return -1; /* pseudo-signal */
    }
    return (int) strtol(name, NULL, 0);
}

/* ---- Array Storage ---- */
CLI_ARRAY cli_arrays[CLI_MAX_ARRAYS];

/* ---- Associative Array Storage ---- */
CLI_ASSOC_ARRAY cli_assoc[CLI_MAX_ASSOC];

/* Local variable scoping stack for functions */
// externs are defined in CLIcore_script.h
CLI_LOCAL_SHADOW cli_local_shadows[CLI_MAX_LOCAL_DEPTH][CLI_MAX_LOCALS_PER_FUNC];
int cli_local_shadow_count[CLI_MAX_LOCAL_DEPTH];
int cli_local_depth = 0;

/* ---- Source Location Tracking ---- */
CLI_SRC_LOC cli_src_stack[CLI_SRC_STACK_DEPTH];
int         cli_src_depth = 0;

/**
 * @brief Print source location stack trace
 *
 * Called on error to show where in the
 * source file hierarchy the error occurred.
 */
void cli_print_source_trace(void)
{
    if(cli_src_depth <= 0)
    {
        return;
    }
    fprintf(stderr,
            "\033[2mStack trace:\033[0m\n");
    for(int i = cli_src_depth - 1;
        i >= 0; i--)
    {
        fprintf(stderr,
                "  \033[2m%s:%d\033[0m\n",
                cli_src_stack[i].file,
                cli_src_stack[i].line);
    }
}

/**
 * @brief Look up a CLI variable by name
 *
 * @param name  Variable name
 * @return pointer to value string, or NULL
 */
const char *cli_var_get(const char *name)
{
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(cli_vars[i].used
                && strcmp(cli_vars[i].name, name)
                == 0)
        {
            return cli_vars[i].val;
        }
    }
    return NULL;
}

/**
 * @brief Export CLI variables to environment (for wordexp and shell sync)
 */
void cli_export_vars_to_env(void)
{
    /* Export scalar variables */
    for(int i = 0; i < CLI_MAX_VARS; i++)
    {
        if(cli_vars[i].used)
        {
            setenv(cli_vars[i].name, cli_vars[i].val, 1);
        }
    }
}

/**
 * @brief Unified variable lookup: CLI vars,
 *        then special vars ($?), then env vars
 *
 * @param name  Variable name
 * @return pointer to value string, or NULL
 */
const char *cli_var_lookup(const char *name)
{
    static char retbuf[32];

    /* $? — last return value */
    if(strcmp(name, "?") == 0)
    {
        snprintf(retbuf, sizeof(retbuf),
                 "%d", cli_last_retval);
        return retbuf;
    }

    /* $MCLIFIFO — current command FIFO path */
    if(strcmp(name, "MCLIFIFO") == 0)
    {
        if(data.fifoON == 1)
        {
            return data.fifoname;
        }
        return "";
    }

    /* $PROCINFO_NCPU — online CPUs */
    if(strcmp(name, "PROCINFO_NCPU") == 0)
    {
        long ncpu = sysconf(
            _SC_NPROCESSORS_ONLN);
        snprintf(retbuf, sizeof(retbuf),
                 "%ld", ncpu);
        return retbuf;
    }

    /* $PROCINFO_NPROC — active procs */
    if(strcmp(name, "PROCINFO_NPROC") == 0)
    {
        int cnt = 0;
        if(pinfolist != NULL)
        {
            for(int pi = 0;
                pi < PROCESSINFOLISTSIZE;
                pi++)
            {
                if(pinfolist->active[pi])
                {
                    cnt++;
                }
            }
        }
        snprintf(retbuf, sizeof(retbuf),
                 "%d", cnt);
        return retbuf;
    }

    /* CLI variable */
    const char *v = cli_var_get(name);
    if(v != NULL)
    {
        return v;
    }

    /* stream.prop — shared memory
     * stream metadata */
    {
        const char *dot =
            strchr(name, '.');
        if(dot != NULL)
        {
            char sname[128];
            int sn = (int)(dot - name);
            if(sn >= (int) sizeof(sname))
            {
                sn =
                    (int) sizeof(sname)
                    - 1;
            }
            memcpy(sname, name,
                   (size_t) sn);
            sname[sn] = '\0';
            const char *prop = dot + 1;

            /* Try stream SHM */
            char spath[256];
            snprintf(spath,
                     sizeof(spath),
                     "/dev/shm/%s"
                     ".im.shm",
                     sname);
            if(access(spath,
                      F_OK) == 0)
            {
                IMAGE img;
                if(ImageStreamIO_read_sharedmem_image_toIMAGE(
                    sname, &img) == 0)
                {
                    if(strcmp(prop,
                        "xsize") == 0
                       && img.md != NULL
                       && img.md->naxis
                       >= 1)
                    {
                        snprintf(
                            retbuf,
                            sizeof(
                                retbuf),
                            "%u",
                            img.md
                            ->size[0]);
                        return retbuf;
                    }
                    if(strcmp(prop,
                        "ysize") == 0
                       && img.md != NULL
                       && img.md->naxis
                       >= 2)
                    {
                        snprintf(
                            retbuf,
                            sizeof(
                                retbuf),
                            "%u",
                            img.md
                            ->size[1]);
                        return retbuf;
                    }
                    if(strcmp(prop,
                        "zsize") == 0
                       && img.md != NULL
                       && img.md->naxis
                       >= 3)
                    {
                        snprintf(
                            retbuf,
                            sizeof(
                                retbuf),
                            "%u",
                            img.md
                            ->size[2]);
                        return retbuf;
                    }
                    if(strcmp(prop,
                        "type") == 0
                       && img.md != NULL)
                    {
                        snprintf(
                            retbuf,
                            sizeof(
                                retbuf),
                            "%d",
                            (int)
                            img.md
                            ->datatype);
                        return retbuf;
                    }
                    if(strcmp(prop,
                        "cnt0") == 0
                       && img.md != NULL)
                    {
                        snprintf(
                            retbuf,
                            sizeof(
                                retbuf),
                            "%lu",
                            (unsigned
                            long)
                            img.md
                            ->cnt0);
                        return retbuf;
                    }
                    if(strcmp(prop,
                        "cnt1") == 0
                       && img.md != NULL)
                    {
                        snprintf(
                            retbuf,
                            sizeof(
                                retbuf),
                            "%lu",
                            (unsigned
                            long)
                            img.md
                            ->cnt1);
                        return retbuf;
                    }
                    if(strcmp(prop,
                        "naxis") == 0
                       && img.md != NULL)
                    {
                        snprintf(
                            retbuf,
                            sizeof(
                                retbuf),
                            "%d",
                            (int)
                            img.md
                            ->naxis);
                        return retbuf;
                    }
                }
            }

            /* Try FPS SHM */
            char fpath[256];
            snprintf(fpath,
                     sizeof(fpath),
                     "/dev/shm/"
                     "fps.%s.shm",
                     sname);
            if(access(fpath,
                      F_OK) == 0)
            {
                if(strcmp(prop,
                    "status") == 0)
                {
                    snprintf(
                        retbuf,
                        sizeof(retbuf),
                        "1");
                    return retbuf;
                }
            }
        }
    }

    /* Fall through to environment */
    return getenv(name);
}


/* ============================================================
 *  Arithmetic Expansion — $(( expr ))
 * ============================================================
 * Functions moved to CLIcore_script_expand.c
 */


