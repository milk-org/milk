#include <stddef.h>
extern int cli_find_in_path(
    const char *cmd,
    char       *outpath,
    size_t     outsize);
extern int processinfo_procdirname(char *procdirname);
#include <stddef.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "CLIcore.h"
#include "CLIcore_script.h"
#include "milkscript.h"
#include "CLIcore_utils.h"
#include "CLIcore_memory.h"
#include "CLIcore_modules.h"
#include "CLIcore_UI_execute.h"
#include "CLIcore_checkargs.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <poll.h>

extern int cli_block_level;
extern int cli_break_flag;
extern int CLI_trap_enable;
extern int cli_cmd_delay_us;

/**
 * @brief Handler: extract basename from path.
 */
int cli_intercept_cmd_basename(const char *p)
{
    if(starts_with(p, "basename "))
    {
        p += 8;
        p = strip_ws(p);
        /* Find last / */
        const char *sl =
            strrchr(p, '/');
        if(sl != NULL)
        {
            printf("%s\n", sl + 1);
        }
        else
        {
            printf("%s\n", p);
        }
        cli_last_retval = 0;
        return 1;
    }
    return 0;
}

/**
 * @brief Handler: extract directory from path.
 */
int cli_intercept_cmd_dirname(const char *p)
{
    if(starts_with(p, "dirname "))
    {
        p += 7;
        p = strip_ws(p);
        const char *sl =
            strrchr(p, '/');
        if(sl != NULL && sl != p)
        {
            printf("%.*s\n",
                   (int)(sl - p), p);
        }
        else if(sl == p)
        {
            printf("/\n");
        }
        else
        {
            printf(".\n");
        }
        cli_last_retval = 0;
        return 1;
    }
    return 0;
}

/**
 * @brief Handler: push directory onto stack.
 */
int cli_intercept_cmd_pushd(const char *p)
{
    if(starts_with(p, "pushd ")
            || starts_with(p, "pushd\t"))
    {
        p += 5;
        p = strip_ws(p);
        char cwd[1024];
        if(getcwd(cwd, sizeof(cwd))
                != NULL)
        {
            /* Push current dir as
             * cli var */
            char idx[32];
            /* Count existing
             * _dirstack entries */
            int dcnt = 0;
            for(int k = 0;
                    k < CLI_MAX_VARS; k++)
            {
                if(cli_vars[k].used
                        && strncmp(
                            cli_vars[k]
                            .name,
                            "_ds_",
                            4) == 0)
                {
                    dcnt++;
                }
            }
            snprintf(idx,
                     sizeof(idx),
                     "_ds_%d", dcnt);
            cli_var_set(idx, cwd);
        }
        if(chdir(p) != 0)
        {
            printf("pushd: %s: %s\n",
                   p,
                   strerror(errno));
            cli_last_retval = 1;
        }
        else
        {
            cli_last_retval = 0;
        }
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_popd(const char *p)
{
    if(strcmp(p, "popd") == 0)
    {
        /* Find highest _ds_N */
        int maxn = -1;
        int maxk = -1;
        for(int k = 0;
                k < CLI_MAX_VARS; k++)
        {
            if(cli_vars[k].used
                    && strncmp(
                        cli_vars[k].name,
                        "_ds_", 4) == 0)
            {
                int n = atoi(
                            cli_vars[k].name
                            + 4);
                if(n > maxn)
                {
                    maxn = n;
                    maxk = k;
                }
            }
        }
        if(maxk >= 0)
        {
            if(chdir(
                        cli_vars[maxk].val)
                    != 0)
            {
                printf("popd: %s\n",
                       strerror(
                           errno));
            }
            cli_vars[maxk].used = 0;
        }
        else
        {
            printf("popd: directory "
                   "stack empty\n");
        }
        cli_last_retval = 0;
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_dirs(const char *p)
{
    if(strcmp(p, "dirs") == 0)
    {
        char cwd[1024];
        if(getcwd(cwd, sizeof(cwd))
                != NULL)
        {
            printf("%s", cwd);
        }
        for(int n = 0;
                n < CLI_MAX_VARS; n++)
        {
            char idx[32];
            snprintf(idx,
                     sizeof(idx),
                     "_ds_%d", n);
            const char *dv =
                cli_var_get(idx);
            if(dv == NULL)
            {
                break;
            }
            printf(" %s", dv);
        }
        printf("\n");
        cli_last_retval = 0;
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_seq(const char *p)
{
    if(starts_with(p, "seq "))
    {
        p += 3;
        p = strip_ws(p);
        double s1 = 0.0;
        double step = 1.0;
        double s2 = 0.0;
        /* Parse up to 3 numbers */
        char *end1 = NULL;
        s1 = strtod(p, &end1);
        if(end1 != NULL
                && *end1 != '\0')
        {
            const char *p2 =
                strip_ws(end1);
            char *end2 = NULL;
            double v2 =
                strtod(p2, &end2);
            if(end2 != NULL
                    && *end2 != '\0')
            {
                const char *p3 =
                    strip_ws(end2);
                double v3 =
                    strtod(p3, NULL);
                /* 3-arg: s1 step s2 */
                step = v2;
                s2 = v3;
            }
            else
            {
                /* 2-arg: s1 s2 */
                s2 = v2;
            }
        }
        else
        {
            /* 1-arg: 1..s1 */
            s2 = s1;
            s1 = 1.0;
        }
        if(step > 0.0)
        {
            for(double v = s1;
                    v <= s2 + 1e-12;
                    v += step)
            {
                printf("%g\n", v);
            }
        }
        else if(step < 0.0)
        {
            for(double v = s1;
                    v >= s2 - 1e-12;
                    v += step)
            {
                printf("%g\n", v);
            }
        }
        cli_last_retval = 0;
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_waitfor_stream(const char *p)
{
    if(starts_with(p, "waitfor_stream "))
    {
        p += 14;
        p = strip_ws(p);
        char sname[CLI_VAR_NAMELEN];
        int si = 0;
        while(*p != '\0'
                && *p != ' '
                && *p != '\t'
                && si
                < CLI_VAR_NAMELEN - 1)
        {
            sname[si++] = *p++;
        }
        sname[si] = '\0';
        p = strip_ws(p);
        double tout = 10.0;
        if(*p != '\0')
        {
            tout = strtod(p, NULL);
        }
        struct timespec wstart;
        clock_gettime(
            CLOCK_MONOTONIC,
            &wstart);
        int found = 0;
        while(1)
        {
            /* Check if SHM exists */
            char shmpath[256];
            snprintf(shmpath,
                     sizeof(shmpath),
                     "%s/%s"
                     ".im.shm",
                     dcshmdir,
                     sname);
            if(access(shmpath,
                      F_OK) == 0)
            {
                found = 1;
                break;
            }
            struct timespec wnow;
            clock_gettime(
                CLOCK_MONOTONIC,
                &wnow);
            double elapsed =
                (double)(wnow.tv_sec
                         - wstart.tv_sec)
                + (double)(
                    wnow.tv_nsec
                    - wstart.tv_nsec)
                / 1e9;
            if(elapsed >= tout)
            {
                break;
            }
            usleep(50000);
        }
        cli_last_retval =
            found ? 0 : 1;
        return 1;
    }
    return 0;
}

int cli_intercept_cmd_waitfor_fps(const char *p)
{
    if(starts_with(p, "waitfor_fps "))
    {
        p += 11;
        p = strip_ws(p);
        char fname[CLI_VAR_NAMELEN];
        int fi = 0;
        while(*p != '\0'
                && *p != ' '
                && *p != '\t'
                && fi
                < CLI_VAR_NAMELEN - 1)
        {
            fname[fi++] = *p++;
        }
        fname[fi] = '\0';
        p = strip_ws(p);
        double tout = 10.0;
        if(*p != '\0')
        {
            tout = strtod(p, NULL);
        }
        struct timespec wstart;
        clock_gettime(
            CLOCK_MONOTONIC,
            &wstart);
        int found = 0;
        while(1)
        {
            char fpath[256];
            snprintf(fpath,
                     sizeof(fpath),
                     "%s/"
                     "fps.%s.shm",
                     dcshmdir,
                     fname);
            if(access(fpath,
                      F_OK) == 0)
            {
                found = 1;
                break;
            }
            struct timespec wnow;
            clock_gettime(
                CLOCK_MONOTONIC,
                &wnow);
            double elapsed =
                (double)(wnow.tv_sec
                         - wstart.tv_sec)
                + (double)(
                    wnow.tv_nsec
                    - wstart.tv_nsec)
                / 1e9;
            if(elapsed >= tout)
            {
                break;
            }
            usleep(50000);
        }
        cli_last_retval =
            found ? 0 : 1;
        return 1;
    }
    return 0;
}
