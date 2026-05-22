/**
 * @file CLImain.c
 * @brief Climain module
 */

#include "milk_config.h"

#include <assert.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "CLIcore.h"

#include "CLIcore/CLIcore_UI_execute.h"

#define STYLE_BOLD "\033[1m"
#define STYLE_NO_BOLD "\033[22m"

#define STRINGMAXLEN_VERSIONSTRING 80
#define STRINGMAXLEN_APPNAME 40

int main(int argc, char *argv[])
{
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-h1") == 0 || strcmp(argv[i], "--help-oneline") == 0)
        {
            printf("milk interactive command-line interface\n");
            return 0;
        }

        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_milk_cli_help();
            return 0;
        }
    }

    char AppName[STRINGMAXLEN_APPNAME];

    char *CLI_APPNAME = getenv("MILKCLI_APPNAME");
    if (CLI_APPNAME != NULL)
    {
        strncpy(AppName, CLI_APPNAME, STRINGMAXLEN_APPNAME - 1);
    }
    else
    {
        strncpy(AppName, "milk-cli", STRINGMAXLEN_APPNAME - 1);
    }

    if (getenv("MILK_QUIET"))
    {
        dcquiet = 1;
    }
    else
    {
        dcquiet = 0;
    }

    if (getenv("MILK_ERROREXIT"))
    {
        dcerrorexit = 1;
    }
    else
    {
        dcerrorexit = 0;
    }

    // Allocate dctestptarr
#ifndef NDEBUG
    printf("        [ENABLED]  Code test point tracing\n");
    // allocate circular buffer memory
    dctestptarr  = (CODETESTPOINT *) malloc(sizeof(CODETESTPOINT) * CODETESTPOINTARRAY_NBCNT);
    dctestptinit = 1;
    // initialize loop counter
    // loop counter increments when reaching end of circular buffer
    dctestptlcnt = 0;
    // set current entry index to zero
    dctestptcnt = 0;
#endif

    char versionstring[STRINGMAXLEN_VERSIONSTRING];
    snprintf(versionstring, STRINGMAXLEN_VERSIONSTRING, "%d.%02d.%02d%s", VERSION_MAJOR,
             VERSION_MINOR, VERSION_PATCH, VERSION_OPTION);

    if (dcquiet == 0)
    {
        printf(STYLE_BOLD);
        printf("\n        milk-cli  v %s  (compiled %s %s)\n", versionstring, __DATE__, __TIME__);
#ifndef NDEBUG
        printf("        === DEBUG MODE : assert() & DEBUG_TRACEPOINT  enabled "
               "===\n");
#endif
        printf(STYLE_NO_BOLD);
        if (dcerrorexit == 1)
        {
            printf("        EXIT-ON-ERROR mode\n");
        }
    }

    strncpy(dcpkgname, PACKAGE_NAME, sizeof(dcpkgname) - 1);
    dcpkgname[sizeof(dcpkgname) - 1] = '\0';

    dcpkgmajor = VERSION_MAJOR;
    dcpkgminor = VERSION_MINOR;
    dcpkgpatch = VERSION_PATCH;

    strncpy(dcpkgver, versionstring, sizeof(dcpkgver) - 1);
    dcpkgver[sizeof(dcpkgver) - 1] = '\0';

    strncpy(dcsourcedir, SOURCEDIR, sizeof(dcsourcedir) - 1);
    dcsourcedir[sizeof(dcsourcedir) - 1] = '\0';
    strncpy(dcconfigdir, CONFIGDIR, sizeof(dcconfigdir) - 1);
    dcconfigdir[sizeof(dcconfigdir) - 1] = '\0';
    strncpy(dcinstalldir, INSTALLDIR, sizeof(dcinstalldir) - 1);
    dcinstalldir[sizeof(dcinstalldir) - 1] = '\0';

    if (dcquiet == 0)
    {
        //printf("        %s version %s\n", dcpkgname, dcpkgver);
#ifdef IMAGESTRUCT_VERSION
        printf("        ImageStreamIO v %s\n", IMAGESTRUCT_VERSION);
#endif
        //printf("        GNU General Public License v3.0\n");
        //printf("        Report bugs to : %s\n", PACKAGE_BUGREPORT);
        //printf("        Type \"help\" for instructions\n");
        printf("        \n");
    }

    // default exit code
    dcexitcode = RETURN_SUCCESS;

    runCLI(argc, argv, AppName);

    //errno_t CLIretval = RETURN_SUCCESS;

    if (dcquiet == 0)
    {
        printf("EXIT CODE %d\n", dcexitcode);
    }
    else
    {
        printf("\n");
    }

    // clean-up calling thread
    //pthread_exit(NULL);

#ifndef NDEBUG

    if (getenv("MILK_WRITECODETRACE"))
    {
        write_tracedebugfile();
    }
    printf("De-allocating test circular buffer\n");
    fflush(stdout);
    dctestptinit = 0;
    free(dctestptarr);
#endif

    /* Final terminal cleanup.
     * The scroll region may still be restricted
     * (rows 1..N-1) with hint text stuck on row N.
     * Escape codes like ESC[2K and ESC[r have no
     * effect in some VTE terminals. But cursor
     * positioning (ESC[r;1H) DOES work — the hint
     * area rendering proves it. So we overwrite
     * the hint text with plain space characters. */
    {
        struct winsize ws;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) >= 0 && ws.ws_row > 0)
        {
            int  r = ws.ws_row;
            int  c = ws.ws_col;
            char esc[128];
            int  n;

            fflush(stdout);

            /* 1. Move cursor to bottom row col 1 */
            n = snprintf(esc, sizeof(esc), "\033[%d;1H", r);
            if (write(STDOUT_FILENO, esc, n) < 0)
            {
            }

            /* 2. Overwrite entire row with spaces */
            {
                char spaces[256];
                int  remain = c;
                memset(spaces, ' ', sizeof(spaces));
                while (remain > 0)
                {
                    int chunk = remain;
                    if (chunk > (int) sizeof(spaces))
                    {
                        chunk = (int) sizeof(spaces);
                    }
                    if (write(STDOUT_FILENO, spaces, chunk) < 0)
                    {
                    }
                    remain -= chunk;
                }
            }

            /* 3. Reset scroll region to full */
            n = snprintf(esc, sizeof(esc), "\033[1;%dr", r);
            if (write(STDOUT_FILENO, esc, n) < 0)
            {
            }

            /* 4. Position cursor for bash prompt */
            n = snprintf(esc, sizeof(esc), "\033[%d;1H", r - 1);
            if (write(STDOUT_FILENO, esc, n) < 0)
            {
            }
        }
    }

    return dcexitcode;
}
