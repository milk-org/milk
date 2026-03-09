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

#include "CLIcore.h"

#include "CLIcore/CLIcore_UI.h"

#define STYLE_BOLD    "\033[1m"
#define STYLE_NO_BOLD "\033[22m"

#define STRINGMAXLEN_VERSIONSTRING 80
#define STRINGMAXLEN_APPNAME       40

int main(int argc, char *argv[])
{
    char AppName[STRINGMAXLEN_APPNAME];

    char *CLI_APPNAME = getenv("MILKCLI_APPNAME");
    if(CLI_APPNAME != NULL)
    {
        strncpy(AppName, CLI_APPNAME, STRINGMAXLEN_APPNAME - 1);
    }
    else
    {
        strncpy(AppName, "milk-cli", STRINGMAXLEN_APPNAME - 1);
    }

    if(getenv("MILK_QUIET"))
    {
        dcquiet = 1;
    }
    else
    {
        dcquiet = 0;
    }

    if(getenv("MILK_ERROREXIT"))
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
    dctestptarr     = (CODETESTPOINT *) malloc(sizeof(CODETESTPOINT) *
                              CODETESTPOINTARRAY_NBCNT);
    dctestptinit = 1;
    // initialize loop counter
    // loop counter increments when reaching end of circular buffer
    dctestptlcnt = 0;
    // set current entry index to zero
    dctestptcnt = 0;
#endif

    char versionstring[STRINGMAXLEN_VERSIONSTRING];
    snprintf(versionstring,
             STRINGMAXLEN_VERSIONSTRING,
             "%d.%02d.%02d%s",
             VERSION_MAJOR,
             VERSION_MINOR,
             VERSION_PATCH,
             VERSION_OPTION);

    if(dcquiet == 0)
    {
        printf(STYLE_BOLD);
        printf("\n        milk-cli  v %s\n", versionstring);
#ifndef NDEBUG
        printf(
            "        === DEBUG MODE : assert() & DEBUG_TRACEPOINT  enabled "
            "===\n");
#endif
        printf(STYLE_NO_BOLD);
        if(dcerrorexit == 1)
        {
            printf("        EXIT-ON-ERROR mode\n");
        }
    }

    strcpy(dcpkgname, PACKAGE_NAME);

    dcpkgmajor = VERSION_MAJOR;
    dcpkgminor = VERSION_MINOR;
    dcpkgpatch = VERSION_PATCH;

    strcpy(dcpkgver, versionstring);

    strcpy(dcsourcedir, SOURCEDIR);
    strcpy(dcconfigdir, CONFIGDIR);
    strcpy(dcinstalldir, INSTALLDIR);

    if(dcquiet == 0)
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

    if(dcquiet == 0)
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

    if(getenv("MILK_WRITECODETRACE"))
    {
        write_tracedebugfile();
    }
    printf("De-allocating test circular buffer\n");
    fflush(stdout);
    dctestptinit = 0;
    free(dctestptarr);
#endif

    return dcexitcode;
}
