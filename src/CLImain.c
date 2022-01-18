#include "milk_config.h"

#include <assert.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CLIcore_UI.h"
#include <CommandLineInterface/CLIcore.h>

#define STYLE_BOLD "\033[1m"
#define STYLE_NO_BOLD "\033[22m"

#define STRINGMAXLEN_VERSIONSTRING 80
#define STRINGMAXLEN_APPNAME 40

int main(int argc, char *argv[])
{
    char AppName[STRINGMAXLEN_APPNAME];

    char *CLI_APPNAME = getenv("MILKCLI_APPNAME");
    if (CLI_APPNAME != NULL)
    {
        strncpy(AppName, CLI_APPNAME, STRINGMAXLEN_APPNAME - 1);
    }
    else
    {
        strncpy(AppName, "milk", STRINGMAXLEN_APPNAME - 1);
    }

    if (getenv("MILK_QUIET"))
    {
        data.quiet = 1;
    }
    else
    {
        data.quiet = 0;
    }

    if (getenv("MILK_ERROREXIT"))
    {
        data.errorexit = 1;
    }
    else
    {
        data.errorexit = 0;
    }

    // Allocate data.testpointarray
#ifndef NDEBUG
    printf("        [ENABLED]  Code test point tracing\n");
    // allocate circular buffer memory
    data.testpointarray = (CODETESTPOINT *)malloc(sizeof(CODETESTPOINT) * CODETESTPOINTARRAY_NBCNT);
    data.testpointarrayinit = 1;
    // initialize loop counter
    // loop counter increments when reaching end of circular buffer
    data.testpointloopcnt = 0;
    // set current entry index to zero
    data.testpointcnt = 0;
#endif

    char versionstring[STRINGMAXLEN_VERSIONSTRING];
    snprintf(versionstring, STRINGMAXLEN_VERSIONSTRING, "%d.%02d.%02d%s", VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH,
             VERSION_OPTION);

    if (data.quiet == 0)
    {
        printf(STYLE_BOLD);
        printf("\n        milk  v %s\n", versionstring);
#ifndef NDEBUG
        printf("        === DEBUG MODE : assert() & DEBUG_TRACEPOINT  enabled ===\n");
#endif
        printf(STYLE_NO_BOLD);
        if (data.errorexit == 1)
        {
            printf("        EXIT-ON-ERROR mode\n");
        }
    }

    strcpy(data.package_name, PACKAGE_NAME);

    data.package_version_major = VERSION_MAJOR;
    data.package_version_minor = VERSION_MINOR;
    data.package_version_patch = VERSION_PATCH;

    strcpy(data.package_version, versionstring);

    strcpy(data.sourcedir, SOURCEDIR);
    strcpy(data.configdir, CONFIGDIR);
    strcpy(data.installdir, INSTALLDIR);

    if (data.quiet == 0)
    {
        //printf("        %s version %s\n", data.package_name, data.package_version);
#ifdef IMAGESTRUCT_VERSION
        printf("        ImageStreamIO v %s\n", IMAGESTRUCT_VERSION);
#endif
        //printf("        GNU General Public License v3.0\n");
        //printf("        Report bugs to : %s\n", PACKAGE_BUGREPORT);
        //printf("        Type \"help\" for instructions\n");
        printf("        \n");
    }

    // default exit code
    data.exitcode = RETURN_SUCCESS;

    runCLI(argc, argv, AppName);

    //errno_t CLIretval = RETURN_SUCCESS;

    if (data.quiet == 0)
    {
        printf("EXIT CODE %d\n", data.exitcode);
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
    data.testpointarrayinit = 0;
    free(data.testpointarray);
#endif

    return data.exitcode;
}
