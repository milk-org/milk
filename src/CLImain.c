#include "milk_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sched.h>
#include <omp.h>
#include <assert.h>
#include <pthread.h>
#include <CommandLineInterface/CLIcore.h>



#define STYLE_BOLD    "\033[1m"
#define STYLE_NO_BOLD "\033[22m"


DATA __attribute__((used)) data;


#define STRINGMAXLEN_VERSIONSTRING 80


int main(
    int argc,
    char *argv[]
)
{
    char *AppName = "milk";


    if(getenv("MILK_QUIET"))
    {
        data.quiet = 1;
    }
    else
    {
		data.quiet = 0;
	}

    char versionstring[STRINGMAXLEN_VERSIONSTRING];
    snprintf(versionstring, STRINGMAXLEN_VERSIONSTRING, "%d.%02d.%02d%s",
             VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, VERSION_OPTION);

    if(data.quiet == 0)
    {
        printf(STYLE_BOLD);
        printf("\n        milk  v %s\n", versionstring);
#ifndef NDEBUG
        printf("        === DEBUG MODE : assert() & DEBUG_TRACEPOINT  enabled ===\n");
#endif
        printf(STYLE_NO_BOLD);
    }

    strcpy(data.package_name, PACKAGE_NAME);


    data.package_version_major = VERSION_MAJOR;
    data.package_version_minor = VERSION_MINOR;
    data.package_version_patch = VERSION_PATCH;

    strcpy(data.package_version, versionstring);

    strcpy(data.sourcedir, SOURCEDIR);
    strcpy(data.configdir, CONFIGDIR);
    strcpy(data.installdir, INSTALLDIR);


    if(data.quiet == 0)
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

    runCLI(argc, argv, AppName);

	if(data.quiet == 0) {
		printf("NORMAL EXIT\n");
	}


    // clean-up calling thread
    //pthread_exit(NULL);

    return RETURN_SUCCESS;
}
