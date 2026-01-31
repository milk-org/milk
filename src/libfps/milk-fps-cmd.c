#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <libgen.h>

#include "fps.h"
#include "fps_globals.h"

int main(int argc, char *argv[])
{
    const char *progname = basename(argv[0]);

    if (argc < 2 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        fprintf(stderr, "Usage: %s <fpsname>\n", progname);
        return 1;
    }

    const char *fpsname = argv[1];

    FUNCTION_PARAMETER_STRUCT fps;
    fps.SMfd = -1;

    if (function_parameter_struct_connect(fpsname, &fps, 0) == -1) {
        fprintf(stderr, "Error: cannot connect to FPS \"%s\".\n", fpsname);
        return 1;
    }

    char *command = NULL;
    if (strcmp(progname, "milk-fps-confstart") == 0) command = "confstart";
    else if (strcmp(progname, "milk-fps-confstop") == 0) command = "confstop";
    else if (strcmp(progname, "milk-fps-runstart") == 0) command = "runstart";
    else if (strcmp(progname, "milk-fps-runstop") == 0) command = "runstop";
    else if (strcmp(progname, "milk-fps-confstep") == 0) command = "confstep";

    if (command == NULL) {
        fprintf(stderr, "Error: unknown command \"%s\".\n", progname);
        function_parameter_struct_disconnect(&fps);
        return 1;
    }

    if (strlen(fps.md->execfullpath) == 0 || strcmp(fps.md->execfullpath, "unknown") == 0) {
        fprintf(stderr, "Error: execfullpath not set for FPS \"%s\".\n", fpsname);
        function_parameter_struct_disconnect(&fps);
        return 1;
    }

    char cmdline[1024];
    snprintf(cmdline, sizeof(cmdline), "%s %s:%s", fps.md->execfullpath, fpsname, command);
    
    printf("Executing: %s\n", cmdline);
    int ret = system(cmdline);

    function_parameter_struct_disconnect(&fps);

    return WEXITSTATUS(ret);
}
