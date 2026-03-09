/**
 * @file milk-fps-cmd.c
 * @brief Milk fps cmd module
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <libgen.h>

#include "fps.h"
#include "fps_globals.h"

void print_help(const char *progname) {
    printf("Usage: %s [OPTIONS] <fpsname>\n\n", progname);
    
    if (strcmp(progname, "milk-fps-confstart") == 0) {
        printf("Starts the configuration process for the given FPS.\n");
    } else if (strcmp(progname, "milk-fps-confstop") == 0) {
        printf("Stops the configuration process for the given FPS.\n");
    } else if (strcmp(progname, "milk-fps-runstart") == 0) {
        printf("Starts the run process for the given FPS.\n");
    } else if (strcmp(progname, "milk-fps-runstop") == 0) {
        printf("Stops the run process for the given FPS.\n");
    } else if (strcmp(progname, "milk-fps-confstep") == 0) {
        printf("Runs a single configuration step for the given FPS.\n");
    } else {
        printf("Executes a command for the given FPS.\n");
    }

    printf("\nOPTIONS:\n");
    printf("  %-15s %s\n", "-h, --help", "Show this help message and exit.");
    printf("  %-15s %s\n", "-tmux", "Run the command inside the FPS tmux session.");
    printf("                  If the tmux session does not exist, it will be automatically created.\n");
    printf("\nDESCRIPTION:\n");
    printf("  This tool allows sending standard lifecycle commands to an existing FPS.\n");
    printf("  Executing without the -tmux flag will run the process locally in the current terminal,\n");
    printf("  which will block the terminal (unless it's a stop command). Using -tmux dispatches\n");
    printf("  the process to the appropriate tmux window ('conf' or 'run') to run in the background.\n");
}

int main(int argc, char *argv[])
{
    const char *progname = basename(argv[0]);

    if (argc < 2) {
        fprintf(stderr, "Usage: %s [OPTIONS] <fpsname>\n", progname);
        return 1;
    }

    int use_tmux = 0;
    const char *fpsname = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(progname);
            return 0;
        } else if (strcmp(argv[i], "-tmux") == 0) {
            use_tmux = 1;
        } else if (fpsname == NULL) {
            fpsname = argv[i];
        } else {
            fprintf(stderr, "Error: Unexpected argument '%s'.\n", argv[i]);
            return 1;
        }
    }

    if (fpsname == NULL) {
        fprintf(stderr, "Error: Missing <fpsname>\n");
        return 1;
    }

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

    if (use_tmux) {
        functionparameter_FPS_tmux_ensure(&fps);
        
        char extra_args[256];
        snprintf(extra_args, sizeof(extra_args), " %s:%s", fpsname, command);
        
        if (functionparameter_FPS_tmux_send_dispatch(fpsname, command, fps.md->execfullpath, extra_args) != 0) {
             fprintf(stderr, "Warning: Command '%s' not recognized for tmux dispatch, running locally...\n", command);
             // fallback
             char cmdline[1024];
             snprintf(cmdline, sizeof(cmdline), "%s %s:%s", fps.md->execfullpath, fpsname, command);
             printf("Executing locally: %s\n", cmdline);
             int ret = system(cmdline);
             function_parameter_struct_disconnect(&fps);
             return WEXITSTATUS(ret);
        }
        
    } else {
        char cmdline[1024];
        snprintf(cmdline, sizeof(cmdline), "%s %s:%s", fps.md->execfullpath, fpsname, command);
        
        printf("Executing locally: %s\n", cmdline);
        int ret = system(cmdline);
        function_parameter_struct_disconnect(&fps);
        return WEXITSTATUS(ret);
    }

    function_parameter_struct_disconnect(&fps);
    return 0;
}
