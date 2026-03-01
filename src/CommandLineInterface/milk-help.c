#include <stdio.h>
#include <stdlib.h>

#define C_RST    "\033[0m"
#define C_TITLE  "\033[1;36m"
#define C_HDR    "\033[1;35m"
#define C_CMD    "\033[32m"
#define C_BOLD   "\033[1m"
#define C_NOTE   "\033[33m"

int main()
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "                    milk OVERVIEW                    \n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf("The milk framework is built around three core pillars for\n");
    printf("high-performance, real-time data processing:\n");
    printf("\n");

    printf(C_HDR "1. ImageStreamIO (Streams)\n" C_RST);
    printf("Fast, low-latency shared-memory data streams designed to\n");
    printf("pass images and multi-dimensional arrays between distinct\n");
    printf("processes with zero-copy overhead.\n");
    printf("  " C_NOTE "Detailed guide:" C_RST " Run " C_CMD "milk-stream-help\n" C_RST);
    printf("\n");

    printf(C_HDR "2. Function Parameter Structure (FPS)\n" C_RST);
    printf("A shared memory architecture providing a unified namespace\n");
    printf("to manage configurations, parameters, and telemetry for\n");
    printf("applications seamlessly across the CLI and API.\n");
    printf("  " C_NOTE "Detailed guide:" C_RST " Run " C_CMD "milk-fps-help\n" C_RST);
    printf("\n");

    printf(C_HDR "3. Processinfo (procinfo API)\n" C_RST);
    printf("Advanced real-time execution management, CPU affinity,\n");
    printf("scheduling policies, and stream-based process triggering.\n");
    printf("  " C_NOTE "Detailed guide:" C_RST " Run " C_CMD "milk-procinfo-help\n" C_RST);
    printf("\n");

    printf(C_TITLE "--------------------------------------------------------\n" C_RST);
    printf(C_HDR "General Usage\n" C_RST);
    printf("To enter the interactive milk shell, simply type:\n");
    printf("  $ " C_CMD "milk\n" C_RST);
    printf("\n");
    printf("From within the milk shell, you can list\n");
    printf("available commands to see all capabilities:\n");
    printf("  $ " C_CMD "help\n" C_RST);
    printf("\n");

    return 0;
}
