/**
 * @file milk-cli-help-synchro.c
 * @brief Terminal color macros for help display
 */

#include "CLIcore.h"

// Terminal color macros for help display
#define C_RST    "\033[0m"
#define C_TITLE  "\033[1;36m"
#define C_HDR    "\033[1;33m"
#define C_CMD    "\033[1;32m"
#define C_NOTE   "\033[1;35m"
#define C_BOLD   "\033[1m"

/**
 * @brief Print help for the milk synchro debug tool.
 */
void print_milk_synchro_help(void)
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "             milk SYNCHRONIZATION OVERVIEW\n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf("The milk framework uses several synchronization mechanisms\n");
    printf("to ensure robust real-time performance and data consistency\n");
    printf("across processes.\n");
    printf("\n");

    printf(C_HDR "1. Synchronization Primitives (Semaphores & Mutexes)\n" C_RST);
    printf("ImageStreamIO streams use POSIX semaphores to notify waiting\n");
    printf("processes when new data is available. Mutexes protect the\n");
    printf("shared memory block during modifications to prevent race\n");
    printf("conditions.\n");
    printf("\n");
    printf("  " C_NOTE "Key Semaphores:" C_RST "\n");
    printf("  - " C_BOLD "SemLog" C_RST ": Tracks stream metadata changes (e.g., size)\n");
    printf("  - " C_BOLD "SemWrit" C_RST ": Readers wait on this. Writers post when new\n");
    printf("    data is ready.\n");
    printf("  - " C_BOLD "SemRead" C_RST ": Writers wait on this. Readers post when they\n");
    printf("    are done reading (useful in strict pipelines).\n");
    printf("\n");

    printf(C_HDR "2. Interactions Between Streams\n" C_RST);
    printf("A common pattern in milk is the \"stream-processing chain.\"\n");
    printf("Process A reads Stream 1, performs computation, and writes\n");
    printf("to Stream 2. Process B waits on Stream 2's SemWrit semaphore\n");
    printf("and awakens immediately when Process A posts it at the end\n");
    printf("of the frame.\n");
    printf("This creates a low-latency, strictly ordered data pipeline\n");
    printf("spanning multiple detached processes.\n");
    printf("\n");

    printf(C_HDR "3. Processinfo (procinfo API)\n" C_RST);
    printf("Processinfo monitors and controls real-time execution.\n");
    printf("Every active process registers a \"procinfo\" struct in\n");
    printf("shared memory, allowing you to view and adjust its state:\n");
    printf("  - Loop counter\n");
    printf("  - Execution time, latency, and jitter measurements\n");
    printf("  - CPU affinity (" C_CMD "taskset" C_RST " equivalent)\n");
    printf("  - RT priority (SCHED_FIFO)\n");
    printf("\n");

    printf(C_HDR "4. FPS and Triggering Modes\n" C_RST);
    printf("The Function Parameter Structure (FPS) is the interface to\n");
    printf("Procinfo. FPS dictates " C_BOLD "when" C_RST " a process runs.\n");
    printf("Using the " C_CMD "procinfo" C_RST " settings, you can select the Triggermode:\n");
    printf("  - " C_NOTE "IMMEDIATE (0):" C_RST " Loop continuously, as fast as possible.\n");
    printf("  - " C_NOTE "CNT0 (1):" C_RST "      Reserved trigger mode.\n");
    printf("  - " C_NOTE "CNT1 (2):" C_RST "      Reserved trigger mode.\n");
    printf("  - " C_NOTE "SEMAPHORE (3):" C_RST " The standard trigger. Process sleeps\n");
    printf("                 until the designated stream's semaphore\n");
    printf("                 is posted.\n");
    printf("  - " C_NOTE "DELAY (4):" C_RST "     Executes on a fixed timer.\n");
    printf("  - " C_NOTE "CNT2 (5):" C_RST "      Reserved trigger mode.\n");
    printf("\n");
    printf("  You define " C_CMD "procinfo.triggersname" C_RST " to select the stream, and\n");
    printf("  " C_CMD "procinfo.semindexrequested" C_RST " (typically 0) to bind the process.\n");
    printf("\n");

    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "                USEFUL COMMANDS                         \n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf(C_CMD "  milk-procinfo-help  " C_RST "Detailed procinfo help\n");
    printf(C_CMD "  milk-fps-help       " C_RST "Detailed FPS help\n");
    printf(C_CMD "  milk-streamCTRL     " C_RST "Interactive stream monitor TUI\n");
    printf(C_CMD "  milk-procCTRL       " C_RST "Interactive processinfo monitor TUI\n");
    printf(C_CMD "  fpslist             " C_RST "List running FPS processes\n");
    printf(C_CMD "  ps-milk             " C_RST "List milk-related OS processes\n");
    printf("\n");
}

int main(
    int argc,
    char *argv[])
{
    /* One-line help - before CLI_startup() */
    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "-h1") == 0 ||
                strcmp(argv[i], "--help-oneline") == 0)
        {
            printf("milk synchronization overview\n");
            return 0;
        }
    }

    // Initialize data structure
    dcquiet = 1;
    CLI_startup();

    // Call the centralized help function
    print_milk_synchro_help();

    return 0;
}
