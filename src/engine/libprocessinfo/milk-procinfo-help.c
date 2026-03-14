/**
 * @file milk-procinfo-help.c
 * @brief Milk procinfo help module
 */

#include <stdio.h>

#define C_TITLE "\033[1;36m"   /* Cyan Bold   -> Main section headers / separators */
#define C_HDR   "\033[1;34m"   /* Blue Bold   -> Subheaders inside sections */
#define C_CMD   "\033[1;32m"   /* Green Bold  -> Command names, syntax, execution */
#define C_NOTE  "\033[1;33m"   /* Yellow Bold -> Tips, Notes, 'run X for more' */
#define C_BOLD  "\033[1m"      /* White Bold  -> Emphasize specific words */
#define C_RST   "\033[0m"      /* Reset */

int main()
{
    printf("\n");
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "             Processinfo & Real-Time Setup              \n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf("Procinfo gives fine-grained execution control to programs\n");
    printf("written with milk. When integrated properly, processes can run\n");
    printf("in daemon-like states, bound to specific CPUs, using real-time\n");
    printf("schedulers, and triggered optimally by ImageStreamIO updates.\n");
    printf("\n");

    printf(C_HDR "Monitoring Processes\n" C_RST);
    printf("You can view all currently running FPS-enabled or procinfo\n");
    printf("tracked processes in the interactive task manager:\n");
    printf("  $ " C_CMD "milk-procCTRL\n" C_RST);
    printf("\n");

    printf(C_HDR "FPS Integration\n" C_RST);
    printf("When an FPS is procinfo-enabled (via the `-procinfo` flag\n");
    printf("upon initialization), a new group (`.procinfo`) appears.\n");
    printf("\n");
    printf("Important Procinfo Parameters:\n");
    printf("  " C_BOLD "procinfo.enabled" C_RST "      Turn ON/OFF the real-time background loop.\n");
    printf("  " C_BOLD "procinfo.RTprio" C_RST "       Assign Linux thread priority (SCHED_FIFO).\n");
    printf("  " C_BOLD "procinfo.taskset" C_RST "      Set CPU affinity mapping.\n");
    printf("  " C_BOLD "procinfo.triggermode" C_RST "  Determine what drives the computation:\n");
    printf("                          0: IMMEDIATE (Continuous loop)\n");
    printf("                          3: SEMAPHORE (Stream semWait trigger)\n");
    printf("                          4: DELAY (Throttle with timer delay)\n");
    printf("  " C_BOLD "procinfo.timersname" C_RST "   (Alias triggersname) The stream to wait on.\n");
    printf("\n");

    printf(C_HDR "Example:\n" C_RST);
    printf("Configure 'myfps00' to compute only when 'cam01' gets a new frame:\n");
    printf("  $ " C_CMD "milk-fps-set myfps00 procinfo.triggermode 3\n" C_RST);
    printf("  $ " C_CMD "milk-fps-set myfps00 procinfo.triggersname cam01\n" C_RST);
    printf("  $ " C_CMD "milk-fps-set myfps00 procinfo.enabled 1\n" C_RST);
    printf("  $ " C_CMD "milk-fps-runstart myfps00\n" C_RST);
    printf("\n");
    
    printf(C_TITLE "========================================================\n" C_RST);
    printf(C_TITLE "             Processinfo Utilities                      \n" C_RST);
    printf(C_TITLE "========================================================\n" C_RST);
    printf("\n");
    printf("  " C_CMD "milk-procCTRL" C_RST "      : Interactive TUI task manager\n");
    printf("  " C_CMD "milk-procinfo-rm" C_RST "   : Remove processinfo shared memory segment\n");
    printf("  " C_CMD "milk-procCTRL-scan" C_RST " : Non-interactive scan of active processes\n");
    printf("\n");
    printf(C_NOTE "Run any of these programs with -h for more help." C_RST "\n");
    printf("\n");
    return 0;
}
