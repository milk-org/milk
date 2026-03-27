/**
 * @file    milk-seq-help.c
 * @brief   Comprehensive documentation for milk-seq
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define COLOR_RESET   "\033[0m"
#define COLOR_BOLD    "\033[1m"
#define COLOR_CYAN    "\033[36m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"

int main()
{
    printf("\n");
    printf(COLOR_BOLD COLOR_CYAN "                     MILK SEQUENCER (milk-seq)\n" COLOR_RESET);
    printf(COLOR_BOLD "                     =========================\n\n" COLOR_RESET);

    printf(COLOR_BOLD "1. ARCHITECTURE\n" COLOR_RESET);
    printf("  The milk sequencer (`milk-seq`) is a standalone, real-time daemon used to\n");
    printf("  execute commands deterministically and headlessly. It is designed for strict\n");
    printf("  timing orchestration, such as hardware calibration or adaptive optics loops.\n\n");
    printf("  Key features:\n");
    printf("  - " COLOR_GREEN "Zero-Copy State:" COLOR_RESET " Exposes execution state via `/dev/shm/milkseq.<name>.shm`.\n");
    printf("  - " COLOR_GREEN "Cross-Process Injection:" COLOR_RESET " Accepts commands injected via a named FIFO pipe.\n");
    printf("  - " COLOR_GREEN "Deterministic Lock-Stepping:" COLOR_RESET " Provides `wait_fps` and `wait_seq` primitives.\n\n");

    printf(COLOR_BOLD "2. BASIC USAGE\n" COLOR_RESET);
    printf("  " COLOR_CYAN "milk-seq -n <name> -f <script.seq>\n" COLOR_RESET);
    printf("  Starts a sequencer instance in the background parsing the script.\n\n");

    printf("  Additionally, you can run commands from the CLI environment:\n");
    printf("    " COLOR_YELLOW "seq.list" COLOR_RESET "                 List all active sequencer instances\n");
    printf("    " COLOR_YELLOW "seq.start <name>" COLOR_RESET "         Start a blank sequencer\n");
    printf("    " COLOR_YELLOW "seq.stop  <name>" COLOR_RESET "         Stop a sequencer safely\n");
    printf("    " COLOR_YELLOW "seq.status <name>" COLOR_RESET "        View task status and error counts\n");
    printf("    " COLOR_YELLOW "seq.submit <name> <cmd>" COLOR_RESET "  Inject a command from CLI to sequence\n\n");

    printf(COLOR_BOLD "3. SCRIPTING COMMANDS\n" COLOR_RESET);
    printf("  Inside a `.seq` file, standard bash executable calls are valid.\n");
    printf("  However, the sequencer natively intercepts certain real-time commands:\n\n");

    printf("  " COLOR_GREEN "wait_fps <name> <val>" COLOR_RESET "\n");
    printf("      Pauses execution until the specified FPS parameter exactly equals <val>.\n");
    printf("      Perfect for synchronizing with hardware loops.\n\n");

    printf("  " COLOR_GREEN "wait_seq <name>" COLOR_RESET "\n");
    printf("      Pauses execution until the target sequencer script completes.\n\n");

    printf("  " COLOR_GREEN "if_fps_status <name> <val> <cmd>" COLOR_RESET "\n");
    printf("      Executes <cmd> only if the FPS parameter matching <name> is equal to <val>.\n\n");

    printf("  " COLOR_GREEN "on_error <abort|skip|retry>" COLOR_RESET "\n");
    printf("      Sets global fault behavior robustly without complex test clauses.\n\n");

    printf(COLOR_BOLD "4. OBSERVABILITY\n" COLOR_RESET);
    printf("  Because the sequencer creates an SHM state struct, you can view its\n");
    printf("  progression exactly. In the milk environment, you can read variables like:\n");
    printf("    " COLOR_YELLOW "echo ${seq.myloop.status}" COLOR_RESET "\n");
    printf("    " COLOR_YELLOW "echo ${seq.myloop.nb_tasks}" COLOR_RESET "\n\n");

    return 0;
}
