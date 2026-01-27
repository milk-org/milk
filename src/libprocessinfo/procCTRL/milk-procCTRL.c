#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>

#include "ImageStreamIO/ImageStreamIO.h"
#include "processinfo.h"
#include "procCTRL_TUI.h"

// Prototypes for functions defined in other procCTRL files
errno_t processinfo_CTRLscreen();

void print_help(const char *progname) {
    printf("Usage: %s [options]\n", progname);
    printf("\n");
    printf("MILK Process Control Tool (procCTRL)\n");
    printf("====================================\n");
    printf("This tool monitors and controls real-time loop processes in the MILK environment.\n");
    printf("It provides a Text User Interface (TUI) to inspect process status, CPU usage,\n");
    printf("scheduling, and shared memory links. It also allows sending control signals\n");
    printf("to processes (pause, step, exit) and managing their affinity.\n");
    printf("\n");
    printf("  File Operations:\n");
    printf("    READS:        processinfo.list.shm (Global Process List)\n");
    printf("                  - Reads PID, name, and active status of all registered processes.\n");
    printf("    READS/WRITES: proc.<process_name>.<PID>.shm (Process Control Block)\n");
    printf("                  - Reads detailed status (loopstat), counters, timing, and triggers.\n");
    printf("                  - Writes control signals (CTRLval) to Pause/Resume/Step/Exit.\n");
    printf("\n");
    printf("Options:\n");
    printf("  -h, --help           Show this help message and exit.\n");
    printf("  -d, --debug          Enable debug output to stdout (disables TUI).\n");
    printf("  -c, --check-scan     Check if milk-procCTRL-scan is running and print its PID.\n");
    printf("\n");
    printf("Key Bindings (Interactive Mode):\n");
    printf("--------------------------------\n");
    printf("  F2  : Switch to CONTROL view (default)\n");
    printf("  F3  : Switch to RESOURCES view (CPU/Memory)\n");
    printf("  F4  : Switch to TRIGGERING view\n");
    printf("  F5  : Switch to TIMING view\n");
    printf("  F6  : Switch to PROCINFO parameters summary view\n");
    printf("  h   : Show in-app Help screen\n");
    printf("  f   : Freeze/Unfreeze display updates\n");
    printf("  s   : Sort process list by currently highlighted column\n");
    printf("  S   : Apply current sort criteria to all tabs/modes\n");
    printf("  + - : Increase/Decrease update frequency\n");
    printf("  x   : Exit milk-procCTRL\n");
    printf("\n");
    printf("Navigation & Column Control:\n");
    printf("  UP/DN     : Move process selection cursor\n");
    printf("  LEFT/RGHT : Move column highlight cursor\n");
    printf("  1-9       : Toggle visibility of specific columns (mode-specific)\n");
    printf("  CTRL + L/R: Cycle through display modes/tabs\n");
    printf("\n");
    printf("Process Control (on selection or current process):\n");
    printf("  SPACE : Select/Unselect current process\n");
    printf("  u     : Unselect all processes\n");
    printf("  p     : Pause/Resume (Writes CTRLval)\n");
    printf("  CTRL+S: Step (Writes CTRLval)\n");
    printf("  e     : Request clean exit (Writes CTRLval)\n");
    printf("  T     : Send SIGTERM\n");
    printf("  K     : Send SIGKILL\n");
    printf("  I     : Send SIGINT\n");
    printf("  r / R : Remove log for selected / all inactive process(es)\n");
    printf("  z / Z : Zero counter for selected / all process(es)\n");
    printf("\n");
    printf("For more details, press 'h' inside the tool.\n");
}

int main(int argc, char *argv[])
{
    int opt;

    // Silence ImageStreamIO library (suppress stderr warnings/errors in TUI)
    ImageStreamIO_set_verbosity(0);

    static struct option long_options[] = {
        {"help",       no_argument,       0, 'h'},
        {"debug",      no_argument,       0, 'd'},
        {"log",        required_argument, 0, 'l'},
        {"check-scan", no_argument,       0, 'c'},
        {0, 0, 0, 0}
    };

    optind = 1; // Ensure getopt starts from beginning

    while ((opt = getopt_long(argc, argv, "hdcl:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'h':
                print_help(argv[0]);
                return 0;
            case 'd':
                procCTRL_debug_mode = 1;
                break;
            case 'c':
                {
                    int scan_ok = 0;
                    int tmux_ok = 0;
                    pid_t my_pid = getpid();

                    // Check scanner
                    FILE *fp = popen("pgrep \"milk-procCTRL-s\"", "r");
                    if (fp != NULL) {
                        char pid_str[64];
                        while (fgets(pid_str, sizeof(pid_str), fp) != NULL) {
                            pid_t pid = (pid_t)atoi(pid_str);
                            if (pid != my_pid) {
                                pid_str[strcspn(pid_str, "\n")] = 0;
                                printf("Scanner milk-procCTRL-scan is running (PID %s)\n", pid_str);
                                scan_ok = 1;
                                break;
                            }
                        }
                        pclose(fp);
                    }
                    if (!scan_ok) {
                        printf("Scanner milk-procCTRL-scan is NOT running\n");
                    }

                    // Check tmux
                    fp = popen("tmux -V 2>/dev/null", "r");
                    if (fp != NULL) {
                        char tmux_ver[64];
                        if (fgets(tmux_ver, sizeof(tmux_ver), fp) != NULL) {
                            tmux_ver[strcspn(tmux_ver, "\n")] = 0;
                            printf("tmux is installed (%s)\n", tmux_ver);
                            tmux_ok = 1;
                        } else {
                            printf("tmux is NOT installed\n");
                        }
                        pclose(fp);
                    } else {
                         printf("tmux is NOT installed\n");
                    }
                    
                    if (scan_ok && tmux_ok) {
                        return 0;
                    } else {
                        return 1;
                    }
                }
            case 'l':
                strncpy(procCTRL_logfile, optarg, 1023);
                break;
            default:
                print_help(argv[0]);
                return 1;
        }
    }

    // Run the tool
    processinfo_CTRLscreen();

    return 0;
}
