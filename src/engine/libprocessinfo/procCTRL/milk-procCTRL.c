/**
 * @file milk-procCTRL.c
 * @brief Milk procctrl module
 */

#include <getopt.h>

#include "processinfo.h"
#include "procCTRL_TUI.h"

#include "milk_help.h"

// Prototypes for functions defined in other procCTRL files
/**
 * @brief Forward declaration stub for procCTRL screen.
 *
 * Implemented in the TUI module; declared here
 * for the standalone main entry point.
 */
errno_t processinfo_CTRLscreen();

/**
 * @brief Print help message for milk-procCTRL.
 */
static void print_help(
    const char *progname,
    int mh_color)
{
    milk_help_banner(progname, "interactive TUI for monitoring and controlling milk processes",
                     mh_color);

    milk_help_section("Usage", mh_color);
    printf("  $ %s [%s]\n\n", progname, MH(MH_OPT, "options"));

    milk_help_section("Description", mh_color);
    printf("  This tool monitors and controls real-time loop processes in the MILK environment.\n"
           "  It provides a Text User Interface (TUI) to inspect process status, CPU usage,\n"
           "  scheduling, and shared memory links. It also allows sending control signals\n"
           "  to processes (pause, step, exit) and managing their affinity.\n"
           "\n"
           "  File Operations:\n"
           "  - READS:        %s (Global Process List)\n"
           "                  Reads PID, name, and active status of all registered processes.\n"
           "  - READS/WRITES: %s (Process Control Block)\n"
           "                  Reads detailed status (loopstat), counters, timing, and triggers.\n"
           "                  Writes control signals (CTRLval) to Pause/Resume/Step/Exit.\n\n",
           MH(MH_BOLD, "processinfo.list.shm"),
           MH(MH_BOLD, "proc.<process_name>.<PID>.shm"));

    milk_help_section("Options", mh_color);
    printf("  %s             Show this help message and exit\n", MH(MH_OPT, "-h, --help"));
    printf("  %s             One-line description and exit\n", MH(MH_OPT, "-h1, --help-oneline"));
    printf("  %s             Full help, forced monochrome\n", MH(MH_OPT, "-hm, --help-mono"));
    printf("  %s             Enable debug output to stdout (disables TUI)\n", MH(MH_OPT,
            "-d, --debug"));
    printf("  %s             Check if milk-procCTRL-scan is running and print its PID\n", MH(MH_OPT,
            "-c, --check-scan"));
    printf("  %s %s                   Log output to specified file\n\n", MH(MH_OPT, "-l, --log"),
           MH(MH_ARG, "FILE"));

    milk_help_section("Key Bindings (Interactive Mode)", mh_color);
    printf("  %s                              Switch to CONTROL view (default)\n", MH(MH_CMD, "F2"));
    printf("  %s                              Switch to RESOURCES view (CPU/Memory)\n", MH(MH_CMD,
            "F3"));
    printf("  %s                              Switch to TRIGGERING view\n", MH(MH_CMD, "F4"));
    printf("  %s                              Switch to TIMING view\n", MH(MH_CMD, "F5"));
    printf("  %s                              Switch to PROCINFO parameters summary view\n", MH(MH_CMD,
            "F6"));
    printf("  %s                               Show in-app Help screen\n", MH(MH_CMD, "h"));
    printf("  %s                               Freeze/Unfreeze display updates\n", MH(MH_CMD, "f"));
    printf("  %s                               Sort process list by currently highlighted column\n",
           MH(MH_CMD, "s"));
    printf("  %s                               Apply current sort criteria to all tabs/modes\n",
           MH(MH_CMD, "S"));
    printf("  %s                             Increase/Decrease update frequency\n", MH(MH_CMD, "+ -"));
    printf("  %s                               Exit milk-procCTRL\n\n", MH(MH_CMD, "x"));

    milk_help_section("Navigation & Column Control", mh_color);
    printf("  %s                           Move process selection cursor\n", MH(MH_CMD, "UP/DN"));
    printf("  %s                       Move column highlight cursor\n", MH(MH_CMD, "LEFT/RGHT"));
    printf("  %s                             Toggle visibility of specific columns (mode-specific)\n",
           MH(MH_CMD, "1-9"));
    printf("  %s                      Cycle through display modes/tabs\n\n", MH(MH_CMD, "CTRL + L/R"));

    milk_help_section("Process Control (on selection or current process)", mh_color);
    printf("  %s                           Select/Unselect current process\n", MH(MH_CMD, "SPACE"));
    printf("  %s                               Unselect all processes\n", MH(MH_CMD, "u"));
    printf("  %s                               Pause/Resume (Writes CTRLval)\n", MH(MH_CMD, "p"));
    printf("  %s                          Step (Writes CTRLval)\n", MH(MH_CMD, "CTRL+S"));
    printf("  %s                               Request clean exit (Writes CTRLval)\n", MH(MH_CMD, "e"));
    printf("  %s                               Send SIGTERM\n", MH(MH_CMD, "T"));
    printf("  %s                               Send SIGKILL\n", MH(MH_CMD, "K"));
    printf("  %s                               Send SIGINT\n", MH(MH_CMD, "I"));
    printf("  %s                           Remove log for selected / all inactive process(es)\n",
           MH(MH_CMD, "r / R"));
    printf("  %s                           Zero counter for selected / all process(es)\n\n", MH(MH_CMD,
            "z / Z"));
}

int main(
    int argc,
    char *argv[])
{
    int action = milk_help_init(argc, argv,
                                "interactive TUI for monitoring and controlling milk processes",
                                "This tool monitors and controls real-time loop processes in the MILK environment.");

    if(action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }

    int mh_color = (action == MH_ACTION_HELP);

    if(action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    int opt;

    // Silence ImageStreamIO library (suppress stderr warnings/errors in TUI)
    // ImageStreamIO_set_verbosity(0);

    static struct option long_options[] =
    {
        {"help",       no_argument,       0, 'h'},
        {"debug",      no_argument,       0, 'd'},
        {"log",        required_argument, 0, 'l'},
        {"check-scan", no_argument,       0, 'c'},
        {0, 0, 0, 0}
    };

    optind = 1; // Ensure getopt starts from beginning

    while((opt = getopt_long(argc, argv, "hdcl:", long_options, NULL)) != -1)
    {
        switch(opt)
        {
        case 'h':
            // Handled by milk_help_init
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
            if(fp != NULL)
            {
                char pid_str[64];
                while(fgets(pid_str, sizeof(pid_str), fp) != NULL)
                {
                    pid_t pid = (pid_t)atoi(pid_str);
                    if(pid != my_pid)
                    {
                        pid_str[strcspn(pid_str, "\n")] = 0;
                        printf("Scanner milk-procCTRL-scan is running (PID %s)\n", pid_str);
                        scan_ok = 1;
                        break;
                    }
                }
                pclose(fp);
            }
            if(!scan_ok)
            {
                printf("Scanner milk-procCTRL-scan is NOT running\n");
            }

            // Check tmux
            fp = popen("tmux -V 2>/dev/null", "r");
            if(fp != NULL)
            {
                char tmux_ver[64];
                if(fgets(tmux_ver, sizeof(tmux_ver), fp) != NULL)
                {
                    tmux_ver[strcspn(tmux_ver, "\n")] = 0;
                    printf("tmux is installed (%s)\n", tmux_ver);
                    tmux_ok = 1;
                }
                else
                {
                    printf("tmux is NOT installed\n");
                }
                pclose(fp);
            }
            else
            {
                printf("tmux is NOT installed\n");
            }

            if(scan_ok && tmux_ok)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }
        case 'l':
            strncpy(procCTRL_logfile, optarg, 1023);
            break;
        case '?':
        default:
            printf("\n\033[1;31mERROR\033[0m: Invalid option.\n\n");
            print_help(argv[0], 1);
            return 1;
        }
    }

    // Run the tool
    processinfo_CTRLscreen();

    return 0;
}
