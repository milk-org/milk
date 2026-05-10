/**
 * @file milk-procinfo-list.c
 * @brief Milk procinfo list module
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <regex.h>
#include <getopt.h>

#include "processinfo.h"
#include "processinfo_shm_list_create.h"
#include "milk_help.h"

#define C_TITLE MH_TITLE
#define C_HDR   MH_HDR
#define C_NAME  MH_CMD
#define C_TYPE  MH_NOTE
#define C_ERR   MH_ERR
#define C_DIM   MH_DIM
#define C_RST   MH_RST

#define PIL_DESC "list processinfo shared-memory entries"
#define PIL_DESC_LONG \
    "Scan the processinfo shared-memory list and print a summary table.\n" \
    "Each row shows process name, PID, and status (RUNNING/STOPPED/CRASHED).\n" \
    "An optional POSIX regex filters which process names are shown."

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, PIL_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] [%sregex%s]\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", PIL_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h, --help",
           mh_color ? MH_RST : "", "Show this help and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "One-line description and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Verbose description and exit");
    printf("  %s%-25s%s %s\n\n",
           mh_color ? MH_OPT : "", "-hm, --help-mono",
           mh_color ? MH_RST : "", "Full help, no ANSI color");
    milk_help_section("Examples", mh_color);
    printf("  %s$ milk-procinfo-list%s\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-procinfo-list%s %smyproc.*%s\n\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    const char *see_also[] = {
        "milk-procinfo-info", "milk-procinfo-rm", "milk-procCTRL"
    };
    milk_help_see_also(see_also, 3, mh_color);
}

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv,
                                PIL_DESC, PIL_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
        return 0;
    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    int opt;

    static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "h",
                              long_options, NULL)) != -1)
    {
        switch (opt)
        {
            case 'h': break; /* handled above */
            default:
                print_help(argv[0], 0);
                return 1;
        }
    }

    const char *pattern = NULL;
    regex_t regex;
    int use_regex = 0;

    if (optind < argc) {
        pattern = argv[optind];
        int ret = regcomp(&regex, pattern, REG_EXTENDED | REG_NOSUB);
        if (ret != 0) {
            char error_msg[128];
            regerror(ret, &regex, error_msg, sizeof(error_msg));
            fprintf(stderr, "Error: Invalid regular expression. %s\n", error_msg);
            return 1;
        }
        use_regex = 1;
    }

    if (processinfo_shm_list_create() == -1) {
        fprintf(stderr, "Error connecting to process list shared memory\n");
        return 1;
    }

    printf(C_TITLE "%-30s %-10s %-10s" C_RST "\n", "Process Name", "PID", "Status");
    printf(C_DIM);
    for (int i=0; i<60; i++) putchar('-');
    printf(C_RST "\n");

    if (pinfolist != NULL) {
        for (long i = 0; i < PROCESSINFOLISTSIZE; i++) {
            if (pinfolist->active[i] != 0) {
                if (use_regex && regexec(&regex, pinfolist->pnamearray[i], 0, NULL, 0) != 0) {
                    continue; // Skip if it doesn't match the regex
                }
                char status_str[32];
                char pid_color[16] = "";
                char pid_reset[16] = "";
                switch(pinfolist->active[i]) {
                    case 1: 
                        snprintf(status_str, 32, C_NAME "RUNNING" C_RST); 
                        snprintf(pid_color,
                                 sizeof(pid_color),
                                 "%s", C_NAME);
                        snprintf(pid_reset,
                                 sizeof(pid_reset),
                                 "%s", C_RST);
                        break;
                    case 2: snprintf(status_str, 32, C_TYPE "STOPPED" C_RST); break;
                    case 3: snprintf(status_str, 32, C_ERR "CRASHED" C_RST); break;
                    default: snprintf(status_str, 32, C_HDR "UNKNOWN" C_RST); break;
                }
                
                printf(C_NAME "%-30s" C_RST " %s%-10ld%s %s\n", 
                    pinfolist->pnamearray[i], 
                    pid_color,
                    (long)pinfolist->PIDarray[i], 
                    pid_reset,
                    status_str);
            }
        }
    }

    if (use_regex) regfree(&regex);

    return 0;
}
