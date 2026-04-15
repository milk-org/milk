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

/* ANSI color codes */
#define C_TITLE "\033[1;36m"  /* Cyan Bold   */
#define C_HDR   "\033[1;34m"  /* Blue Bold   */
#define C_NAME  "\033[1;32m"  /* Green Bold  */
#define C_TYPE  "\033[1;33m"  /* Yellow Bold */
#define C_SIZE  "\033[1m"     /* White Bold  */
#define C_CNT   "\033[1;35m"  /* Magenta Bold */
#define C_SEM   "\033[36m"    /* Cyan        */
#define C_LINK  "\033[36m"    /* Cyan        */
#define C_ERR   "\033[1;31m"  /* Red Bold    */
#define C_DIM   "\033[2m"     /* Dim         */
#define C_RST   "\033[0m"     /* Reset       */

void print_help(const char *progname) {
    printf("Usage: %s [options] [regex pattern]\n", progname);
    printf("List active processinfo instances.\n");
    printf("\n");
    printf("Options:\n");
    printf("  -h, --help      Show this help message\n");
}

int main(int argc, char *argv[])
{
    int opt;

    static struct option long_options[] = {
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'h':
                print_help(argv[0]);
                return 0;
            default:
                print_help(argv[0]);
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
