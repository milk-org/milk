#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <regex.h>
#include <getopt.h>

#include "processinfo.h"
#include "processinfo_shm_list_create.h"

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

    printf("%-30s %-10s %-10s\n", "Process Name", "PID", "Status");
    printf("------------------------------------------------------------\n");

    if (pinfolist != NULL) {
        for (long i = 0; i < PROCESSINFOLISTSIZE; i++) {
            if (pinfolist->active[i] != 0) {
                if (use_regex && regexec(&regex, pinfolist->pnamearray[i], 0, NULL, 0) != 0) {
                    continue; // Skip if it doesn't match the regex
                }
                
                char status_str[32];
                switch(pinfolist->active[i]) {
                    case 1: strcpy(status_str, "ACTIVE"); break;
                    case 2: strcpy(status_str, "STOPPED"); break;
                    case 3: strcpy(status_str, "CRASHED"); break;
                    default: strcpy(status_str, "UNKNOWN"); break;
                }
                
                printf("%-30s %-10ld %s\n", 
                    pinfolist->pnamearray[i], 
                    (long)pinfolist->PIDarray[i], 
                    status_str);
            }
        }
    }

    if (use_regex) regfree(&regex);

    return 0;
}
