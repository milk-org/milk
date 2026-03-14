/**
 * @file milk-procinfo-rm.c
 * @brief Milk procinfo rm module
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <getopt.h>
#include <sys/stat.h>
#include <regex.h>

#include "processinfo_internal.h"
#include "processinfo.h"
#include "processinfo_procdirname.h"
#include "processinfo_shm_list_create.h"
#include "milkDebugTools.h"

void print_help(const char *progname) {
    printf("Usage: %s [options] <regex pattern>\n", progname);
    printf("Remove processinfo shared memory segments for given process name(s).\n");
    printf("\n");
    printf("Options:\n");
    printf("  -v, --verbose   Verbose mode\n");
    printf("  -h, --help      Show this help message\n");
}

int main(int argc, char *argv[])
{
    int verbose = 0;
    int opt;

    static struct option long_options[] = {
        {"verbose", no_argument,       0, 'v'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "vh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'v':
                verbose = 1;
                break;
            case 'h':
                print_help(argv[0]);
                return 0;
            default:
                print_help(argv[0]);
                return 1;
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "Error: missing process name.\n");
        print_help(argv[0]);
        return 1;
    }

    const char *pattern = argv[optind];
    regex_t regex;
    int ret = regcomp(&regex, pattern, REG_EXTENDED | REG_NOSUB);
    if (ret != 0) {
        char error_msg[128];
        regerror(ret, &regex, error_msg, sizeof(error_msg));
        fprintf(stderr, "Error: Invalid regular expression. %s\n", error_msg);
        return 1;
    }

    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);

    if (verbose) {
        printf("Scanning directory '%s' to remove processes matching '%s'...\n", procdname, pattern);
    }

    DIR *dir = opendir(procdname);
    if (!dir) {
        perror("opendir");
        return 1;
    }

    struct dirent *entry;

    int removed_count = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "proc.", 5) == 0 &&
            strstr(entry->d_name, ".shm") != NULL) {
            
            // Extract pname from proc.PNAME.XXXXXX.shm
            char ext_pname[256];
            strncpy(ext_pname, entry->d_name + 5, sizeof(ext_pname));
            char *dot = strchr(ext_pname, '.');
            if (dot) *dot = '\0';
            
            if (regexec(&regex, ext_pname, 0, NULL, 0) == 0) {
                // Match found
                char fullpath[STRINGMAXLEN_FULLFILENAME];
                snprintf(fullpath, sizeof(fullpath), "%s/%s", procdname, entry->d_name);
            
                if (verbose) {
                    printf("Removing %s\n", fullpath);
                }
                if (unlink(fullpath) == 0) {
                    removed_count++;
                } else {
                    perror("unlink");
                }
            }
        }
    }
    closedir(dir);

    // Update global list
    if (processinfo_shm_list_create() != -1) {
        if (pinfolist != NULL) {
            for (int i = 0; i < PROCESSINFOLISTSIZE; i++) {
                if (pinfolist->active[i] != 0 && regexec(&regex, pinfolist->pnamearray[i], 0, NULL, 0) == 0) {
                    if (verbose) {
                        printf("Deactivating entry %d in pinfolist (PID %d)\n", i, pinfolist->PIDarray[i]);
                    }
                    pinfolist->active[i] = 0;
                }
            }
        }
    }

    printf("Removed %d shared memory segments for processes matching '%s'.\n", removed_count, pattern);

    regfree(&regex);

    return 0;
}
