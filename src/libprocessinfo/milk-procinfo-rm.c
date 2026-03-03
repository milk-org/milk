#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <getopt.h>
#include <sys/stat.h>

#include "processinfo_internal.h"
#include "processinfo.h"
#include "processinfo_procdirname.h"
#include "processinfo_shm_list_create.h"
#include "CommandLineInterface/milkDebugTools.h"

void print_help(const char *progname) {
    printf("Usage: %s [options] <pname>\n", progname);
    printf("Remove processinfo shared memory segments for a given process name.\n");
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

    const char *pname = argv[optind];

    char procdname[STRINGMAXLEN_DIRNAME];
    processinfo_procdirname(procdname);

    if (verbose) {
        printf("Scanning directory '%s' for process '%s'...\n", procdname, pname);
    }

    DIR *dir = opendir(procdname);
    if (!dir) {
        perror("opendir");
        return 1;
    }

    struct dirent *entry;
    char prefix[STRINGMAXLEN_PROCESSINFO_NAME + 10];
    snprintf(prefix, sizeof(prefix), "proc.%s.", pname);

    int removed_count = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, prefix, strlen(prefix)) == 0 &&
            strstr(entry->d_name, ".shm") != NULL) {
            
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
    closedir(dir);

    // Update global list
    if (processinfo_shm_list_create() != -1) {
        if (pinfolist != NULL) {
            for (int i = 0; i < PROCESSINFOLISTSIZE; i++) {
                if (pinfolist->active[i] != 0 && strcmp(pinfolist->pnamearray[i], pname) == 0) {
                    if (verbose) {
                        printf("Deactivating entry %d in pinfolist (PID %d)\n", i, pinfolist->PIDarray[i]);
                    }
                    pinfolist->active[i] = 0;
                }
            }
        }
    }

    printf("Removed %d shared memory segments for process '%s'.\n", removed_count, pname);

    return 0;
}
