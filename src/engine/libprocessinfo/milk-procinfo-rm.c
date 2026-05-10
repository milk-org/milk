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
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <regex.h>

#include "processinfo_internal.h"
#include "processinfo.h"
#include "processinfo_procdirname.h"
#include "processinfo_shm_list_create.h"
#include "milkDebugTools.h"
#include "milk_help.h"

#define PI_RM_DESC \
    "remove processinfo shared-memory entries matching a regex"
#define PI_RM_DESC_LONG \
    "Scan the processinfo directory (e.g. /dev/shm) and remove all\n" \
    "proc.<name>.<pid>.shm files whose base name matches the given\n" \
    "POSIX extended regular expression.\n" \
    "If --clean-dead is used, removes all entries with status CRASHED or STOPPED.\n" \
    "Also deactivates matching entries in the global pinfolist."

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, PI_RM_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] %s<regex>%s\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", PI_RM_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-c, --clean-dead",
           mh_color ? MH_RST : "", "Remove all CRASHED or STOPPED entries");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-v, --verbose",
           mh_color ? MH_RST : "", "Verbose output");
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
    printf("  %s$ milk-procinfo-rm%s %smyproc%s\n\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    const char *see_also[] = { "milk-procinfo-list", "milk-procinfo-info" };
    milk_help_see_also(see_also, 2, mh_color);
}

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv,
                                PI_RM_DESC, PI_RM_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
        return 0;
    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    int verbose = 0;
    int clean_dead = 0;
    int opt;

    static struct option long_options[] = {
        {"clean-dead", no_argument,       0, 'c'},
        {"verbose",    no_argument,       0, 'v'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "cvh",
                              long_options, NULL)) != -1)
    {
        switch (opt)
        {
            case 'c': clean_dead = 1; break;
            case 'v': verbose = 1; break;
            case 'h': break; /* handled above */
            default:
                print_help(argv[0], 0);
                return 1;
        }
    }

    const char *pattern;
    if (optind >= argc)
    {
        if (clean_dead) {
            pattern = ".*";
        } else {
            fprintf(stderr, "Error: missing process name.\n");
            print_help(argv[0], 0);
            return 1;
        }
    }
    else
    {
        pattern = argv[optind];
    }
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
                char fullpath[STRINGMAXLEN_FULLFILENAME + 256];
                snprintf(fullpath, sizeof(fullpath), "%s/%s", procdname, entry->d_name);
            
                int should_remove = 1;

                if (clean_dead) {
                    should_remove = 0;
                    int fd = open(fullpath, O_RDONLY);
                    if (fd != -1) {
                        PROCESSINFO *pinfo = mmap(NULL, sizeof(PROCESSINFO), PROT_READ, MAP_SHARED, fd, 0);
                        if (pinfo != MAP_FAILED) {
                            if (kill(pinfo->PID, 0) == -1 && errno == ESRCH) {
                                // Process no longer exists
                                should_remove = 1;
                            } else if (pinfo->loopstat == PROCESSINFO_LOOPSTAT_CRASHED || 
                                       pinfo->loopstat == PROCESSINFO_LOOPSTAT_STOP) {
                                // Process exists but is explicitly marked crashed/stopped
                                should_remove = 1;
                            }
                            munmap(pinfo, sizeof(PROCESSINFO));
                        }
                        close(fd);
                    }
                }

                if (should_remove) {
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
    }
    closedir(dir);

    // Update global list
    if (processinfo_shm_list_create() != -1) {
        if (pinfolist != NULL) {
            for (int i = 0; i < PROCESSINFOLISTSIZE; i++) {
                if (pinfolist->active[i] != 0 && regexec(&regex, pinfolist->pnamearray[i], 0, NULL, 0) == 0) {
                    
                    int should_deactivate = 1;

                    if (clean_dead) {
                        should_deactivate = 0;
                        char fullpath[STRINGMAXLEN_FULLFILENAME + 256];
                        snprintf(fullpath, sizeof(fullpath), "%s/proc.%s.%d.shm", 
                                 procdname, pinfolist->pnamearray[i], pinfolist->PIDarray[i]);
                        
                        // We check if the file was deleted in the previous step
                        // If it doesn't exist anymore, it means it was deleted.
                        if (access(fullpath, F_OK) != 0) {
                            should_deactivate = 1;
                        }
                    }

                    if (should_deactivate) {
                        if (verbose) {
                            printf("Deactivating entry %d in pinfolist (PID %d)\n", i, pinfolist->PIDarray[i]);
                        }
                        pinfolist->active[i] = 0;
                    }
                }
            }
        }
    }

    printf("Removed %d shared memory segments for processes matching '%s'.\n", removed_count, pattern);

    regfree(&regex);

    return 0;
}
