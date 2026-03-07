#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include <signal.h>
#include <regex.h>

#include "fps.h"
#include "fps_globals.h"
#include "fps_scan.h"

void print_help(const char *progname) {
    printf("Usage: %s [options] [regex pattern]\n", progname);
    printf("List active FPS instances.\n");
    printf("\n");
    printf("Options:\n");
    printf("  -v, --verbose   Verbose mode (print search directory and details)\n");
    printf("  -e, --exec      Show full path to executable\n");
    printf("  -h, --help      Show this help message\n");
}

int main(int argc, char *argv[])
{
    int verbose = 0;
    int show_exec = 0;
    int opt;

    static struct option long_options[] = {
        {"verbose", no_argument,       0, 'v'},
        {"exec",    no_argument,       0, 'e'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "veh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'v':
                verbose = 1;
                break;
            case 'e':
                show_exec = 1;
                break;
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

    // Initialize fpsarray
    fpsarray = (FUNCTION_PARAMETER_STRUCT *) calloc(NB_FPS_MAX, sizeof(FUNCTION_PARAMETER_STRUCT));
    if(fpsarray == NULL)
    {
        fprintf(stderr, "Error: cannot allocate fpsarray\n");
        return 1;
    }
    for(int i = 0; i < NB_FPS_MAX; i++)
    {
        fpsarray[i].SMfd = -1;
    }

    // Keywnode for scan
    KEYWORD_TREE_NODE *keywnode = (KEYWORD_TREE_NODE *) calloc(NB_KEYWNODE_MAX, sizeof(KEYWORD_TREE_NODE));
    if(keywnode == NULL)
    {
        fprintf(stderr, "Error: cannot allocate keywnode\n");
        free(fpsarray);
        return 1;
    }

    int NBkwn = 0;
    int NBfps = 0;
    long NBpindex = 0;

    // Scan FPS
    // mode 0: scan all
    functionparameter_scan_fps(0, "_ALL", fpsarray, keywnode, &NBkwn, &NBfps, &NBpindex, verbose);

    if (NBfps > 0) {
        if (show_exec) {
            printf("\033[1m%-25s %-30s %-20s %s %s   %s\033[0m\n",
                   "FPS Name", "Executable",
                   "Cmd Key", "   CONF",
                   "    RUN", "Description");
            printf("--------------------------------------"
                   "--------------------------------------"
                   "--------------------------------------"
                   "----\n");
        } else {
            printf("\033[1m%-25s %-30s %-20s %s %s"
                   "   %s\033[0m\n",
                   "FPS Name", "Executable",
                   "Cmd Key", "   CONF",
                   "    RUN", "Description");
            printf("--------------------------------------"
                   "--------------------------------------"
                   "--------------------------------------"
                   "----\n");
        }

        for(int i = 0; i < NBfps; i++)
        {
            if (use_regex && regexec(&regex, fpsarray[i].md->name, 0, NULL, 0) != 0) {
                // Disconnect skipped elements
                function_parameter_struct_disconnect(&fpsarray[i]);
                continue;
            }

            char status_str[128] = "";
            char conf_pid_str[32] = "";
            char run_pid_str[32] = "";
            char tmux_str[32] = "";

            // Extract executable basename
            const char *exec_basename =
                strrchr(
                    fpsarray[i].md->execfullpath,
                    '/');
            if (exec_basename)
                exec_basename++;
            else
                exec_basename =
                    fpsarray[i].md->execfullpath;

            // Check CONF process
            pid_t confpid = fpsarray[i].md->confpid;
            if (confpid > 0 && kill(confpid, 0) == 0) {
                snprintf(conf_pid_str, 32, "%s%7d%s", COLORCOMMAND, (int)confpid, COLORRESET);
            } else {
                snprintf(conf_pid_str, 32, "%7d", (int)confpid);
            }

            // Check RUN process
            pid_t runpid = fpsarray[i].md->runpid;
            if (runpid > 0 && kill(runpid, 0) == 0) {
                snprintf(run_pid_str, 32, "%s%7d%s", COLORCOMMAND, (int)runpid, COLORRESET);
            } else {
                snprintf(run_pid_str, 32, "%7d", (int)runpid);
            }

            // Check tmux session
            char tmux_cmd[256];
            snprintf(tmux_cmd, sizeof(tmux_cmd), "tmux has-session -t %s 2> /dev/null", fpsarray[i].md->name);
            if (system(tmux_cmd) == 0) {
                snprintf(tmux_str, 32, "[%stmu%s]", COLORCOMMAND, COLORRESET);
            } else {
                snprintf(tmux_str, 32, "[---]");
            }

            snprintf(status_str, 128, "%s %s %s", conf_pid_str, run_pid_str, tmux_str);
            
            if (show_exec) {
                printf("\033[36m%-25s\033[0m "
                       "\033[35m%-30s\033[0m "
                       "\033[33m%-20s\033[0m "
                       "%s   %s\n",
                       fpsarray[i].md->name,
                       fpsarray[i].md->execfullpath,
                       fpsarray[i].md->callprogname,
                       status_str,
                       fpsarray[i].md->description);
            } else {
                printf("\033[36m%-25s\033[0m "
                       "\033[35m%-30s\033[0m "
                       "\033[33m%-20s\033[0m "
                       "%s   %s\n",
                       fpsarray[i].md->name,
                       exec_basename,
                       fpsarray[i].md->callprogname,
                       status_str,
                       fpsarray[i].md->description);
            }
            
            // Disconnect to clean up
            function_parameter_struct_disconnect(&fpsarray[i]);
        }
        printf("\n");
    } else {
        if (!verbose) {
            printf("No FPS instances found. Use -v for more details.\n");
        }
    }

    free(keywnode);
    free(fpsarray);

    if (use_regex) {
        regfree(&regex);
    }

    return 0;
}