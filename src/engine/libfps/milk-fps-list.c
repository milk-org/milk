/**
 * @file milk-fps-list.c
 * @brief Milk fps list module
 */

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
#include "milk_help.h"

/* local color aliases -- keep for table rendering, map to MH_* */
#define C_TITLE MH_TITLE
#define C_HDR   MH_HDR
#define C_NAME  MH_CMD
#define C_TYPE  MH_NOTE
#define C_SIZE  MH_BOLD
#define C_CNT   MH_ARG
#define C_SEM   MH_DFLT
#define C_LINK  MH_DFLT
#define C_ERR   MH_ERR
#define C_DIM   MH_DIM
#define C_RST   MH_RST

#define FL_DESC "list Function Parameter Structures (FPS) in shared memory"
#define FL_DESC_LONG \
    "Scan /dev/shm for active FPS instances and print a summary table.\n" \
    "Each row shows FPS name, executable, CLI key, conf/run PIDs,\n" \
    "tmux status, and short description.\n" \
    "An optional regex pattern filters which FPS names are shown."

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, FL_DESC, mh_color);

    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] [%sregex%s]\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    milk_help_section("Description", mh_color);
    printf("  %s\n\n", FL_DESC_LONG);

    milk_help_section("Arguments", mh_color);
    printf("  %s%-14s%s %s\n\n",
           mh_color ? MH_ARG : "", "[regex]",
           mh_color ? MH_RST : "",
           "Optional POSIX extended regex to filter by FPS name");

    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-v, --verbose",
           mh_color ? MH_RST : "",
           "Verbose (print search directory and details)");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-e, --exec",
           mh_color ? MH_RST : "", "Show full path to executable");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h, --help",
           mh_color ? MH_RST : "", "Show this help and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h1, --help-oneline",
           mh_color ? MH_RST : "", "Print one-line description and exit");
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-h2, --help-description",
           mh_color ? MH_RST : "", "Print verbose description and exit");
    printf("  %s%-25s%s %s\n\n",
           mh_color ? MH_OPT : "", "-hm, --help-mono",
           mh_color ? MH_RST : "", "Full help, no ANSI color");

    milk_help_section("Examples", mh_color);
    printf("  %s$ %smilk-fps-list%s\n",
           mh_color ? MH_DIM : "",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "");
    printf("  %s$ %smilk-fps-list%s %smyfps.*%s\n\n",
           mh_color ? MH_DIM : "",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");

    const char *see_also[] = {
        "milk-fps-info:inspect FPS directory contents",
        "milk-fps-rm:remove an FPS instance",
        "milk-fps-set:set an FPS parameter value",
        "milk-fpsCTRL:launch the FPS dashboard TUI"
    };
    milk_help_see_also(see_also, 4, mh_color);
}

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv,
                                FL_DESC, FL_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }
    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    int verbose = 0;
    int show_exec = 0;
    int opt;

    static struct option long_options[] = {
        {"verbose", no_argument,       0, 'v'},
        {"exec",    no_argument,       0, 'e'},
        {"help",    no_argument,       0, 'h'},
        {"help-oneline", no_argument, 0, '1'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "veh1",
                              long_options, NULL)) != -1)
    {
        switch (opt)
        {
            case 'v':
                verbose = 1;
                break;
            case 'e':
                show_exec = 1;
                break;
            case 'h':
            case '1':
                /* Handled above by milk_help_init() */
                break;
            default:
                printf("\n\033[1;31mERROR\033[0m invalid option\n\n");
                print_help(argv[0], 1);
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
            PRINT_ERROR("Error: Invalid regular expression. %s", error_msg);
            return 1;
        }
        use_regex = 1;
    }

    // Initialize fpsarray
    fpsarray = (FPS *) calloc(NB_FPS_MAX, sizeof(FPS));
    if(fpsarray == NULL)
    {
        PRINT_ERROR("Error: cannot allocate fpsarray");
        return 1;
    }
    for(int ii = 0; ii < NB_FPS_MAX; ii++)
    {
        fpsarray[ii].SMfd = -1;
    }

    // Keywnode for scan
    KEYWORD_TREE_NODE *keywnode = (KEYWORD_TREE_NODE *) calloc(NB_KEYWNODE_MAX, sizeof(KEYWORD_TREE_NODE));
    if(keywnode == NULL)
    {
        PRINT_ERROR("Error: cannot allocate keywnode");
        free(fpsarray);
        return 1;
    }

    int NBkwn = 0;
    int NBfps = 0;
    long NBpindex = 0;

    // Scan FPS
    // mode 0: scan all
    functionparameter_scan_fps(0, "_ALL", fpsarray, keywnode, &NBkwn, &NBfps, &NBpindex, verbose);

    printf(C_TITLE "%-25s %-30s %-20s %s %s %s   %s" C_RST "\n",
           "FPS Name", "Executable",
           "Cmd Key", "   CONF",
           "    RUN", "P", "Description");

    printf(C_DIM);
    for (int ii=0; ii<116; ii++) putchar('-');
    printf(C_RST "\n");

    if (NBfps > 0) {
        for(int ii = 0; ii < NBfps; ii++)
        {
            if (use_regex && regexec(&regex, fpsarray[ii].md->name, 0, NULL, 0) != 0) {
                // Disconnect skipped elements
                fps_disconnect(&fpsarray[ii]);
                continue;
            }

            char status_str[256] = "";
            char conf_pid_str[32] = "";
            char run_pid_str[32] = "";
            char tmux_str[32] = "";
            char proc_str[32] = "";

            // Extract executable basename

            const char *exec_basename =
                strrchr(
                    fpsarray[ii].md->execfullpath,
                    '/');
            if (exec_basename)
                exec_basename++;
            else
                exec_basename =
                    fpsarray[ii].md->execfullpath;

            // Check CONF process
            pid_t confpid = fpsarray[ii].md->confpid;
            if (confpid > 0 && kill(confpid, 0) == 0) {
                snprintf(conf_pid_str, 32, "%s%7d%s", COLORCOMMAND, (int)confpid, COLORRESET);
            } else {
                snprintf(conf_pid_str, 32, "%7d", (int)confpid);
            }

            // Check RUN process
            pid_t runpid = fpsarray[ii].md->runpid;
            if (runpid > 0 && kill(runpid, 0) == 0) {
                snprintf(run_pid_str, 32, "%s%7d%s", COLORCOMMAND, (int)runpid, COLORRESET);
            } else {
                snprintf(run_pid_str, 32, "%7d", (int)runpid);
            }

            // Check tmux session
            char tmux_cmd[256];
            snprintf(tmux_cmd, sizeof(tmux_cmd), "tmux has-session -t %s 2> /dev/null", fpsarray[ii].md->name);
            if (system(tmux_cmd) == 0) {
                snprintf(tmux_str, 32, "[%stmu%s]", COLORCOMMAND, COLORRESET);
            } else {
                snprintf(tmux_str, 32, "[---]");
            }

            // Check processinfo
            if (fpsarray[ii].parray != NULL && functionparameter_GetParamIndex(&fpsarray[ii], ".procinfo.enabled") != -1) {
                snprintf(proc_str, 32, "%sP%s", C_NAME, C_RST);
            } else {
                snprintf(proc_str, 32, "%s-%s", C_DIM, C_RST);
            }

            snprintf(status_str, 256, "%s %s %s  %s ", conf_pid_str, run_pid_str, tmux_str, proc_str);

            
            if (show_exec) {
                printf(C_NAME "%-25s" C_RST " "
                       C_TYPE "%-30s" C_RST " "
                       C_HDR "%-20s" C_RST " "
                       "%s   %s\n",
                       fpsarray[ii].md->name,
                       fpsarray[ii].md->execfullpath,
                       fpsarray[ii].md->callprogname,
                       status_str,
                       fpsarray[ii].md->description);
            } else {
                printf(C_NAME "%-25s" C_RST " "
                       C_TYPE "%-30s" C_RST " "
                       C_HDR "%-20s" C_RST " "
                       "%s   %s\n",
                       fpsarray[ii].md->name,
                       exec_basename,
                       fpsarray[ii].md->callprogname,
                       status_str,
                       fpsarray[ii].md->description);
            }
            
            // Disconnect to clean up
            fps_disconnect(&fpsarray[ii]);
        }
        printf("\n");
    }

    free(keywnode);
    free(fpsarray);

    if (use_regex) {
        regfree(&regex);
    }

    return 0;
}