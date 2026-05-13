/**
 * @file milk-fps-search.c
 * @brief Milk fps search module
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include <regex.h>

#include "fps.h"
#include "fps_globals.h"
#include "fps_scan.h"
#include "fps_printparameter_valuestring.h"
#include "milk_help.h"

#define C_TITLE MH_TITLE
#define C_HDR   MH_HDR
#define C_NAME  MH_CMD
#define C_TYPE  MH_NOTE
#define C_DIM   MH_DIM
#define C_RST   MH_RST

#define FS_DESC "search FPS parameters for matching values"
#define FS_DESC_LONG \
    "Scan all active FPS instances in /dev/shm and print every\n" \
    "parameter whose full name (<fpsname>.<key>) matches the given\n" \
    "POSIX extended regular expression.\n" \
    "Output columns: keyword, type, current value, description."

static void print_help(const char *progname, int mh_color)
{
    milk_help_banner(progname, FS_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] %s<regex>%s\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", FS_DESC_LONG);
    milk_help_section("Options", mh_color);
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
    printf("  %s$ milk-fps-search %s\".*\"%s\n",
           mh_color ? MH_DIM : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-fps-search %s\"^myfps\\\\.\"\n%s\n",
           mh_color ? MH_DIM : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    const char *see_also[] = { "milk-fps-list", "milk-fps-info" };
    milk_help_see_also(see_also, 2, mh_color);
}

int main(int argc, char *argv[])
{
    int action = milk_help_init(argc, argv,
                                FS_DESC, FS_DESC_LONG);
    if (action == MH_ACTION_H1 || action == MH_ACTION_H2)
        return 0;
    int mh_color = (action == MH_ACTION_HELP);
    if (action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    int verbose = 0;
    int opt;

    static struct option long_options[] = {
        {"verbose", no_argument,       0, 'v'},
        {"help",    no_argument,       0, 'h'},
        {"help-oneline", no_argument, 0, '1'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "vh1",
                              long_options, NULL)) != -1)
    {
        switch (opt)
        {
            case 'v': verbose = 1; break;
            case 'h':
            case '1': break; /* handled above */
            default:
                printf("\n\033[1;31mERROR\033[0m invalid option\n\n");
                print_help(argv[0], 1);
                return 1;
        }
    }

    if (optind >= argc)
    {
        printf("\n\033[1;31mERROR\033[0m missing regex pattern argument\n\n");
        print_help(argv[0], 1);
        return 1;
    }

    const char *pattern = argv[optind];
    regex_t regex;

    int ret = regcomp(&regex, pattern, REG_EXTENDED | REG_NOSUB);
    if (ret != 0) {
        char error_msg[128];
        regerror(ret, &regex, error_msg, sizeof(error_msg));
        PRINT_ERROR("Error: Invalid regular expression. %s", error_msg);
        return 1;
    }

    // Initialize fpsarray
    fpsarray = (FPS *) calloc(NB_FPS_MAX, sizeof(FPS));
    if(fpsarray == NULL)
    {
        PRINT_ERROR("Error: cannot allocate fpsarray");
        regfree(&regex);
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
        PRINT_ERROR("Error: cannot allocate keywnode");
        free(fpsarray);
        regfree(&regex);
        return 1;
    }

    int NBkwn = 0;
    int NBfps = 0;
    long NBpindex = 0;

    // Scan FPS
    // mode 0: scan all
    functionparameter_scan_fps(0, "_ALL", fpsarray, keywnode, &NBkwn, &NBfps, &NBpindex, verbose);

    if (NBfps == 0) {
        if (verbose) {
            printf("No connected FPSs found.\n");
        }
        free(keywnode);
        free(fpsarray);
        regfree(&regex);
        return 0;
    }

    int match_found = 0;
    
    // widths for formatting
    int kw_width = 30;
    int val_width = 20;

    for(int i = 0; i < NBfps; i++)
    {
        FPS* fps = &fpsarray[i];
        if (fps == NULL || fps->md == NULL || fps->parray == NULL) {
            fps_disconnect(fps);
            continue;
        }

        int fps_has_match = 0;

        for (int pindex = 0; pindex < fps->md->NBparamMAX; pindex++) {
            if (fps->parray[pindex].fpflag & FPFLAG_USED) {
                const char *display_keyword = fps->parray[pindex].keywordfull;
                
                // Try matching full keyword against regex
                if (regexec(&regex, display_keyword, 0, NULL, 0) == 0) {
                    
                    if (!fps_has_match) {
                        // Print FPS header once
                        printf(C_TITLE "========================================================\n" C_RST);
                        printf(C_TITLE " %-20s : " C_HDR "%s" C_RST "\n", "FPS Name", fps->md->name);
                        printf(C_TITLE "========================================================\n" C_RST);
                        
                        printf("%-30s %12s %-20s %s\n", "Keyword", "Type", "Value", "Description");
                        for (int k=0; k<80; k++) printf("-");
                        printf("\n");

                        fps_has_match = 1;
                        match_found = 1;
                    }

                    // Remove prefix for display if it matches FPS name
                    const char *short_keyword = display_keyword;
                    int prefix_len = strlen(fps->md->name);
                    if (strncmp(display_keyword, fps->md->name, prefix_len) == 0 && display_keyword[prefix_len] == '.') {
                        short_keyword += prefix_len + 1;
                    }

                    char valstring[200];
                    if (fps->parray[pindex].type == FPTYPE_STREAMNAME) {
                        snprintf(valstring, 200, "%s", fps->parray[pindex].val.string[0]);
                    } else {
                        functionparameter_GetParamValueString(
                            &fps->parray[pindex],
                            valstring,
                            200);
                    }
                    
                    const char* type_str = "UNKNOWN";
                    switch(fps->parray[pindex].type) {
                        case FPTYPE_UNDEF: type_str = "UNDEF"; break;
                        case FPTYPE_INT32: type_str = "INT32"; break;
                        case FPTYPE_UINT32: type_str = "UINT32"; break;
                        case FPTYPE_INT64: type_str = "INT64"; break;
                        case FPTYPE_UINT64: type_str = "UINT64"; break;
                        case FPTYPE_FLOAT32: type_str = "FLOAT32"; break;
                        case FPTYPE_FLOAT64: type_str = "FLOAT64"; break;
                        case FPTYPE_PID: type_str = "PID"; break;
                        case FPTYPE_TIMESPEC: type_str = "TIMESPEC"; break;
                        case FPTYPE_FILENAME: type_str = "FILENAME"; break;
                        case FPTYPE_FITSFILENAME: type_str = "FITSFILENAME"; break;
                        case FPTYPE_EXECFILENAME: type_str = "EXECFILENAME"; break;
                        case FPTYPE_DIRNAME: type_str = "DIRNAME"; break;
                        case FPTYPE_STREAMNAME: type_str = "STREAMNAME"; break;
                        case FPTYPE_STRING: type_str = "STRING"; break;
                        case FPTYPE_ONOFF: type_str = "ONOFF"; break;
                        case FPTYPE_PROCESS: type_str = "PROCESS"; break;
                        case FPTYPE_FPSNAME: type_str = "FPSNAME"; break;
                        case FPTYPE_STRING_NOT_STREAM: type_str = "STRING_NOT_STREAM"; break;
                    }

                    printf(C_NAME "%-*s" C_RST " %12s %-*s %s\n",
                           kw_width,
                           short_keyword,
                           type_str,
                           val_width,
                           valstring,
                           fps->parray[pindex].description);
                }
            }
        }
        
        if (fps_has_match) {
            printf("\n");
        }

        // Disconnect to clean up
        fps_disconnect(fps);
    }

    if (!match_found && verbose) {
        printf("No parameters matched the pattern '%s'.\n", pattern);
    }

    free(keywnode);
    free(fpsarray);
    regfree(&regex);

    return 0;
}
