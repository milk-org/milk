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
    printf("Usage: %s [options] <regex pattern>\n", progname);
    printf("Search for parameters across all active FPS instances matching a regex pattern.\n");
    printf("The regex pattern is matched against the full parameter name, which is formatted as:\n");
    printf("  <fpsname>.<parameter_key>\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s \".*\"\n", progname);
    printf("      Match all parameters in all active FPS instances.\n");
    printf("  %s \"^myfps\\.\"\n", progname);
    printf("      Match all parameters in the FPS named 'myfps'.\n");
    printf("  %s \"\\.procinfo\\.\"\n", progname);
    printf("      Match all parameters containing '.procinfo.' across all FPS instances.\n");
    printf("  %s \"^myfps\\.procinfo\\.enabled$\" \n", progname);
    printf("      Match the exact parameter 'procinfo.enabled' in the FPS 'myfps'.\n");
    printf("\n");
    printf("Options:\n");
    printf("  -v, --verbose   Verbose mode (print search details)\n");
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
        fprintf(stderr, "Error: missing regex pattern argument.\n");
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

    // Initialize fpsarray
    fpsarray = (FUNCTION_PARAMETER_STRUCT *) calloc(NB_FPS_MAX, sizeof(FUNCTION_PARAMETER_STRUCT));
    if(fpsarray == NULL)
    {
        fprintf(stderr, "Error: cannot allocate fpsarray\n");
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
        fprintf(stderr, "Error: cannot allocate keywnode\n");
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
        FUNCTION_PARAMETER_STRUCT* fps = &fpsarray[i];
        if (fps == NULL || fps->md == NULL || fps->parray == NULL) {
            function_parameter_struct_disconnect(fps);
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
        function_parameter_struct_disconnect(fps);
    }

    if (!match_found && verbose) {
        printf("No parameters matched the pattern '%s'.\n", pattern);
    }

    free(keywnode);
    free(fpsarray);
    regfree(&regex);

    return 0;
}
