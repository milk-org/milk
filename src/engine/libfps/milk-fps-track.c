/**
 * @file milk-fps-track.c
 * @brief Milk fps track module
 */

#include <getopt.h>
#include <signal.h>
#include <regex.h>

#include "fps_globals.h"

#define FT_DESC "track and display FPS parameter values in real time"
#define FT_DESC_LONG \
    "Poll all active FPS instances and print any parameter whose value\n" \
    "has changed since the previous scan. Timestamps are in UTC ISO-8601.\n" \
    "An optional POSIX extended regex filters which FPS names are tracked.\n" \
    "Output is one line per change: timestamp, FPS, parameter, value."

/**
 * @brief Print help message for milk-fps-track.
 */
static void print_help(
    const char *progname,
    int        mh_color)
{
    milk_help_banner(progname, FT_DESC, mh_color);
    milk_help_section("Usage", mh_color);
    printf("  %s%s%s [%soptions%s] [%sregex%s]\n\n",
           mh_color ? MH_CMD : "", progname, mh_color ? MH_RST : "",
           mh_color ? MH_OPT : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    milk_help_section("Description", mh_color);
    printf("  %s\n\n", FT_DESC_LONG);
    milk_help_section("Options", mh_color);
    printf("  %s%-25s%s %s\n",
           mh_color ? MH_OPT : "", "-i, --interval SEC",
           mh_color ? MH_RST : "",
           "Polling interval in seconds (default: 0.1)");
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
    printf("  %s$ milk-fps-track%s\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "");
    printf("  %s$ milk-fps-track%s -i 0.5 %smyfps.*%s\n\n",
           mh_color ? MH_CMD : "", mh_color ? MH_RST : "",
           mh_color ? MH_ARG : "", mh_color ? MH_RST : "");
    const char *see_also[] =
    {
        "milk-fps-info:inspect FPS directory contents",
        "milk-fps-set:set an FPS parameter value",
        "milk-fps-list:list active FPS instances"
    };
    milk_help_see_also(see_also, 3, mh_color);
}

#define VALSTR_LEN 256

typedef struct
{
    char keywordfull[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char value[VALSTR_LEN];
    long cnt0;
} PARAM_TRACK;

typedef struct
{
    char name[STRINGMAXLEN_FPS_NAME];
    int active;
    long NBparam;
    PARAM_TRACK *params;
} FPS_TRACK;

FPS_TRACK *track_list = NULL;
int track_list_cnt = 0;

/**
 * @brief Print the current UTC timestamp.
 *
 * Formats as ISO 8601 with microsecond precision.
 */
void print_ut_timestamp()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm *ut_tm = gmtime(&ts.tv_sec);
    printf("%04d%02d%02dT%02d:%02d:%02d.%03ld",
           ut_tm->tm_year + 1900, ut_tm->tm_mon + 1, ut_tm->tm_mday,
           ut_tm->tm_hour, ut_tm->tm_min, ut_tm->tm_sec,
           ts.tv_nsec / 1000000);
}

static volatile int keep_running = 1;
/**
 * @brief SIGINT handler for graceful exit.
 */
void sigint_handler(int sig)
{
    (void)sig;
    keep_running = 0;
}

int main(
    int argc,
    char *argv[])
{
    int action = milk_help_init(argc, argv,
                                FT_DESC, FT_DESC_LONG);
    if(action == MH_ACTION_H1 || action == MH_ACTION_H2)
    {
        return 0;
    }
    int mh_color = (action == MH_ACTION_HELP);
    if(action == MH_ACTION_HELP || action == MH_ACTION_MONO)
    {
        print_help(argv[0], mh_color);
        return 0;
    }

    double interval = 0.1;
    int opt;
    char *regex_pattern = ".*";

    static struct option long_options[] =
    {
        {"interval", required_argument, 0, 'i'},
        {"help",     no_argument,       0, 'h'},
        {"help-oneline", no_argument, 0, '1'},
        {0, 0, 0, 0}
    };

    while((opt = getopt_long(argc, argv, "i:h1",
                             long_options, NULL)) != -1)
    {
        switch(opt)
        {
        case 'i':
            interval = atof(optarg);
            break;
        case 'h':
        case '1':
            break; /* handled above */
        default:
            printf("\n\033[1;31mERROR\033[0m invalid option\n\n");
            print_help(argv[0], 1);
            return 1;
        }
    }

    if(optind < argc)
    {
        regex_pattern = argv[optind];
    }

    regex_t regex;
    int reti = regcomp(&regex, regex_pattern, REG_EXTENDED | REG_NOSUB);
    if(reti)
    {
        PRINT_ERROR("Could not compile regex");
        return 1;
    }

    {
        struct sigaction sa;
        sa.sa_handler = sigint_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);
    }

    // Initialize global arrays for scan
    fpsarray = (FPS *) calloc(NB_FPS_MAX, sizeof(FPS));
    for(int ii = 0; ii < NB_FPS_MAX; ii++)
    {
        fpsarray[ii].SMfd = -1;
    }

    KEYWORD_TREE_NODE *keywnode = (KEYWORD_TREE_NODE *) calloc(NB_KEYWNODE_MAX,
                                  sizeof(KEYWORD_TREE_NODE));

    track_list = (FPS_TRACK *) calloc(NB_FPS_MAX, sizeof(FPS_TRACK));

    printf("Tracking FPS matching pattern: %s\n", regex_pattern);
    printf("Tracked instances:\n");

    int first_scan = 1;

    while(keep_running)
    {
        int NBkwn = 0;
        int NBfps = 0;
        long NBpindex = 0;

        // Scan FPS
        functionparameter_scan_fps(0, "_ALL", fpsarray, keywnode, &NBkwn, &NBfps, &NBpindex, 0);

        // Mark all tracked as inactive
        for(int ii = 0; ii < track_list_cnt; ii++)
        {
            track_list[ii].active = 0;
        }

        for(int ii = 0; ii < NBfps; ii++)
        {
            // Check regex
            if(regexec(&regex, fpsarray[ii].md->name, 0, NULL, 0) != 0)
            {
                continue;
            }

            int track_idx = -1;
            // Find existing tracker
            for(int jj = 0; jj < track_list_cnt; jj++)
            {
                if(strcmp(track_list[jj].name, fpsarray[ii].md->name) == 0)
                {
                    track_idx = jj;
                    break;
                }
            }

            if(track_idx == -1)
            {
                // New FPS found
                track_idx = track_list_cnt++;
                strncpy(track_list[track_idx].name,
                        fpsarray[ii].md->name,
                        STRINGMAXLEN_FPS_NAME - 1);
                track_list[track_idx].NBparam = fpsarray[ii].md->NBparamMAX;
                track_list[track_idx].params = (PARAM_TRACK *) calloc(fpsarray[ii].md->NBparamMAX,
                                               sizeof(PARAM_TRACK));

                if(first_scan)
                {
                    printf("  - %s\n", track_list[track_idx].name);
                }
                else
                {
                    print_ut_timestamp();
                    printf(" NEW_FPS %s tracked\n", track_list[track_idx].name);
                }

                // Initialize values
                for(int pp = 0; pp < fpsarray[ii].md->NBparamMAX; pp++)
                {
                    if(fpsarray[ii].parray[pp].fpflag & FPFLAG_ACTIVE)
                    {
                        strncpy(track_list[track_idx].params[pp].keywordfull,
                                fpsarray[ii].parray[pp].keywordfull,
                                sizeof(track_list[0].params[0].keywordfull) - 1);
                        track_list[track_idx].params[pp].cnt0 = fpsarray[ii].parray[pp].cnt0;
                        functionparameter_GetParamValueString(
                            &fpsarray[ii].parray[pp],
                            track_list[track_idx].params[pp].value,
                            VALSTR_LEN);
                    }
                    else
                    {
                        track_list[track_idx].params[pp].cnt0 = -1;
                        track_list[track_idx].params[pp].keywordfull[0] = '\0';
                    }
                }
            }
            track_list[track_idx].active = 1;

            if(fpsarray[ii].md->NBparamMAX != track_list[track_idx].NBparam)
            {
                PARAM_TRACK *tmp = (PARAM_TRACK *) realloc(
                                       track_list[track_idx].params,
                                       fpsarray[ii].md->NBparamMAX
                                       * sizeof(PARAM_TRACK));
                if(tmp == NULL)
                {
                    PRINT_ERROR("realloc failed for %s params", track_list[track_idx].name);
                    continue;
                }
                track_list[track_idx].params = tmp;
                // Initialize new ones if any
                for(int pp = track_list[track_idx].NBparam; pp < fpsarray[ii].md->NBparamMAX; pp++)
                {
                    if(fpsarray[ii].parray[pp].fpflag & FPFLAG_ACTIVE)
                    {
                        strncpy(track_list[track_idx].params[pp].keywordfull,
                                fpsarray[ii].parray[pp].keywordfull,
                                sizeof(track_list[0].params[0].keywordfull) - 1);
                        track_list[track_idx].params[pp].cnt0 = fpsarray[ii].parray[pp].cnt0;
                        functionparameter_GetParamValueString(
                            &fpsarray[ii].parray[pp],
                            track_list[track_idx].params[pp].value,
                            VALSTR_LEN);
                    }
                    else
                    {
                        track_list[track_idx].params[pp].keywordfull[0] = '\0';
                        track_list[track_idx].params[pp].cnt0 = -1;
                    }
                }
                track_list[track_idx].NBparam = fpsarray[ii].md->NBparamMAX;
            }

            for(int pp = 0; pp < fpsarray[ii].md->NBparamMAX; pp++)
            {
                if(!(fpsarray[ii].parray[pp].fpflag & FPFLAG_ACTIVE))
                {
                    continue;
                }

                if(fpsarray[ii].parray[pp].cnt0 != track_list[track_idx].params[pp].cnt0)
                {
                    char current_val[VALSTR_LEN];
                    functionparameter_GetParamValueString(
                        &fpsarray[ii].parray[pp],
                        current_val,
                        VALSTR_LEN);

                    if(track_list[track_idx].params[pp].cnt0 != -1)
                    {
                        print_ut_timestamp();
                        printf(" %s %s : %s -> %s  (cnt: %ld)\n",
                               track_list[track_idx].name,
                               fpsarray[ii].parray[pp].keywordfull,
                               track_list[track_idx].params[pp].value,
                               current_val,
                               fpsarray[ii].parray[pp].cnt0);
                        fflush(stdout);
                    }

                    strncpy(track_list[track_idx].params[pp].keywordfull,
                            fpsarray[ii].parray[pp].keywordfull,
                            sizeof(track_list[0].params[0].keywordfull) - 1);
                    strncpy(track_list[track_idx].params[pp].value,
                            current_val,
                            VALSTR_LEN - 1);
                    track_list[track_idx].params[pp].cnt0 = fpsarray[ii].parray[pp].cnt0;
                }
            }
        }

        if(first_scan)
        {
            printf("\n");
            first_scan = 0;
        }

        // Detect deleted FPS
        for(int jj = 0; jj < track_list_cnt; jj++)
        {
            if(!track_list[jj].active &&
                    track_list[jj].name[0] != '\0')
            {
                print_ut_timestamp();
                printf(" DEL_FPS %s\n",
                       track_list[jj].name);
                fflush(stdout);
                if(track_list[jj].params != NULL)
                {
                    free(track_list[jj].params);
                    track_list[jj].params = NULL;
                }
                track_list[jj].NBparam = 0;
                track_list[jj].name[0] = '\0';
            }
        }

        usleep((useconds_t)(interval * 1000000));
    }

    // Final cleanup
    regfree(&regex);
    for(int jj = 0; jj < track_list_cnt; jj++)
    {
        if(track_list[jj].params)
        {
            free(track_list[jj].params);
        }
    }
    free(track_list);
    free(keywnode);
    free(fpsarray);

    return 0;
}
