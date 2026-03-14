/**
 * @file milk-fps-track.c
 * @brief Milk fps track module
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include <time.h>
#include <signal.h>
#include <regex.h>

#include "fps.h"
#include "fps_globals.h"
#include "fps_scan.h"

#define VALSTR_LEN 256

typedef struct {
    char keywordfull[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char value[VALSTR_LEN];
    long cnt0;
} PARAM_TRACK;

typedef struct {
    char name[STRINGMAXLEN_FPS_NAME];
    int active;
    long NBparam;
    PARAM_TRACK *params;
} FPS_TRACK;

FPS_TRACK *track_list = NULL;
int track_list_cnt = 0;

void print_ut_timestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm *ut_tm = gmtime(&ts.tv_sec);
    printf("%04d%02d%02dT%02d:%02d:%02d.%03ld",
           ut_tm->tm_year + 1900, ut_tm->tm_mon + 1, ut_tm->tm_mday,
           ut_tm->tm_hour, ut_tm->tm_min, ut_tm->tm_sec,
           ts.tv_nsec / 1000000);
}

void print_help(const char *progname) {
    printf("Usage: %s [options] [regex_pattern]\n", progname);
    printf("Monitor FPS parameter changes.\n");
    printf("\n");
    printf("Options:\n");
    printf("  -i, --interval SEC   Polling interval in seconds (default 0.1)\n");
    printf("  -h, --help           Show this help message\n");
    printf("\n");
    printf("Arguments:\n");
    printf("  regex_pattern        Optional regex to filter FPS names (default \".*\")\n");
}

static volatile int keep_running = 1;
void sigint_handler(int sig) {
    (void)sig;
    keep_running = 0;
}

int main(int argc, char *argv[])
{
    double interval = 0.1;
    int opt;
    char *regex_pattern = ".*";

    static struct option long_options[] = {
        {"interval", required_argument, 0, 'i'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "i:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'i':
                interval = atof(optarg);
                break;
            case 'h':
                print_help(argv[0]);
                return 0;
            default:
                print_help(argv[0]);
                return 1;
        }
    }

    if (optind < argc) {
        regex_pattern = argv[optind];
    }

    regex_t regex;
    int reti = regcomp(&regex, regex_pattern, REG_EXTENDED | REG_NOSUB);
    if (reti) {
        fprintf(stderr, "Could not compile regex\n");
        return 1;
    }

    signal(SIGINT, sigint_handler);

    // Initialize global arrays for scan
    fpsarray = (FUNCTION_PARAMETER_STRUCT *) calloc(NB_FPS_MAX, sizeof(FUNCTION_PARAMETER_STRUCT));
    for(int i = 0; i < NB_FPS_MAX; i++) fpsarray[i].SMfd = -1;

    KEYWORD_TREE_NODE *keywnode = (KEYWORD_TREE_NODE *) calloc(NB_KEYWNODE_MAX, sizeof(KEYWORD_TREE_NODE));

    track_list = (FPS_TRACK *) calloc(NB_FPS_MAX, sizeof(FPS_TRACK));

    printf("Tracking FPS matching pattern: %s\n", regex_pattern);
    printf("Tracked instances:\n");

    int first_scan = 1;

    while(keep_running) {
        int NBkwn = 0;
        int NBfps = 0;
        long NBpindex = 0;

        // Scan FPS
        functionparameter_scan_fps(0, "_ALL", fpsarray, keywnode, &NBkwn, &NBfps, &NBpindex, 0);

        // Mark all tracked as inactive
        for(int i = 0; i < track_list_cnt; i++) track_list[i].active = 0;

        for(int i = 0; i < NBfps; i++) {
            // Check regex
            if (regexec(&regex, fpsarray[i].md->name, 0, NULL, 0) != 0) {
                continue;
            }

            int track_idx = -1;
            // Find existing tracker
            for(int j = 0; j < track_list_cnt; j++) {
                if(strcmp(track_list[j].name, fpsarray[i].md->name) == 0) {
                    track_idx = j;
                    break;
                }
            }

            if(track_idx == -1) {
                // New FPS found
                track_idx = track_list_cnt++;
                strncpy(track_list[track_idx].name,
                    fpsarray[i].md->name,
                    STRINGMAXLEN_FPS_NAME - 1);
                track_list[track_idx].NBparam = fpsarray[i].md->NBparamMAX;
                track_list[track_idx].params = (PARAM_TRACK *) calloc(fpsarray[i].md->NBparamMAX, sizeof(PARAM_TRACK));
                
                if (first_scan) {
                    printf("  - %s\n", track_list[track_idx].name);
                } else {
                    print_ut_timestamp();
                    printf(" NEW_FPS %s tracked\n", track_list[track_idx].name);
                }

                // Initialize values
                for(int p = 0; p < fpsarray[i].md->NBparamMAX; p++) {
                    if (fpsarray[i].parray[p].fpflag & FPFLAG_ACTIVE) {
                        strncpy(track_list[track_idx].params[p].keywordfull,
                            fpsarray[i].parray[p].keywordfull,
                            sizeof(track_list[0].params[0].keywordfull)-1);
                        track_list[track_idx].params[p].cnt0 = fpsarray[i].parray[p].cnt0;
                        functionparameter_GetParamValueString(
                            &fpsarray[i].parray[p],
                            track_list[track_idx].params[p].value,
                            VALSTR_LEN);
                    } else {
                        track_list[track_idx].params[p].cnt0 = -1;
                        track_list[track_idx].params[p].keywordfull[0] = '\0';
                    }
                }
            }
            track_list[track_idx].active = 1;

            if(fpsarray[i].md->NBparamMAX != track_list[track_idx].NBparam) {
                track_list[track_idx].params = (PARAM_TRACK *) realloc(track_list[track_idx].params, fpsarray[i].md->NBparamMAX * sizeof(PARAM_TRACK));
                // Initialize new ones if any
                for(int p = track_list[track_idx].NBparam; p < fpsarray[i].md->NBparamMAX; p++) {
                    if (fpsarray[i].parray[p].fpflag & FPFLAG_ACTIVE) {
                        strncpy(track_list[track_idx].params[p].keywordfull,
                            fpsarray[i].parray[p].keywordfull,
                            sizeof(track_list[0].params[0].keywordfull)-1);
                        track_list[track_idx].params[p].cnt0 = fpsarray[i].parray[p].cnt0;
                        functionparameter_GetParamValueString(
                            &fpsarray[i].parray[p],
                            track_list[track_idx].params[p].value,
                            VALSTR_LEN);
                    } else {
                        track_list[track_idx].params[p].keywordfull[0] = '\0';
                        track_list[track_idx].params[p].cnt0 = -1;
                    }
                }
                track_list[track_idx].NBparam = fpsarray[i].md->NBparamMAX;
            }

            for(int p = 0; p < fpsarray[i].md->NBparamMAX; p++) {
                if (!(fpsarray[i].parray[p].fpflag & FPFLAG_ACTIVE)) {
                    continue;
                }
                
                if(fpsarray[i].parray[p].cnt0 != track_list[track_idx].params[p].cnt0) {
                    char current_val[VALSTR_LEN];
                    functionparameter_GetParamValueString(
                        &fpsarray[i].parray[p],
                        current_val,
                        VALSTR_LEN);
                    
                    if (track_list[track_idx].params[p].cnt0 != -1) {
                         print_ut_timestamp();
                         printf(" %s %s : %s -> %s  (cnt: %ld)\n", 
                                track_list[track_idx].name, 
                                fpsarray[i].parray[p].keywordfull,
                                track_list[track_idx].params[p].value,
                                current_val,
                                fpsarray[i].parray[p].cnt0);
                         fflush(stdout);
                    }
                    
                    strncpy(track_list[track_idx].params[p].keywordfull,
                        fpsarray[i].parray[p].keywordfull,
                        sizeof(track_list[0].params[0].keywordfull)-1);
                    strncpy(track_list[track_idx].params[p].value,
                        current_val,
                        VALSTR_LEN - 1);
                    track_list[track_idx].params[p].cnt0 = fpsarray[i].parray[p].cnt0;
                }
            }
        }

        if (first_scan) {
            printf("\n");
            first_scan = 0;
        }

        // Detect deleted FPS
        for(int j = 0; j < track_list_cnt; j++) {
            if(!track_list[j].active &&
               track_list[j].name[0] != '\0')
            {
                print_ut_timestamp();
                printf(" DEL_FPS %s\n",
                       track_list[j].name);
                fflush(stdout);
                if (track_list[j].params != NULL) {
                    free(track_list[j].params);
                    track_list[j].params = NULL;
                }
                track_list[j].NBparam = 0;
                track_list[j].name[0] = '\0';
            }
        }

        usleep((useconds_t)(interval * 1000000));
    }

    // Final cleanup
    regfree(&regex);
    for(int j = 0; j < track_list_cnt; j++) {
        if(track_list[j].params) free(track_list[j].params);
    }
    free(track_list);
    free(keywnode);
    free(fpsarray);

    return 0;
}