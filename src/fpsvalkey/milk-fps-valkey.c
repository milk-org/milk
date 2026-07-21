// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    milk-fps-valkey.c
 * @brief   FPS parameter sync to Valkey database
 *
 * Standalone executable that polls local FPS shared
 * memory for parameter changes, pushes them to a
 * Valkey key-value store, and receives remote
 * changes via Pub/Sub for bidirectional sync.
 *
 * Usage:
 *   milk-fps-valkey [options] [regex_pattern]
 *
 * Based on milk-fps-track.c but adds Valkey
 * integration for cross-machine FPS sharing.
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
#include "fps_GetTypeString.h"

#include "fps_valkey.h"

#define VALSTR_LEN 256

typedef struct
{
    char keywordfull[FUNCTION_PARAMETER_KEYWORD_STRMAXLEN * FUNCTION_PARAMETER_KEYWORD_MAXLEVEL];
    char value[VALSTR_LEN];
    char typestr[32];
    long cnt0;
} PARAM_TRACK;

typedef struct
{
    char         name[STRINGMAXLEN_FPS_NAME];
    int          active;
    long         NBparam;
    PARAM_TRACK *params;
} FPS_TRACK;

static FPS_TRACK   *track_list     = NULL;
static int          track_list_cnt = 0;
static volatile int keep_running   = 1;

static void sigint_handler(int sig)
{
    (void) sig;
    keep_running = 0;
}

static void print_ut_timestamp(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm *ut = gmtime(&ts.tv_sec);
    printf("%04d%02d%02dT%02d:%02d:%02d.%03ld", ut->tm_year + 1900, ut->tm_mon + 1, ut->tm_mday,
           ut->tm_hour, ut->tm_min, ut->tm_sec, ts.tv_nsec / 1000000);
}

/**
 * @brief Get type name string for FPS parameter
 *
 * @param type  FPTYPE_* bitmask
 * @return static string
 */
static const char *get_type_name(uint32_t type)
{
    if (type & FPTYPE_INT32)
    {
        return "INT32";
    }
    if (type & FPTYPE_UINT32)
    {
        return "UINT32";
    }
    if (type & FPTYPE_INT64)
    {
        return "INT64";
    }
    if (type & FPTYPE_UINT64)
    {
        return "UINT64";
    }
    if (type & FPTYPE_FLOAT32)
    {
        return "FLOAT32";
    }
    if (type & FPTYPE_FLOAT64)
    {
        return "FLOAT64";
    }
    if (type & FPTYPE_ONOFF)
    {
        return "ONOFF";
    }
    if (type & FPTYPE_TIMESPEC)
    {
        return "TIMESPEC";
    }
    if (type & FPTYPE_PID)
    {
        return "PID";
    }
    if (type & FPTYPE_STRING)
    {
        return "STRING";
    }
    if (type & FPTYPE_FILENAME)
    {
        return "FILENAME";
    }
    if (type & FPTYPE_FITSFILENAME)
    {
        return "FITSFILENAME";
    }
    if (type & FPTYPE_EXECFILENAME)
    {
        return "EXECFILENAME";
    }
    if (type & FPTYPE_DIRNAME)
    {
        return "DIRNAME";
    }
    if (type & FPTYPE_STREAMNAME)
    {
        return "STREAMNAME";
    }
    if (type & FPTYPE_FPSNAME)
    {
        return "FPSNAME";
    }
    if (type & FPTYPE_PROCESS)
    {
        return "PROCESS";
    }
    if (type & FPTYPE_STRING_NOT_STREAM)
    {
        return "STRING_NOT_STREAM";
    }
    return "UNKNOWN";
}

static void print_help(const char *progname)
{
    printf("Usage: %s [options] [regex]\n", progname);
    printf("\n");
    printf("Sync FPS parameters to/from a Valkey "
           "database.\n");
    printf("\n");
    printf("Options:\n");
    printf("  -i, --interval SEC  "
           "Polling interval (default 0.1)\n");
    printf("  -V, --valkey-host H "
           "Valkey server (default 127.0.0.1)\n");
    printf("  -P, --valkey-port P "
           "Valkey port (default 6379)\n");
    printf("  -h, --help          "
           "Show this help message\n");
    printf("\n");
    printf("Arguments:\n");
    printf("  regex               "
           "FPS name filter (default \".*\")\n");
}

/**
 * @brief Get simple value string for a parameter
 *
 * Produces a type-appropriate string representation
 * without extra stream metadata.
 */
static void param_value_str(FPS_PARAM *fp, char *buf, int buflen)
{
    switch (fp->type)
    {
    case FPTYPE_INT32:
        snprintf(buf, buflen, "%d", fp->val.i32[0]);
        break;
    case FPTYPE_UINT32:
        snprintf(buf, buflen, "%u", fp->val.ui32[0]);
        break;
    case FPTYPE_INT64:
        snprintf(buf, buflen, "%ld", fp->val.i64[0]);
        break;
    case FPTYPE_UINT64:
        snprintf(buf, buflen, "%lu", fp->val.ui64[0]);
        break;
    case FPTYPE_FLOAT32:
        snprintf(buf, buflen, "%.10g", fp->val.f32[0]);
        break;
    case FPTYPE_FLOAT64:
        snprintf(buf, buflen, "%.17g", fp->val.f64[0]);
        break;
    case FPTYPE_PID:
        snprintf(buf, buflen, "%ld", (long) fp->val.pid[0]);
        break;
    case FPTYPE_TIMESPEC:
        snprintf(buf, buflen, "%ld.%09ld", fp->val.ts[0].tv_sec, fp->val.ts[0].tv_nsec);
        break;
    case FPTYPE_ONOFF:
        snprintf(buf, buflen, "%s", fp->val.i64[0] ? "ON" : "OFF");
        break;
    case FPTYPE_STRING:
    case FPTYPE_FILENAME:
    case FPTYPE_FITSFILENAME:
    case FPTYPE_EXECFILENAME:
    case FPTYPE_DIRNAME:
    case FPTYPE_STREAMNAME:
    case FPTYPE_FPSNAME:
    case FPTYPE_PROCESS:
    case FPTYPE_STRING_NOT_STREAM:
        snprintf(buf, buflen, "%s", fp->val.string[0]);
        break;
    default:
        snprintf(buf, buflen, "?");
        break;
    }
}


int main(int argc, char *argv[])
{
    double      interval      = 0.1;
    char       *regex_pattern = ".*";
    const char *valkey_host   = "127.0.0.1";
    int         valkey_port   = 6379;
    int         opt;

    static struct option long_options[] = { { "interval", required_argument, 0, 'i' },
                                            { "valkey-host", required_argument, 0, 'V' },
                                            { "valkey-port", required_argument, 0, 'P' },
                                            { "help", no_argument, 0, 'h' },
                                            { 0, 0, 0, 0 } };

    while ((opt = getopt_long(argc, argv, "i:V:P:h", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'i':
            interval = atof(optarg);
            break;
        case 'V':
            valkey_host = optarg;
            break;
        case 'P':
            valkey_port = atoi(optarg);
            break;
        case 'h':
            print_help(argv[0]);
            return 0;
        case '?':
        default:
            printf("\n\033[1;31mERROR\033[0m: Invalid option.\n\n");
            print_help(argv[0]);
            return 1;
        }
    }

    if (optind < argc)
    {
        regex_pattern = argv[optind];
    }

    /* Compile regex */
    regex_t regex;
    if (regcomp(&regex, regex_pattern, REG_EXTENDED | REG_NOSUB) != 0)
    {
        fprintf(stderr, "Error: bad regex pattern\n");
        return 1;
    }

    {
        struct sigaction sa;
        sa.sa_handler = sigint_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        sigaction(SIGINT, &sa, NULL);
        sigaction(SIGTERM, &sa, NULL);
    }

    /* Connect to Valkey */
    FPS_VALKEY_CTX vctx;
    if (fps_valkey_connect(&vctx, valkey_host, valkey_port) != 0)
    {
        fprintf(stderr,
                "Error: cannot connect to Valkey "
                "at %s:%d\n",
                valkey_host, valkey_port);
        regfree(&regex);
        return 1;
    }

    /* Start subscriber thread */
    if (fps_valkey_sub_start(&vctx) != 0)
    {
        fprintf(stderr, "Error: subscriber start failed\n");
        fps_valkey_disconnect(&vctx);
        regfree(&regex);
        return 1;
    }

    /* Allocate FPS scan arrays */
    fpsarray = (FPS *) calloc(NB_FPS_MAX, sizeof(FPS));
    for (int i = 0; i < NB_FPS_MAX; i++)
    {
        fpsarray[i].SMfd = -1;
    }

    KEYWORD_TREE_NODE *keywnode =
        (KEYWORD_TREE_NODE *) calloc(NB_KEYWNODE_MAX, sizeof(KEYWORD_TREE_NODE));

    track_list = (FPS_TRACK *) calloc(NB_FPS_MAX, sizeof(FPS_TRACK));

    printf("Syncing FPS matching: %s\n", regex_pattern);
    printf("Valkey: %s:%d\n", valkey_host, valkey_port);
    printf("Interval: %.3f s\n", interval);
    printf("Tracked instances:\n");

    int  first_scan = 1;
    long loop_cnt   = 0;

    while (keep_running)
    {
        int  NBkwn    = 0;
        int  NBfps    = 0;
        long NBpindex = 0;

        /* Scan FPS */
        functionparameter_scan_fps(0, "_ALL", fpsarray, keywnode, &NBkwn, &NBfps, &NBpindex, 0);

        /* Mark all tracked as inactive */
        for (int i = 0; i < track_list_cnt; i++)
        {
            track_list[i].active = 0;
        }

        for (int i = 0; i < NBfps; i++)
        {
            /* Check regex */
            if (regexec(&regex, fpsarray[i].md->name, 0, NULL, 0) != 0)
            {
                continue;
            }

            /* Find existing tracker */
            int tidx = -1;
            for (int j = 0; j < track_list_cnt; j++)
            {
                if (strcmp(track_list[j].name, fpsarray[i].md->name) == 0)
                {
                    tidx = j;
                    break;
                }
            }

            if (tidx == -1)
            {
                /* New FPS discovered */
                tidx = track_list_cnt++;
                strncpy(track_list[tidx].name, fpsarray[i].md->name, STRINGMAXLEN_FPS_NAME - 1);
                track_list[tidx].NBparam = fpsarray[i].md->NBparamMAX;
                track_list[tidx].params =
                    (PARAM_TRACK *) calloc(fpsarray[i].md->NBparamMAX, sizeof(PARAM_TRACK));

                if (first_scan)
                {
                    printf("  - %s\n", track_list[tidx].name);
                }
                else
                {
                    print_ut_timestamp();
                    printf(" NEW_FPS %s\n", track_list[tidx].name);
                }

                /* Register in Valkey */
                fps_valkey_register_fps(&vctx, track_list[tidx].name);

                /* Init + push all active params */
                for (int p = 0; p < fpsarray[i].md->NBparamMAX; p++)
                {
                    if (fpsarray[i].parray[p].fpflag & FPFLAG_ACTIVE)
                    {
                        strncpy(track_list[tidx].params[p].keywordfull,
                                fpsarray[i].parray[p].keywordfull,
                                sizeof(track_list[0].params[0].keywordfull) - 1);

                        track_list[tidx].params[p].cnt0 = fpsarray[i].parray[p].cnt0;

                        const char *tn = get_type_name(fpsarray[i].parray[p].type);
                        strncpy(track_list[tidx].params[p].typestr, tn, 31);

                        char vbuf[VALSTR_LEN];
                        param_value_str(&fpsarray[i].parray[p], vbuf, VALSTR_LEN);
                        strncpy(track_list[tidx].params[p].value, vbuf, VALSTR_LEN - 1);

                        fps_valkey_push_param(&vctx, track_list[tidx].name,
                                              fpsarray[i].parray[p].keywordfull, vbuf, tn,
                                              fpsarray[i].parray[p].cnt0);
                    }
                    else
                    {
                        track_list[tidx].params[p].cnt0           = -1;
                        track_list[tidx].params[p].keywordfull[0] = '\0';
                    }
                }

                /* Push metadata */
                fps_valkey_push_metadata(&vctx, track_list[tidx].name, fpsarray[i].md);
            }

            track_list[tidx].active = 1;

            /* Handle param array resize */
            if (fpsarray[i].md->NBparamMAX != track_list[tidx].NBparam)
            {
                PARAM_TRACK *tmp = (PARAM_TRACK *) realloc(
                    track_list[tidx].params, fpsarray[i].md->NBparamMAX * sizeof(PARAM_TRACK));
                if (tmp == NULL)
                {
                    fprintf(stderr,
                            "realloc failed for %s"
                            " params\n",
                            track_list[tidx].name);
                    continue;
                }
                track_list[tidx].params = tmp;

                for (int p = track_list[tidx].NBparam; p < fpsarray[i].md->NBparamMAX; p++)
                {
                    if (fpsarray[i].parray[p].fpflag & FPFLAG_ACTIVE)
                    {
                        strncpy(track_list[tidx].params[p].keywordfull,
                                fpsarray[i].parray[p].keywordfull,
                                sizeof(track_list[0].params[0].keywordfull) - 1);
                        track_list[tidx].params[p].cnt0 = fpsarray[i].parray[p].cnt0;

                        const char *tn = get_type_name(fpsarray[i].parray[p].type);
                        strncpy(track_list[tidx].params[p].typestr, tn, 31);

                        char vbuf[VALSTR_LEN];
                        param_value_str(&fpsarray[i].parray[p], vbuf, VALSTR_LEN);
                        strncpy(track_list[tidx].params[p].value, vbuf, VALSTR_LEN - 1);
                    }
                    else
                    {
                        track_list[tidx].params[p].keywordfull[0] = '\0';
                        track_list[tidx].params[p].cnt0           = -1;
                    }
                }
                track_list[tidx].NBparam = fpsarray[i].md->NBparamMAX;
            }

            /* Detect parameter changes */
            for (int p = 0; p < fpsarray[i].md->NBparamMAX; p++)
            {
                if (!(fpsarray[i].parray[p].fpflag & FPFLAG_ACTIVE))
                {
                    continue;
                }

                if (fpsarray[i].parray[p].cnt0 != track_list[tidx].params[p].cnt0)
                {
                    char cur_val[VALSTR_LEN];
                    param_value_str(&fpsarray[i].parray[p], cur_val, VALSTR_LEN);

                    const char *tn = get_type_name(fpsarray[i].parray[p].type);

                    if (track_list[tidx].params[p].cnt0 != -1)
                    {
                        print_ut_timestamp();
                        printf(" PUSH %s %s : "
                               "%s -> %s\n",
                               track_list[tidx].name, fpsarray[i].parray[p].keywordfull,
                               track_list[tidx].params[p].value, cur_val);
                        fflush(stdout);
                    }

                    /* Push to Valkey */
                    fps_valkey_push_param(&vctx, track_list[tidx].name,
                                          fpsarray[i].parray[p].keywordfull, cur_val, tn,
                                          fpsarray[i].parray[p].cnt0);

                    /* Update tracker */
                    strncpy(track_list[tidx].params[p].keywordfull,
                            fpsarray[i].parray[p].keywordfull,
                            sizeof(track_list[0].params[0].keywordfull) - 1);
                    strncpy(track_list[tidx].params[p].value, cur_val, VALSTR_LEN - 1);
                    strncpy(track_list[tidx].params[p].typestr, tn, 31);
                    track_list[tidx].params[p].cnt0 = fpsarray[i].parray[p].cnt0;
                }
            }
        }

        if (first_scan)
        {
            printf("\n");
            first_scan = 0;
        }

        /* Detect deleted FPS */
        for (int j = 0; j < track_list_cnt; j++)
        {
            if (!track_list[j].active && track_list[j].name[0] != '\0')
            {
                print_ut_timestamp();
                printf(" DEL_FPS %s\n", track_list[j].name);
                fflush(stdout);

                fps_valkey_unregister_fps(&vctx, track_list[j].name);

                if (track_list[j].params != NULL)
                {
                    free(track_list[j].params);
                    track_list[j].params = NULL;
                }
                track_list[j].NBparam = 0;
                track_list[j].name[0] = '\0';
            }
        }

        /* Periodic metadata sync (every 10 iter) */
        if (loop_cnt % 10 == 0)
        {
            for (int i = 0; i < NBfps; i++)
            {
                if (regexec(&regex, fpsarray[i].md->name, 0, NULL, 0) == 0)
                {
                    fps_valkey_push_metadata(&vctx, fpsarray[i].md->name, fpsarray[i].md);
                }
            }
        }

        loop_cnt++;
        usleep((useconds_t) (interval * 1000000));
    }

    /* Cleanup */
    printf("\nShutting down...\n");
    fps_valkey_disconnect(&vctx);

    regfree(&regex);
    for (int j = 0; j < track_list_cnt; j++)
    {
        if (track_list[j].params)
        {
            free(track_list[j].params);
        }
    }
    free(track_list);
    free(keywnode);
    free(fpsarray);

    return 0;
}
