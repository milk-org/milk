/**
 * @file milk-fps-set.c
 * @brief Milk fps set module
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <time.h>

#include "fps.h"
#include "fps_globals.h"
#include "fps_scan.h"
#include "fps_connect.h"
#include "fps_disconnect.h"
#include "fps_paramvalue.h"
#include "fps_GetParamIndex.h"

// Helper to check if string starts with prefix
int starts_with(const char *pre, const char *str) {
    size_t lenpre = strlen(pre);
    size_t lenstr = strlen(str);
    return lenstr < lenpre ? 0 : strncmp(pre, str, lenpre) == 0;
}

const char* get_type_name(uint32_t type) {
    if (type & FPTYPE_INT32) return "INT32";
    if (type & FPTYPE_UINT32) return "UINT32";
    if (type & FPTYPE_INT64) return "INT64";
    if (type & FPTYPE_UINT64) return "UINT64";
    if (type & FPTYPE_FLOAT32) return "FLOAT32";
    if (type & FPTYPE_FLOAT64) return "FLOAT64";
    if (type & FPTYPE_STRING) return "STRING";
    if (type & FPTYPE_FILENAME) return "FILENAME";
    if (type & FPTYPE_FITSFILENAME) return "FITSFILENAME";
    if (type & FPTYPE_EXECFILENAME) return "EXECFILENAME";
    if (type & FPTYPE_DIRNAME) return "DIRNAME";
    if (type & FPTYPE_STREAMNAME) return "STREAMNAME";
    if (type & FPTYPE_FPSNAME) return "FPSNAME";
    if (type & FPTYPE_ONOFF) return "ONOFF";
    if (type & FPTYPE_TIMESPEC) return "TIMESPEC";
    if (type & FPTYPE_PID) return "PID";
    return "UNKNOWN";
}

void print_help(const char *progname) {
    printf("Usage: %s [options] <FPSname>.<parameter> <value>\n", progname);
    printf("Set value of an FPS parameter.\n");
    printf("\n");
    printf("Options:\n");
    printf("  -h, --help      Show this help message\n");
}

void do_completion_scan(const char *word) {
    char *dot = strchr(word, '.');
    
    if (dot == NULL) {
        // No dot, scan for FPS names
        FUNCTION_PARAMETER_STRUCT *fpsarray = (FUNCTION_PARAMETER_STRUCT *) calloc(NB_FPS_MAX, sizeof(FUNCTION_PARAMETER_STRUCT));
        KEYWORD_TREE_NODE *keywnode = (KEYWORD_TREE_NODE *) calloc(NB_KEYWNODE_MAX, sizeof(KEYWORD_TREE_NODE));
        int NBkwn = 0;
        int NBfps = 0;
        long NBpindex = 0;

        functionparameter_scan_fps(0, "_ALL", fpsarray, keywnode, &NBkwn, &NBfps, &NBpindex, 0);

        for(int i=0; i<NBfps; i++) {
            if(starts_with(word, fpsarray[i].md->name)) {
                printf("%s.\n", fpsarray[i].md->name);
            }
            function_parameter_struct_disconnect(&fpsarray[i]);
        }
        free(fpsarray);
        free(keywnode);
    } else {
        // Dot found, extract FPS name
        char fpsname[128];
        int len = dot - word;
        if(len >= 128) len = 127;
        strncpy(fpsname, word, len);
        fpsname[len] = '\0';
        
        char *param_prefix = dot + 1;

        FUNCTION_PARAMETER_STRUCT fps;
        long NBparam = function_parameter_struct_connect(fpsname,
            &fps,
            FPSCONNECT_SIMPLE);
        if(NBparam != -1) {
            fps.NBparam = NBparam;
            for(int i=0; i<fps.NBparam; i++) {
                if (!(fps.parray[i].fpflag & FPFLAG_ACTIVE)) continue;

                const char *kw = fps.parray[i].keywordfull;
                // kw is typically "FPSname.keyword"
                
                // We want to match param_prefix against the part AFTER FPSname.
                
                // Verify kw starts with fpsname
                if (strncmp(kw, fpsname, strlen(fpsname)) == 0 && kw[strlen(fpsname)] == '.') {
                     const char *suffix = kw + strlen(fpsname) + 1;
                     if (starts_with(param_prefix, suffix)) {
                         printf("%s\n", kw);
                     }
                } else {
                    // Fallback if kw format is unexpected (e.g. just .keyword)
                    // If kw starts with '.', treat it as suffix directly
                    if (kw[0] == '.') {
                        if (starts_with(param_prefix, kw+1)) {
                             printf("%s%s\n", fpsname, kw);
                        }
                    }
                }
            }
            function_parameter_struct_disconnect(&fps);
        }
    }
}

int main(int argc, char *argv[])
{
    // Check for bash completion mode
    if (getenv("COMP_LINE") != NULL) {
        if (argc >= 3) {
            do_completion_scan(argv[2]);
            return 0;
        }
    }

    int opt;
    int option_index = 0;
    static struct option long_options[] = {
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'h':
                print_help(argv[0]);
                return 0;
            default:
                print_help(argv[0]);
                return 1;
        }
    }

    if (optind + 2 > argc) {
        if (optind + 1 == argc) {
             fprintf(stderr, "Error: Missing value.\n");
        } else {
             fprintf(stderr, "Error: Missing arguments.\n");
        }
        print_help(argv[0]);
        return 1;
    }

    char *fullkey = argv[optind];
    char *value_str = argv[optind+1];

    char *dot = strchr(fullkey, '.');
    if(dot == NULL) {
        fprintf(stderr, "Error: Invalid format '%s'. Expected <FPSname>.<parameter>\n", fullkey);
        return 1;
    }

    char fpsname[128];
    int len = dot - fullkey;
    if(len >= 128) len = 127;
    strncpy(fpsname, fullkey, len);
    fpsname[len] = '\0';
    
    char *keyword = dot; // Includes the dot

    FUNCTION_PARAMETER_STRUCT fps;
    long NBparam = function_parameter_struct_connect(fpsname,
        &fps,
        FPSCONNECT_SIMPLE);
    if(NBparam == -1) {
        fprintf(stderr, "Error: Could not connect to FPS '%s'\n", fpsname);
        return 1;
    }
    fps.NBparam = NBparam;

    long pindex = functionparameter_GetParamIndex(&fps, keyword);
    if(pindex == -1) {
        pindex = functionparameter_GetParamIndex(&fps, dot+1);
        if(pindex == -1) {
             fprintf(stderr, "Error: Parameter '%s' not found in FPS '%s'\n", keyword, fpsname);
             function_parameter_struct_disconnect(&fps);
             return 1;
        }
    }

    int type = fps.parray[pindex].type;
    int vOK = 1;
    char *endptr;
    long lval;
    double fval;

    switch(type) {
        case FPTYPE_INT32:
            lval = strtol(value_str, &endptr, 10);
            if (*endptr != '\0') vOK = 0;
            else fps.parray[pindex].val.i32[0] = (int32_t)lval;
            break;
        case FPTYPE_UINT32:
            lval = strtol(value_str, &endptr, 10);
            if (*endptr != '\0' || lval < 0) vOK = 0;
            else fps.parray[pindex].val.ui32[0] = (uint32_t)lval;
            break;
        case FPTYPE_INT64:
            lval = strtol(value_str, &endptr, 10);
            if (*endptr != '\0') vOK = 0;
            else fps.parray[pindex].val.i64[0] = (int64_t)lval;
            break;
        case FPTYPE_UINT64:
            lval = strtol(value_str, &endptr, 10);
            if (*endptr != '\0' || lval < 0) vOK = 0;
            else fps.parray[pindex].val.ui64[0] = (uint64_t)lval;
            break;
        case FPTYPE_FLOAT32:
            fval = strtod(value_str, &endptr);
            if (*endptr != '\0') vOK = 0;
            else fps.parray[pindex].val.f32[0] = (float)fval;
            break;
        case FPTYPE_FLOAT64:
            fval = strtod(value_str, &endptr);
            if (*endptr != '\0') vOK = 0;
            else fps.parray[pindex].val.f64[0] = fval;
            break;
        case FPTYPE_STRING:
        case FPTYPE_FILENAME:
        case FPTYPE_FITSFILENAME:
        case FPTYPE_EXECFILENAME:
        case FPTYPE_DIRNAME:
        case FPTYPE_STREAMNAME:
        case FPTYPE_FPSNAME:
            strncpy(fps.parray[pindex].val.string[0],
                value_str,
                FUNCTION_PARAMETER_STRMAXLEN-1);
            break;
        case FPTYPE_ONOFF:
            if(strcasecmp(value_str, "ON") == 0 || strcmp(value_str, "1") == 0) {
                fps.parray[pindex].fpflag |= FPFLAG_ONOFF;
                fps.parray[pindex].val.i64[0] = 1;
            } else if(strcasecmp(value_str, "OFF") == 0 || strcmp(value_str, "0") == 0) {
                fps.parray[pindex].fpflag &= ~FPFLAG_ONOFF;
                fps.parray[pindex].val.i64[0] = 0;
            } else {
                vOK = 0;
            }
            break;
        case FPTYPE_TIMESPEC:
            fval = strtod(value_str, &endptr);
            if (*endptr != '\0') vOK = 0;
            else {
                struct timespec ts;
                ts.tv_sec = (time_t)fval;
                ts.tv_nsec = (long)((fval - (double)ts.tv_sec) * 1000000000.0);
                if (ts.tv_nsec < 0) ts.tv_nsec = 0;
                if (ts.tv_nsec >= 1000000000) ts.tv_nsec = 999999999;
                
                fps.parray[pindex].val.ts[0] = ts;
            }
            break;
        default:
            fprintf(stderr, "Error: Unsupported parameter type '%s' (0x%x) for CLI setting.\n", get_type_name(type), type);
            vOK = 0;
    }

    if (!vOK) {
        fprintf(stderr, "Error: Failed to set parameter '%s'. Type mismatch or invalid format.\n", fullkey);
        fprintf(stderr, "       Parameter Type: %s\n", get_type_name(type));
        fprintf(stderr, "       Input Value:    '%s'\n", value_str);
    } else {
        fps.parray[pindex].cnt0++;
        fps.parray[pindex].value_cnt++;

        fps.md->signal |= FUNCTION_PARAMETER_STRUCT_SIGNAL_UPDATE;
        
        functionparameter_outlog("SETVAL", "%s %s", fps.parray[pindex].keywordfull, value_str);
        
        printf("Parameter '%s' set to '%s'\n", fullkey, value_str);
    }

    function_parameter_struct_disconnect(&fps);
    return vOK ? 0 : 1;
}