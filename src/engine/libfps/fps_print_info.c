/**
 * @file    fps_print_info.c
 * @brief   Print content of a Function Parameter Structure (FPS)
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "fps.h"
#include "fps_print_info.h"
#include "fps_printparameter_valuestring.h"

#define C_TITLE  "\033[1;36m"
#define C_HDR    "\033[1;35m"
#define C_CMD    "\033[1;32m"
#define C_NOTE   "\033[1;33m"
#define C_BOLD   "\033[1m"

int function_parameter_print_info(
    FUNCTION_PARAMETER_STRUCT *fps,
    int verbose,
    int show_info
)
{
    if (fps == NULL || fps->md == NULL || fps->parray == NULL) {
        return -1;
    }

    printf(C_TITLE "========================================================\n" COLORRESET);
    printf(C_TITLE " %-20s : " C_HDR "%s" COLORRESET "\n", "FPS Name", fps->md->name);
    printf(C_TITLE " %-20s : " C_CMD "%s" COLORRESET "\n", "Command Key", fps->md->callprogname);
    printf(C_TITLE " %-20s : " COLORRESET "%s\n", "Description", fps->md->description);
    
    char * exec_basename = strrchr(fps->md->execfullpath, '/');
    if (exec_basename) {
        exec_basename++;
    } else {
        exec_basename = fps->md->execfullpath;
    }
    printf(C_TITLE " %-20s : " C_CMD "./%s" COLORRESET "\n", "Executable", exec_basename);

    if (verbose) {
        printf(C_TITLE " %-20s : " COLORRESET "%s\n", "Work Directory", fps->md->workdir);
        printf(C_TITLE " %-20s : " COLORRESET "%s:%d\n", "Source File", fps->md->sourcefname, fps->md->sourceline);
        printf(C_TITLE " %-20s : " COLORRESET "%s\n", "Keywords", fps->md->keywordarray);
    }
    printf(C_TITLE " %-20s : " COLORRESET "%ld / %ld active\n", "Parameters", fps->NBparamActive, fps->md->NBparamMAX);
    if (functionparameter_GetParamIndex(fps, ".procinfo.enabled") != -1) {
        printf(C_TITLE " %-20s : " C_CMD "ENABLED" COLORRESET "\n", "Processinfo API");
    } else {
        printf(C_TITLE " %-20s : " COLORRESET "DISABLED\n", "Processinfo API");
    }
    printf(C_TITLE "========================================================\n" COLORRESET);
    printf("\n");

    // Calculate dynamic widths
    int kw_width = 7; // "Keyword"
    int val_width = 5; // "Value"
    for (int pindex = 0; pindex < fps->md->NBparamMAX; pindex++) {
        if (fps->parray[pindex].fpflag & FPFLAG_USED) {
            const char *display_keyword = fps->parray[pindex].keywordfull;
            int prefix_len = strlen(fps->md->name);
            if (strncmp(display_keyword, fps->md->name, prefix_len) == 0 && display_keyword[prefix_len] == '.') {
                display_keyword += prefix_len + 1;
            }
            int kl = strlen(display_keyword);
            if (kl > kw_width) kw_width = kl;

            char valstring[200];
            if (fps->parray[pindex].type == FPTYPE_STREAMNAME) {
                snprintf(valstring, 200, "%s", fps->parray[pindex].val.string[0]);
            } else {
                functionparameter_GetParamValueString(&fps->parray[pindex],
                    valstring,
                    200);
            }
            int vl = strlen(valstring);
            if (vl > val_width) val_width = vl;
        }
    }
    if (kw_width > 60) kw_width = 60;
    if (val_width > 60) val_width = 60;

    printf("%4s %-*s %12s %*s %8s %s\n", "CLI", kw_width, "Keyword", "Type", val_width, "Value", "Count", "Description");
    for (int i=0; i<4 + 1 + kw_width + 1 + 12 + 1 + val_width + 1 + 8 + 1 + 30; i++) printf("-");
    printf("\n");

    for (int pindex = 0; pindex < fps->md->NBparamMAX; pindex++) {
        if (fps->parray[pindex].fpflag & FPFLAG_USED) {
            char valstring[200];
            if (fps->parray[pindex].type == FPTYPE_STREAMNAME) {
                snprintf(valstring, 200, "%s", fps->parray[pindex].val.string[0]);
            } else {
                functionparameter_GetParamValueString(&fps->parray[pindex],
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

            const char *color_start = COLORRESET;
            const char *color_end   = COLORRESET;

            if(fps->parray[pindex].cli_index >= 0)
            {
                color_start = COLORPRIMARY;
            }

            char cli_idx_str[8];
            if(fps->parray[pindex].cli_index >= 0)
            {
                snprintf(cli_idx_str, 8, "%d", fps->parray[pindex].cli_index);
            }
            else
            {
                strcpy(cli_idx_str, "---");
            }

            const char *display_keyword = fps->parray[pindex].keywordfull;
            int prefix_len = strlen(fps->md->name);
            if (strncmp(display_keyword, fps->md->name, prefix_len) == 0 && display_keyword[prefix_len] == '.') {
                display_keyword += prefix_len + 1;
            }

            printf("%4s %s%-*s%s %12s %*s %8lu %s\n",
                   cli_idx_str,
                   color_start,
                   kw_width,
                   display_keyword,
                   color_end,
                   type_str,
                   val_width,
                   valstring,
                   fps->parray[pindex].value_cnt,
                   fps->parray[pindex].description);

            if (show_info && fps->parray[pindex].type == FPTYPE_STREAMNAME) {
                IMAGE tmpimg;
                if (ImageStreamIO_openIm(&tmpimg, fps->parray[pindex].val.string[0]) == IMAGESTREAMIO_SUCCESS) {
                    const char* dtype_str = ImageStreamIO_typename(tmpimg.md->datatype);
                    char size_str[64];
                    if (tmpimg.md->naxis == 1) snprintf(size_str, 64, "%u", tmpimg.md->size[0]);
                    else if (tmpimg.md->naxis == 2) snprintf(size_str, 64, "%ux%u", tmpimg.md->size[0], tmpimg.md->size[1]);
                    else snprintf(size_str, 64, "%ux%ux%u", tmpimg.md->size[0], tmpimg.md->size[1], tmpimg.md->size[2]);

                    printf("%*s -> [%s %s cnt=%lu]\n", kw_width + 5, "", dtype_str, size_str, tmpimg.md->cnt0);
                    ImageStreamIO_closeIm(&tmpimg);
                } else {
                    printf("%*s -> [NOTFOUND]\n", kw_width + 5, "");
                }
            }
        }
    }

    return 0;
}
