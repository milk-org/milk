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

int function_parameter_print_info(
    FUNCTION_PARAMETER_STRUCT *fps,
    int verbose,
    int show_info
)
{
    if (fps == NULL || fps->md == NULL || fps->parray == NULL) {
        return -1;
    }

    printf("FPS Name        : %s\n", fps->md->name);
    printf("Description     : %s\n", fps->md->description);
    
    char * exec_basename = strrchr(fps->md->execfullpath, '/');
    if (exec_basename) {
        exec_basename++;
    } else {
        exec_basename = fps->md->execfullpath;
    }
    printf("Exec            : %s\n", exec_basename);

    if (verbose) {
        printf("Work Directory  : %s\n", fps->md->workdir);
        printf("Source File     : %s:%d\n", fps->md->sourcefname, fps->md->sourceline);
        printf("Keywords        : %s\n", fps->md->keywordarray);
    }
    printf("Parameters      : %ld / %ld active\n", fps->NBparamActive, fps->md->NBparamMAX);
    printf("\n");

    // Calculate dynamic widths
    int kw_width = 7; // "Keyword"
    int val_width = 5; // "Value"
    for (int pindex = 0; pindex < fps->md->NBparamMAX; pindex++) {
        if (fps->parray[pindex].fpflag & FPFLAG_USED) {
            int kl = strlen(fps->parray[pindex].keywordfull);
            if (kl > kw_width) kw_width = kl;

            char valstring[200];
            if (fps->parray[pindex].type == FPTYPE_STREAMNAME) {
                snprintf(valstring, 200, "%s", fps->parray[pindex].val.string[0]);
            } else {
                functionparameter_GetParamValueString(&fps->parray[pindex], valstring, 200);
            }
            int vl = strlen(valstring);
            if (vl > val_width) val_width = vl;
        }
    }
    if (kw_width > 60) kw_width = 60;
    if (val_width > 60) val_width = 60;

    printf("%4s %-*s %12s %*s %s\n", "CLI", kw_width, "Keyword", "Type", val_width, "Value", "Description");
    for (int i=0; i<4 + 1 + kw_width + 1 + 12 + 1 + val_width + 1 + 30; i++) printf("-");
    printf("\n");

    for (int pindex = 0; pindex < fps->md->NBparamMAX; pindex++) {
        if (fps->parray[pindex].fpflag & FPFLAG_USED) {
            char valstring[200];
            if (fps->parray[pindex].type == FPTYPE_STREAMNAME) {
                snprintf(valstring, 200, "%s", fps->parray[pindex].val.string[0]);
            } else {
                functionparameter_GetParamValueString(&fps->parray[pindex], valstring, 200);
            }
            
            const char* type_str = "UNKNOWN";
            switch(fps->parray[pindex].type) {
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
                case FPTYPE_FPSNAME: type_str = "FPSNAME"; break;
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

            printf("%4s %s%-*s%s %12s %*s %s\n",
                   cli_idx_str,
                   color_start,
                   kw_width,
                   fps->parray[pindex].keywordfull,
                   color_end,
                   type_str,
                   val_width,
                   valstring,
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
