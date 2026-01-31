#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>

#include "fps.h"
#include "fps_globals.h"
#include "fps_scan.h"
#include "fps_printparameter_valuestring.h"

void print_help(const char *progname) {
    printf("Usage: %s [options] <fpsname>\n", progname);
    printf("Print content of a Function Parameter Structure (FPS).\n");
    printf("\n");
    printf("Options:\n");
    printf("  -v, --verbose   Verbose mode\n");
    printf("  -i, --info      Show detailed stream information on separate line\n");
    printf("  -h, --help      Show this help message\n");
}

int main(int argc, char *argv[])
{
    int verbose = 0;
    int show_info = 0;
    int opt;

    static struct option long_options[] = {
        {"verbose", no_argument,       0, 'v'},
        {"info",    no_argument,       0, 'i'},
        {"help",    no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "vih", long_options, NULL)) != -1) {
        switch (opt) {
            case 'v':
                verbose = 1;
                break;
            case 'i':
                show_info = 1;
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
        fprintf(stderr, "Error: missing FPS name.\n");
        print_help(argv[0]);
        return 1;
    }

    const char *fpsname = argv[optind];

    FUNCTION_PARAMETER_STRUCT fps;
    fps.SMfd = -1;

    if (function_parameter_struct_connect(fpsname, &fps, 0) == -1) {
        fprintf(stderr, "Error: cannot connect to FPS '%s'.\n", fpsname);
        return 1;
    }

    printf("FPS Name        : %s\n", fps.md->name);
    printf("Description     : %s\n", fps.md->description);
    
    char * exec_basename = strrchr(fps.md->execfullpath, '/');
    if (exec_basename) {
        exec_basename++;
    } else {
        exec_basename = fps.md->execfullpath;
    }
    printf("Exec            : %s\n", exec_basename);

    if (verbose) {
        printf("Work Directory  : %s\n", fps.md->workdir);
        printf("Source File     : %s:%d\n", fps.md->sourcefname, fps.md->sourceline);
        printf("Keywords        : %s\n", fps.md->keywordarray);
    }
    printf("Parameters      : %ld / %ld active\n", fps.NBparamActive, fps.md->NBparamMAX);
    printf("\n");

    // Calculate dynamic widths
    int kw_width = 7; // "Keyword"
    int val_width = 5; // "Value"
    for (int pindex = 0; pindex < fps.md->NBparamMAX; pindex++) {
        if (fps.parray[pindex].fpflag & FPFLAG_USED) {
            int kl = strlen(fps.parray[pindex].keywordfull);
            if (kl > kw_width) kw_width = kl;

            char valstring[200];
            if (fps.parray[pindex].type == FPTYPE_STREAMNAME) {
                snprintf(valstring, 200, "%s", fps.parray[pindex].val.string[0]);
            } else {
                functionparameter_GetParamValueString(&fps.parray[pindex], valstring, 200);
            }
            int vl = strlen(valstring);
            if (vl > val_width) val_width = vl;
        }
    }
    if (kw_width > 60) kw_width = 60;
    if (val_width > 60) val_width = 60;

    printf("%-*s %12s %*s %s\n", kw_width, "Keyword", "Type", val_width, "Value", "Description");
    for (int i=0; i<kw_width + 12 + val_width + 30; i++) printf("-");
    printf("\n");

    for (int pindex = 0; pindex < fps.md->NBparamMAX; pindex++) {
        if (fps.parray[pindex].fpflag & FPFLAG_USED) {
            char valstring[200];
            if (fps.parray[pindex].type == FPTYPE_STREAMNAME) {
                snprintf(valstring, 200, "%s", fps.parray[pindex].val.string[0]);
            } else {
                functionparameter_GetParamValueString(&fps.parray[pindex], valstring, 200);
            }
            
            const char* type_str = "UNKNOWN";
            switch(fps.parray[pindex].type) {
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

            printf("%-*s %12s %*s %s\n", 
                   kw_width,
                   fps.parray[pindex].keywordfull, 
                   type_str, 
                   val_width,
                   valstring, 
                   fps.parray[pindex].description);

            if (show_info && fps.parray[pindex].type == FPTYPE_STREAMNAME) {
                IMAGE tmpimg;
                if (ImageStreamIO_openIm(&tmpimg, fps.parray[pindex].val.string[0]) == IMAGESTREAMIO_SUCCESS) {
                    const char* dtype_str = ImageStreamIO_typename(tmpimg.md->datatype);
                    char size_str[64];
                    if (tmpimg.md->naxis == 1) snprintf(size_str, 64, "%u", tmpimg.md->size[0]);
                    else if (tmpimg.md->naxis == 2) snprintf(size_str, 64, "%ux%u", tmpimg.md->size[0], tmpimg.md->size[1]);
                    else snprintf(size_str, 64, "%ux%ux%u", tmpimg.md->size[0], tmpimg.md->size[1], tmpimg.md->size[2]);

                    printf("%*s -> [%s %s cnt=%lu]\n", kw_width, "", dtype_str, size_str, tmpimg.md->cnt0);
                    ImageStreamIO_closeIm(&tmpimg);
                } else {
                    printf("%*s -> [NOTFOUND]\n", kw_width, "");
                }
            }
        }
    }

    function_parameter_struct_disconnect(&fps);

    return 0;
}
