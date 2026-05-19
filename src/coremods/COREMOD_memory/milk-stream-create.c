/**
 * @file    milk-stream-create.c
 * @brief   Create a shared memory image stream
 *
 * Standalone tool to create ImageStreamIO streams
 * from the command line.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "ImageStreamIO/ImageStreamIO.h"

typedef struct
{
    const char *name;
    uint8_t     code;
} DTYPE_ENTRY;

static const DTYPE_ENTRY dtype_table[] =
{
    {"uint8",   _DATATYPE_UINT8},
    {"u8",      _DATATYPE_UINT8},
    {"int8",    _DATATYPE_INT8},
    {"i8",      _DATATYPE_INT8},
    {"uint16",  _DATATYPE_UINT16},
    {"u16",     _DATATYPE_UINT16},
    {"int16",   _DATATYPE_INT16},
    {"i16",     _DATATYPE_INT16},
    {"uint32",  _DATATYPE_UINT32},
    {"u32",     _DATATYPE_UINT32},
    {"int32",   _DATATYPE_INT32},
    {"i32",     _DATATYPE_INT32},
    {"uint64",  _DATATYPE_UINT64},
    {"u64",     _DATATYPE_UINT64},
    {"int64",   _DATATYPE_INT64},
    {"i64",     _DATATYPE_INT64},
    {"float",   _DATATYPE_FLOAT},
    {"f32",     _DATATYPE_FLOAT},
    {"double",  _DATATYPE_DOUBLE},
    {"f64",     _DATATYPE_DOUBLE},
    {NULL, 0}
};

static uint8_t parse_dtype(const char *s)
{
    /* Try by name */
    for(int i = 0; dtype_table[i].name; i++)
    {
        if(strcasecmp(s, dtype_table[i].name) == 0)
        {
            return dtype_table[i].code;
        }
    }
    /* Try numeric code */
    char *end;
    long v = strtol(s, &end, 10);
    if(*end == '\0' && v >= 1 && v <= 12)
    {
        return (uint8_t) v;
    }
    return 0;
}

/* ANSI color codes */
#define C_HDR   "\033[1;34m"  /* Blue Bold   */
#define C_CMD   "\033[1;32m"  /* Green Bold  */
#define C_OPT   "\033[1;33m"  /* Yellow Bold */
#define C_ARG   "\033[1;35m"  /* Magenta Bold*/
#define C_B     "\033[1m"     /* Bold        */
#define C_DIM   "\033[2m"     /* Dim         */
#define C_RST   "\033[0m"     /* Reset       */

void print_help(const char *progname)
{
    printf("\n" C_HDR "Usage:" C_RST
           " %s " C_OPT "[options]" C_RST
           " " C_ARG "<name>" C_RST
           " " C_ARG "<xsize>" C_RST
           " " C_DIM "[ysize] [zsize]" C_RST
           "\n", progname);
    printf("  Compiled: %s %s\n\n",
           __DATE__, __TIME__);
    printf(C_HDR "Description:" C_RST "\n"
           "  Create a shared memory image"
           " stream.\n\n");
    printf(C_HDR "Arguments:" C_RST "\n");
    printf("  " C_ARG "name" C_RST
           "       Stream name\n");
    printf("  " C_ARG "xsize" C_RST
           "      Width (required)\n");
    printf("  " C_ARG "ysize" C_RST
           "      Height (optional,"
           " makes 2D)\n");
    printf("  " C_ARG "zsize" C_RST
           "      Depth (optional,"
           " makes 3D / circ buffer)\n\n");
    printf(C_HDR "Options:" C_RST "\n");
    printf("  " C_OPT "-t, --type" C_RST
           " " C_ARG "TYPE" C_RST
           "   Data type"
           " (default: float)\n");
    printf("  " C_OPT "-k, --kw" C_RST
           " " C_ARG "N" C_RST
           "       Number of keywords"
           " (default: 10)\n");
    printf("  " C_OPT "-h, --help" C_RST
           "         Show this help\n\n");
    printf(C_HDR "Data Types:" C_RST "\n");
    printf("  " C_CMD "uint8" C_RST "/u8"
           "     " C_CMD "int8" C_RST "/i8"
           "      " C_CMD "uint16" C_RST "/u16"
           "   " C_CMD "int16" C_RST "/i16\n");
    printf("  " C_CMD "uint32" C_RST "/u32"
           "    " C_CMD "int32" C_RST "/i32"
           "     " C_CMD "uint64" C_RST "/u64"
           "   " C_CMD "int64" C_RST "/i64\n");
    printf("  " C_CMD "float" C_RST "/f32"
           "     " C_CMD "double" C_RST
           "/f64\n\n");
    printf(C_HDR "Examples:" C_RST "\n");
    printf("  $ " C_CMD "%s" C_RST " "
           C_ARG "wfs" C_RST " 128 128\n",
           progname);
    printf("  $ " C_CMD "%s" C_RST " "
           C_OPT "-t uint16" C_RST " "
           C_ARG "cam" C_RST " 512 512\n",
           progname);
    printf("  $ " C_CMD "%s" C_RST " "
           C_OPT "-t double" C_RST " "
           C_ARG "signal" C_RST " 1024\n\n",
           progname);
}

int main(
    int argc,
    char *argv[])
{
    /* One-line help — before getopt so it works without any positional args */
    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "-h1") == 0 ||
                strcmp(argv[i], "--help-oneline") == 0)
        {
            printf("create a shared-memory image stream\n");
            return 0;
        }
    }

    uint8_t dtype = _DATATYPE_FLOAT;
    int nbkw = 10;
    int opt;

    static struct option long_options[] =
    {
        {"type", required_argument, 0, 't'},
        {"kw",   required_argument, 0, 'k'},
        {"help", no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    while((opt = getopt_long(argc, argv,
                             "t:k:h",
                             long_options,
                             NULL)) != -1)
    {
        switch(opt)
        {
        case 't':
            dtype = parse_dtype(optarg);
            if(dtype == 0)
            {
                printf("\n\033[1;31mERROR\033[0m unknown type: %s\n\n", optarg);
                print_help(argv[0]);
                return 1;
            }
            break;
        case 'k':
            nbkw = atoi(optarg);
            break;
        case 'h':
            print_help(argv[0]);
            return 0;
        default:
            printf("\n\033[1;31mERROR\033[0m invalid option\n\n");
            print_help(argv[0]);
            return 1;
        }
    }

    int npos = argc - optind;
    if(npos < 2)
    {
        printf("\n\033[1;31mERROR\033[0m need at least name and xsize\n\n");
        print_help(argv[0]);
        return 1;
    }

    const char *name = argv[optind];
    uint32_t sz[3];
    long naxis = npos - 1;
    if(naxis > 3)
    {
        naxis = 3;
    }

    for(int i = 0; i < naxis; i++)
    {
        sz[i] = (uint32_t) atol(argv[optind + 1 + i]);
        if(sz[i] == 0)
        {
            fprintf(stderr,
                    "Error: invalid size %s\n",
                    argv[optind + 1 + i]);
            return 1;
        }
    }

    IMAGE image;
    memset(&image, 0, sizeof(IMAGE));

    int cbsize = 0;
    if(naxis == 3)
    {
        cbsize = sz[2];
    }

    errno_t ret = ImageStreamIO_createIm(
                      &image, name, naxis, sz,
                      dtype, 1, nbkw, cbsize);

    if(ret != IMAGESTREAMIO_SUCCESS)
    {
        fprintf(stderr,
                "Failed to create stream '%s'\n",
                name);
        return 1;
    }

    /* Print confirmation */
    const char *tname =
        ImageStreamIO_typename(dtype);

    printf("Created stream '%s'  ", name);
    printf("type=%s  size=",
           tname ? tname : "?");
    for(int i = 0; i < naxis; i++)
    {
        if(i > 0)
        {
            printf("x");
        }
        printf("%u", sz[i]);
    }
    printf("\n");

    ImageStreamIO_closeIm(&image);
    return 0;
}
