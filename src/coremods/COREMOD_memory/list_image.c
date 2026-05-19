/**
 * @file    list_image.c
 * @brief   Image/variable listing and memory monitor
 *
 * Provides image listing in multiple output formats:
 *  - list_image_ID()         — print to stdout
 *  - list_image_ID_ncurses() — ncurses display
 *  - list_image_ID_ofp()     — formatted to FILE*
 *  - list_image_ID_ofp_json()— JSON output
 *  - list_image_ID_ofp_porcelain() — machine-
 *    parseable
 *
 * Note: the interactive "mmon" TUI has been
 * moved to the streamCTRL CLI tools suite.
 */



#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "CLIcore.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "compute_image_memory.h"
#include "compute_nb_image.h"
#include "image_ID.h"

#define STYPESIZE 10

/* forward decls */
errno_t list_image_ID_ofp(FILE *fo);
errno_t list_image_ID_ofp_simple(FILE *fo);
errno_t list_image_ID_ofp_json(FILE *fo);
errno_t list_image_ID_ofp_porcelain(FILE *fo);
/**
 * @brief Print a summary of all images in the array.
 *
 * Lists name, type, size, and memory usage for
 * each active image.
 */
errno_t list_image_ID();
errno_t list_image_ID_file(
    const char *fname);
errno_t list_variable_ID(
    const char *regexstr);
errno_t list_variable_ID_file(
    const char *fname);


/* ================================================================
 *  CMD 2: listim (0 args)
 * ============================================================= */

static FPS_APP_INFO FPS_app_info_listim =
{
    .fps_name    = "listim",
    .cmdkey      = "listim",
    .description =
    "list images in memory",
    .description_long =
    "List all images currently loaded in the process memory space, showing name, dimensions, data type, and shared memory status."
};

static CLICMDDATA CLIcmddata_listim =
{
    "", "", CLICMD_FIELDS_NOPARAM
};

FPS_CMDSETTINGS_INIT(listim, CLIcmddata_listim, FPS_app_info_listim)

static errno_t __attribute__((unused)) compute_listim()
{
    int json_mode = 0;
    int porcelain_mode = 0;

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)
    long arg;
    for(arg = 1; arg < data.cmdNBarg; arg++)
    {
        if(data.cmdargtoken[arg].type == CMDARGTOKEN_TYPE_STRING
                || data.cmdargtoken[arg].type == CMDARGTOKEN_TYPE_RAWSTRING)
        {
            if(strcmp(data.cmdargtoken[arg].val.string, "--json") == 0)
            {
                json_mode = 1;
            }
            if(strcmp(data.cmdargtoken[arg].val.string, "--porcelain") == 0)
            {
                porcelain_mode = 1;
            }
        }
    }
#endif

    if(json_mode)
    {
        list_image_ID_ofp_json(stdout);
    }
    else if(porcelain_mode)
    {
        list_image_ID_ofp_porcelain(stdout);
    }
    else
    {
        list_image_ID();
    }
    return RETURN_SUCCESS;
}


/* ================================================================
 *  REGISTRATION
 * ============================================================= */

#if !defined(FPS_STANDALONE) && !defined(MILK_NO_CLI)

static errno_t CLIfunction_listim(void)
{
    return safe_fps_generic_CLIfunction(
               &FPS_app_info_listim,
               NULL, &CLIcmddata_listim,
               NULL, 0,
               compute_listim);
}

errno_t
CLIADDCMD_COREMOD_memory__list_image()
{

    {
        int cmdi = RegisterCLIcmd(
                       CLIcmddata_listim,
                       CLIfunction_listim);
        CLIcmddata_listim.cmdsettings =
            &data.cmd[cmdi].cmdsettings;
    }

    return RETURN_SUCCESS;
}
#endif /* MILK_NO_CLI */

errno_t list_image_ID_ofp(FILE *fo)
{
    long               i;
    long               j;
    long long          tmp_long;
    char               type[STYPESIZE];
    uint8_t            datatype;
    int                n;
    unsigned long long sizeb, sizeKb, sizeMb, sizeGb;
    int strmaxlen = 500;
    char               str[strmaxlen];
    int str1maxlen = 512;
    char               str1[str1maxlen];
    struct timespec    timenow;
    double             timediff;
    //struct mallinfo minfo;

    sizeb = compute_image_memory();
    //minfo = mallinfo();

    clock_gettime(CLOCK_MILK, &timenow);
    //fprintf(fo, "time:  %ld.%09ld\n", timenow.tv_sec % 60, timenow.tv_nsec);

    fprintf(fo, "\n");
    fprintf(fo,
            "INDEX    NAME         SIZE                    TYPE        SIZE  "
            "[percent]    LAST ACCESS\n");
    fprintf(fo, "\n");

    for(i = 0; i < dcnimg; i++)
        if(dcimg[i].used == 1)
        {
            datatype = dcimg[i].md[0].datatype;
            tmp_long = ((long long)(dcimg[i].md[0].nelement)) *
                       ImageStreamIO_typesize(datatype);

            if(dcimg[i].md[0].shared == 1)
            {
                fprintf(fo,
                        "%4ld %c[%d;%dm%14s%c[%d;m ",
                        i,
                        (char) 27,
                        1,
                        34,
                        dcimg[i].name,
                        (char) 27,
                        0);
            }
            else
            {
                fprintf(fo,
                        "%4ld %c[%d;%dm%14s%c[%d;m ",
                        i,
                        (char) 27,
                        1,
                        33,
                        dcimg[i].name,
                        (char) 27,
                        0);
            }
            //fprintf(fo, "%s", str);

            snprintf(str, strmaxlen, "[ %6ld", (long) dcimg[i].md[0].size[0]);

            for(j = 1; j < dcimg[i].md[0].naxis; j++)
            {
                snprintf(str1,
                         str1maxlen,
                         "%s x %6ld",
                         str,
                         (long) dcimg[i].md[0].size[j]);
                snprintf(str, strmaxlen,
                         "%s", str1);
            }
            snprintf(str1, str1maxlen,
                     "%s]", str);
            snprintf(str, strmaxlen,
                     "%s", str1);

            fprintf(fo, "%-32s", str);

            n = snprintf(type, STYPESIZE, "%s", ImageStreamIO_typename_7(datatype));

            fprintf(fo, "%7s ", type);

            if(n >= STYPESIZE)
            {
                PRINT_ERROR(
                    "Attempted to write string buffer with too many "
                    "characters");
            }

            fprintf(fo,
                    "%10ld Kb %6.2f   ",
                    (long)(tmp_long / 1024),
                    (float)(100.0 * tmp_long / sizeb));

            timediff =
                (1.0 * timenow.tv_sec + 0.000000001 * timenow.tv_nsec) -
                (1.0 * dcimg[i].md[0].lastaccesstime.tv_sec +
                 0.000000001 * dcimg[i].md[0].lastaccesstime.tv_nsec);

            fprintf(fo, "%15.9f\n", timediff);
        }
    fprintf(fo, "\n");

    sizeGb = 0;
    sizeMb = 0;
    sizeKb = 0;
    sizeb  = compute_image_memory();

    if(sizeb > 1024 - 1)
    {
        sizeKb = sizeb / 1024;
        sizeb  = sizeb - 1024 * sizeKb;
    }
    if(sizeKb > 1024 - 1)
    {
        sizeMb = sizeKb / 1024;
        sizeKb = sizeKb - 1024 * sizeMb;
    }
    if(sizeMb > 1024 - 1)
    {
        sizeGb = sizeMb / 1024;
        sizeMb = sizeMb - 1024 * sizeGb;
    }

    fprintf(fo, "%ld image(s)   ", compute_nb_image());
    if(sizeGb > 0)
    {
        fprintf(fo, " %ld Gb", (long)(sizeGb));
    }
    if(sizeMb > 0)
    {
        fprintf(fo, " %ld Mb", (long)(sizeMb));
    }
    if(sizeKb > 0)
    {
        fprintf(fo, " %ld Kb", (long)(sizeKb));
    }
    if(sizeb > 0)
    {
        fprintf(fo, " %ld", (long)(sizeb));
    }
    fprintf(fo, "\n");

    fflush(fo);

    return RETURN_SUCCESS;
}

errno_t list_image_ID_ofp_simple(FILE *fo)
{
    long i, j;
    //long long   tmp_long;
    uint8_t datatype;

    for(i = 0; i < dcnimg; i++)
        if(dcimg[i].used == 1)
        {
            datatype = dcimg[i].md[0].datatype;
            //tmp_long = ((long long) (dcimg[i].md[0].nelement)) * ImageStreamIO_typesize(datatype);

            fprintf(fo,
                    "%20s %d %ld %d %4ld",
                    dcimg[i].name,
                    datatype,
                    (long) dcimg[i].md[0].naxis,
                    dcimg[i].md[0].shared,
                    (long) dcimg[i].md[0].size[0]);

            for(j = 1; j < dcimg[i].md[0].naxis; j++)
            {
                fprintf(fo, " %4ld", (long) dcimg[i].md[0].size[j]);
            }
            fprintf(fo, "\n");

        }
    fprintf(fo, "\n");

    return RETURN_SUCCESS;
}

/**
 * @brief Print a summary of all images in the array.
 *
 * Lists name, type, size, and memory usage for
 * each active image.
 */
errno_t list_image_ID()
{
    list_image_ID_ofp(stdout);
    //malloc_stats();
    return RETURN_SUCCESS;
}

errno_t list_image_ID_ofp_json(FILE *fo)
{
    long i, j;
    long long   tmp_long;
    uint8_t datatype;
    unsigned long long sizeb = compute_image_memory();
    struct timespec timenow;
    clock_gettime(CLOCK_MILK, &timenow);

    fprintf(fo, "{\n  \"images\": [\n");
    int first = 1;
    for(i = 0; i < dcnimg; i++)
        if(dcimg[i].used == 1)
        {
            if(!first)
            {
                fprintf(fo, ",\n");
            }
            first = 0;
            datatype = dcimg[i].md[0].datatype;
            tmp_long = ((long long)(dcimg[i].md[0].nelement)) * ImageStreamIO_typesize(datatype);

            fprintf(fo, "    {\n");
            fprintf(fo, "      \"index\": %ld,\n", i);
            fprintf(fo, "      \"name\": \"%s\",\n", dcimg[i].name);
            fprintf(fo, "      \"naxis\": %ld,\n", (long) dcimg[i].md[0].naxis);
            fprintf(fo, "      \"size\": [");
            for(j = 0; j < dcimg[i].md[0].naxis; j++)
            {
                fprintf(fo, "%ld%s", (long) dcimg[i].md[0].size[j], (j < dcimg[i].md[0].naxis - 1) ? ", " : "");
            }
            fprintf(fo, "],\n");
            fprintf(fo, "      \"type\": \"%s\",\n", ImageStreamIO_typename_7(datatype));
            fprintf(fo, "      \"shared\": %ld,\n", (long) dcimg[i].md[0].shared);
            fprintf(fo, "      \"size_kb\": %ld,\n", (long)(tmp_long / 1024));

            double timediff = (1.0 * timenow.tv_sec + 0.000000001 * timenow.tv_nsec) -
                              (1.0 * dcimg[i].md[0].lastaccesstime.tv_sec +
                               0.000000001 * dcimg[i].md[0].lastaccesstime.tv_nsec);
            fprintf(fo, "      \"last_access_dt\": %.9f\n", timediff);
            fprintf(fo, "    }");
        }
    fprintf(fo, "\n  ],\n");
    fprintf(fo, "  \"summary\": {\n");
    fprintf(fo, "    \"total_images\": %ld,\n", compute_nb_image());
    fprintf(fo, "    \"total_size_b\": %llu\n", (unsigned long long)sizeb);
    fprintf(fo, "  }\n}\n");

    return RETURN_SUCCESS;
}

errno_t list_image_ID_ofp_porcelain(FILE *fo)
{
    long i, j;
    long long   tmp_long;
    uint8_t datatype;
    struct timespec timenow;
    clock_gettime(CLOCK_MILK, &timenow);

    fprintf(fo, "INDEX\tNAME\tNAXIS\tSIZE\tTYPE\tSHARED\tSIZE_KB\tLAST_ACCESS_DT\n");

    for(i = 0; i < dcnimg; i++)
        if(dcimg[i].used == 1)
        {
            datatype = dcimg[i].md[0].datatype;
            tmp_long = ((long long)(dcimg[i].md[0].nelement)) * ImageStreamIO_typesize(datatype);

            fprintf(fo, "%ld\t%s\t%ld\t", i, dcimg[i].name, (long) dcimg[i].md[0].naxis);
            for(j = 0; j < dcimg[i].md[0].naxis; j++)
            {
                fprintf(fo, "%ld%s", (long) dcimg[i].md[0].size[j], (j < dcimg[i].md[0].naxis - 1) ? "x" : "");
            }
            double timediff = (1.0 * timenow.tv_sec + 0.000000001 * timenow.tv_nsec) -
                              (1.0 * dcimg[i].md[0].lastaccesstime.tv_sec +
                               0.000000001 * dcimg[i].md[0].lastaccesstime.tv_nsec);
            fprintf(fo, "\t%s\t%ld\t%ld\t%.9f\n",
                    ImageStreamIO_typename_7(datatype),
                    (long) dcimg[i].md[0].shared,
                    (long)(tmp_long / 1024),
                    timediff);
        }

    return RETURN_SUCCESS;
}

/* list all images in memory
   output is written in ASCII file
   only basic info is listed
   image name
   number of axis
   size
   type
 */

errno_t list_image_ID_file(const char *fname)
{
    FILE   *fp;
    long    i, j;
    uint8_t datatype;
    char    type[STYPESIZE];
    int     n;

    fp = fopen(fname, "w");
    if(fp == NULL)
    {
        PRINT_ERROR("Cannot create file %s", fname);
        abort();
    }

    for(i = 0; i < dcnimg; i++)
        if(dcimg[i].used == 1)
        {
            datatype = dcimg[i].md[0].datatype;
            fprintf(fp, "%ld %s", i, dcimg[i].name);
            fprintf(fp, " %ld", (long) dcimg[i].md[0].naxis);
            for(j = 0; j < dcimg[i].md[0].naxis; j++)
            {
                fprintf(fp, " %ld", (long) dcimg[i].md[0].size[j]);
            }

            n = snprintf(type, STYPESIZE, "%s", ImageStreamIO_typename_7(datatype));

            if(n >= STYPESIZE)
            {
                PRINT_ERROR(
                    "Attempted to write string buffer with too many "
                    "characters");
            }

            fprintf(fp, " %s\n", type);
        }
    fclose(fp);

    return RETURN_SUCCESS;
}
