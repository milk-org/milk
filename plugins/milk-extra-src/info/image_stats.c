/** @file image_stats.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_tools/COREMOD_tools.h"


// ==========================================
// Forward declaration(s)
// ==========================================

errno_t info_image_stats(const char *ID_name, const char *options);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t info_image_stats_cli()
{
    if(CLI_checkarg(1, CLIARG_IMG) == 0)
    {
        info_image_stats(data.cmdargtoken[1].val.string, "");
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

static errno_t info_image_statsf_cli()
{
    if(CLI_checkarg(1, CLIARG_IMG) == 0)
    {
        info_image_stats(data.cmdargtoken[1].val.string, "fileout");
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t image_stats_addCLIcmd()
{
    RegisterCLIcommand("imstats",
                       __FILE__,
                       info_image_stats_cli,
                       "image stats",
                       "<image>",
                       "imgstats im1",
                       "int info_image_stats(const char *ID_name, \"\")");

    RegisterCLIcommand(
        "imstatsf",
        __FILE__,
        info_image_statsf_cli,
        "image stats with file output",
        "<image>",
        "imgstatsf im1",
        "int info_image_stats(const char *ID_name, \"fileout\")");

    return RETURN_SUCCESS;
}

// option "fileout" : output to file imstat.info.txt
errno_t info_image_stats(const char *ID_name, const char *options)
{
    imageID  ID;
    double   min, max;
    double   rms;
    uint64_t nelements;
    double   tot;
    double  *array;
    long     iimin, iimax;
    uint8_t  datatype;
    long     tmp_long;
    char     type[20];
    char     vname[200];
    double   xtot, ytot;
    double   vbx, vby;
    FILE    *fp;
    int      mode = 0;

    // printf("OPTIONS = %s\n",options);
    if(strstr(options, "fileout") != NULL)
    {
        mode = 1;
    }

    if(mode == 1)
    {
        fp = fopen("imstat.info.txt", "w");
    }

    ID = image_ID_noaccessupdate(ID_name);
    if(ID != -1)
    {
        nelements = data.image[ID].md[0].nelement;

        datatype = data.image[ID].md[0].datatype;
        tmp_long =
            data.image[ID].md[0].nelement * ImageStreamIO_typesize(datatype);
        printf("\n");
        printf("Image size (->imsize0...):     [");
        printf("% ld", (long) data.image[ID].md[0].size[0]);

        unsigned long j = 0;
        sprintf(vname, "imsize%ld", j);

        create_variable_ID(vname, 1.0 * data.image[ID].md[0].size[j]);
        for(j = 1; j < data.image[ID].md[0].naxis; j++)
        {
            printf(" %ld", (long) data.image[ID].md[0].size[j]);
            sprintf(vname, "imsize%ld", j);
            create_variable_ID(vname, 1.0 * data.image[ID].md[0].size[j]);
        }
        printf(" ]\n");

        printf("write = %d   cnt0 = %ld   cnt1 = %ld\n",
               data.image[ID].md[0].write,
               data.image[ID].md[0].cnt0,
               data.image[ID].md[0].cnt1);

        switch(datatype)
        {
            case _DATATYPE_FLOAT:
                sprintf(type, "  FLOAT");
                break;

            case _DATATYPE_INT8:
                sprintf(type, "   INT8");
                break;

            case _DATATYPE_UINT8:
                sprintf(type, "  UINT8");
                break;

            case _DATATYPE_INT16:
                sprintf(type, "  INT16");
                break;

            case _DATATYPE_UINT16:
                sprintf(type, " UINT16");
                break;

            case _DATATYPE_INT32:
                sprintf(type, "  INT32");
                break;

            case _DATATYPE_UINT32:
                sprintf(type, " UINT32");
                break;

            case _DATATYPE_INT64:
                sprintf(type, "  INT64");
                break;

            case _DATATYPE_UINT64:
                sprintf(type, " UINT64");
                break;

            case _DATATYPE_DOUBLE:
                sprintf(type, " DOUBLE");
                break;

            case _DATATYPE_COMPLEX_FLOAT:
                sprintf(type, "CFLOAT");
                break;

            case _DATATYPE_COMPLEX_DOUBLE:
                sprintf(type, "CDOUBLE");
                break;

            default:
                sprintf(type, "??????");
                break;
        }

        printf("type:            %s\n", type);
        printf("Memory size:     %ld Kb\n", (long) tmp_long / 1024);
        //      printf("Created:         %f\n", data.image[ID].creation_time);
        //      printf("Last access:     %f\n", data.image[ID].last_access);

        if(datatype == _DATATYPE_FLOAT)
        {
            min = data.image[ID].array.F[0];
            max = data.image[ID].array.F[0];

            iimin = 0;
            iimax = 0;
            for(unsigned long ii = 0; ii < nelements; ii++)
            {
                if(min > data.image[ID].array.F[ii])
                {
                    min   = data.image[ID].array.F[ii];
                    iimin = ii;
                }
                if(max < data.image[ID].array.F[ii])
                {
                    max   = data.image[ID].array.F[ii];
                    iimax = ii;
                }
            }

            array = (double *) malloc(nelements * sizeof(double));
            tot   = 0.0;

            rms = 0.0;
            for(unsigned long ii = 0; ii < nelements; ii++)
            {
                if(isnan(data.image[ID].array.F[ii]) != 0)
                {
                    printf(
                        "element %ld is NAN -> replacing by "
                        "0\n",
                        ii);
                    data.image[ID].array.F[ii] = 0.0;
                }
                tot += data.image[ID].array.F[ii];
                rms += data.image[ID].array.F[ii] * data.image[ID].array.F[ii];
                array[ii] = data.image[ID].array.F[ii];
            }
            rms = sqrt(rms);

            printf("minimum         (->vmin)     %20.18e [ pix %ld ]\n",
                   min,
                   iimin);
            if(mode == 1)
            {
                fprintf(fp,
                        "minimum                  %20.18e [ pix "
                        "%ld ]\n",
                        min,
                        iimin);
            }
            create_variable_ID("vmin", min);
            printf("maximum         (->vmax)     %20.18e [ pix %ld ]\n",
                   max,
                   iimax);
            if(mode == 1)
            {
                fprintf(fp,
                        "maximum                  %20.18e [ pix "
                        "%ld ]\n",
                        max,
                        iimax);
            }
            create_variable_ID("vmax", max);
            printf("total           (->vtot)     %20.18e\n", tot);
            if(mode == 1)
            {
                fprintf(fp, "total                    %20.18e\n", tot);
            }
            create_variable_ID("vtot", tot);
            printf("rms             (->vrms)     %20.18e\n", rms);
            if(mode == 1)
            {
                fprintf(fp, "rms                      %20.18e\n", rms);
            }
            create_variable_ID("vrms", rms);
            printf("rms per pixel   (->vrmsp)    %20.18e\n",
                   rms / sqrt(nelements));
            if(mode == 1)
            {
                fprintf(fp,
                        "rms per pixel            %20.18e\n",
                        rms / sqrt(nelements));
            }
            create_variable_ID("vrmsp", rms / sqrt(nelements));
            printf("rms dev per pix (->vrmsdp)   %20.18e\n",
                   sqrt(rms * rms / nelements -
                        tot * tot / nelements / nelements));
            create_variable_ID("vrmsdp",
                               sqrt(rms * rms / nelements -
                                    tot * tot / nelements / nelements));
            printf("mean            (->vmean)    %20.18e\n", tot / nelements);
            if(mode == 1)
            {
                fprintf(fp,
                        "mean                     %20.18e\n",
                        tot / nelements);
            }
            create_variable_ID("vmean", tot / nelements);

            if(data.image[ID].md[0].naxis == 2)
            {
                xtot = 0.0;
                ytot = 0.0;
                for(unsigned long ii = 0; ii < data.image[ID].md[0].size[0];
                        ii++)
                    for(unsigned long jj = 0;
                            jj < data.image[ID].md[0].size[1];
                            jj++)
                    {
                        xtot += data.image[ID]
                                .array
                                .F[jj * data.image[ID].md[0].size[0] + ii] *
                                ii;
                        ytot += data.image[ID]
                                .array
                                .F[jj * data.image[ID].md[0].size[0] + ii] *
                                jj;
                    }
                vbx = xtot / tot;
                vby = ytot / tot;
                printf("Barycenter x    (->vbx)      %20.18f\n", vbx);
                if(mode == 1)
                {
                    fprintf(fp, "photocenterX             %20.18e\n", vbx);
                }
                create_variable_ID("vbx", vbx);
                printf("Barycenter y    (->vby)      %20.18f\n", vby);
                if(mode == 1)
                {
                    fprintf(fp, "photocenterY             %20.18e\n", vby);
                }
                create_variable_ID("vby", vby);
            }

            quick_sort_double(array, nelements);
            printf("\n");
            printf("percentile values:\n");

            printf("1  percent      (->vp01)     %20.18e\n",
                   array[(long)(0.01 * nelements)]);
            if(mode == 1)
            {
                fprintf(fp,
                        "percentile01             %20.18e\n",
                        array[(long)(0.01 * nelements)]);
            }
            create_variable_ID("vp01", array[(long)(0.01 * nelements)]);

            printf("5  percent      (->vp05)     %20.18e\n",
                   array[(long)(0.05 * nelements)]);
            if(mode == 1)
            {
                fprintf(fp,
                        "percentile05             %20.18e\n",
                        array[(long)(0.05 * nelements)]);
            }
            create_variable_ID("vp05", array[(long)(0.05 * nelements)]);

            printf("10 percent      (->vp10)     %20.18e\n",
                   array[(long)(0.1 * nelements)]);
            if(mode == 1)
            {
                fprintf(fp,
                        "percentile10             %20.18e\n",
                        array[(long)(0.10 * nelements)]);
            }
            create_variable_ID("vp10", array[(long)(0.1 * nelements)]);

            printf("20 percent      (->vp20)     %20.18e\n",
                   array[(long)(0.2 * nelements)]);
            if(mode == 1)
            {
                fprintf(fp,
                        "percentile20             %20.18e\n",
                        array[(long)(0.20 * nelements)]);
            }
            create_variable_ID("vp20", array[(long)(0.2 * nelements)]);

            printf("50 percent      (->vp50)     %20.18e\n",
                   array[(long)(0.5 * nelements)]);
            if(mode == 1)
            {
                fprintf(fp,
                        "percentile50             %20.18e\n",
                        array[(long)(0.50 * nelements)]);
            }
            create_variable_ID("vp50", array[(long)(0.5 * nelements)]);

            printf("80 percent      (->vp80)     %20.18e\n",
                   array[(long)(0.8 * nelements)]);
            if(mode == 1)
            {
                fprintf(fp,
                        "percentile80             %20.18e\n",
                        array[(long)(0.80 * nelements)]);
            }
            create_variable_ID("vp80", array[(long)(0.8 * nelements)]);

            printf("90 percent      (->vp90)     %20.18e\n",
                   array[(long)(0.9 * nelements)]);
            if(mode == 1)
            {
                fprintf(fp,
                        "percentile90             %20.18e\n",
                        array[(long)(0.90 * nelements)]);
            }
            create_variable_ID("vp90", array[(long)(0.9 * nelements)]);

            printf("95 percent      (->vp95)     %20.18e\n",
                   array[(long)(0.95 * nelements)]);
            if(mode == 1)
            {
                fprintf(fp,
                        "percentile95             %20.18e\n",
                        array[(long)(0.95 * nelements)]);
            }
            create_variable_ID("vp95", array[(long)(0.95 * nelements)]);

            printf("99 percent      (->vp99)     %20.18e\n",
                   array[(long)(0.99 * nelements)]);
            if(mode == 1)
            {
                fprintf(fp,
                        "percentile99             %20.18e\n",
                        array[(long)(0.99 * nelements)]);
            }
            create_variable_ID("vp99", array[(long)(0.99 * nelements)]);

            printf("99.5 percent    (->vp995)    %20.18e\n",
                   array[(long)(0.995 * nelements)]);
            if(mode == 1)
            {
                fprintf(fp,
                        "percentile995            %20.18e\n",
                        array[(long)(0.995 * nelements)]);
            }
            create_variable_ID("vp995", array[(long)(0.995 * nelements)]);

            printf("99.8 percent    (->vp998)    %20.18e\n",
                   array[(long)(0.998 * nelements)]);
            if(mode == 1)
            {
                fprintf(fp,
                        "percentile998            %20.18e\n",
                        array[(long)(0.998 * nelements)]);
            }
            create_variable_ID("vp998", array[(long)(0.998 * nelements)]);

            printf("99.9 percent    (->vp999)    %20.18e\n",
                   array[(long)(0.999 * nelements)]);
            if(mode == 1)
            {
                fprintf(fp,
                        "percentile999            %20.18e\n",
                        array[(long)(0.999 * nelements)]);
            }
            create_variable_ID("vp999", array[(long)(0.999 * nelements)]);

            printf("\n");
            free(array);
        }
    }

    if(mode == 1)
    {
        fclose(fp);
    }

    return RETURN_SUCCESS;
}
