/** @file imtoASCII.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

errno_t IMAGE_FORMAT_im_to_ASCII(const char *__restrict IDname,
                                 const char *__restrict foutname);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t IMAGE_FORMAT_im_to_ASCII_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 3) == 0)
    {
        IMAGE_FORMAT_im_to_ASCII(data.cmdargtoken[1].val.string,
                                 data.cmdargtoken[2].val.string);
        return RETURN_SUCCESS;
    }
    else
    {
        return RETURN_FAILURE;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t imtoASCII_addCLIcmd()
{

    RegisterCLIcommand(
        "im2ascii",
        __FILE__,
        IMAGE_FORMAT_im_to_ASCII_cli,
        "convert image file to ASCII",
        "<input image> <output ASCII file>",
        "im2ascii im im.txt",
        "int IMAGE_FORMAT_im_to_ASCII(const char *IDname, const char *fname)");

    return RETURN_SUCCESS;
}

errno_t IMAGE_FORMAT_im_to_ASCII(const char *__restrict IDname,
                                 const char *__restrict foutname)
{
    long    ii;
    long    k;
    imageID ID;
    FILE   *fpout;
    long    naxis;
    long   *coord;
    long    npix;

    ID    = image_ID(IDname);
    naxis = data.image[ID].md[0].naxis;
    coord = (long *) malloc(sizeof(long) * naxis);
    if(coord == NULL)
    {
        PRINT_ERROR("malloc returns NULL pointer");
        abort();
    }

    npix = 1;
    for(k = 0; k < naxis; k++)
    {
        npix *= data.image[ID].md[0].size[k];
        coord[k] = 0;
    }

    printf("npix = %ld\n", npix);

    fpout = fopen(foutname, "w");

    for(ii = 0; ii < npix; ii++)
    {
        int kOK;

        for(k = 0; k < naxis; k++)
        {
            fprintf(fpout, "%4ld ", coord[k]);
        }
        switch(data.image[ID].md[0].datatype)
        {
            case _DATATYPE_UINT8:
                fprintf(fpout, " %5u\n", data.image[ID].array.UI8[ii]);
                break;
            case _DATATYPE_UINT16:
                fprintf(fpout, " %5u\n", data.image[ID].array.UI16[ii]);
                break;
            case _DATATYPE_UINT32:
                fprintf(fpout, " %u\n", data.image[ID].array.UI32[ii]);
                break;
            case _DATATYPE_UINT64:
                fprintf(fpout, " %lu\n", data.image[ID].array.UI64[ii]);
                break;

            case _DATATYPE_INT8:
                fprintf(fpout, " %5d\n", data.image[ID].array.SI8[ii]);
                break;
            case _DATATYPE_INT16:
                fprintf(fpout, " %5d\n", data.image[ID].array.SI16[ii]);
                break;
            case _DATATYPE_INT32:
                fprintf(fpout, " %5d\n", data.image[ID].array.SI32[ii]);
                break;
            case _DATATYPE_INT64:
                fprintf(fpout, " %5ld\n", data.image[ID].array.SI64[ii]);
                break;

            case _DATATYPE_FLOAT:
                fprintf(fpout, " %f\n", data.image[ID].array.F[ii]);
                break;
            case _DATATYPE_DOUBLE:
                fprintf(fpout, " %lf\n", data.image[ID].array.D[ii]);
                break;
        }
        coord[0]++;

        k   = 0;
        kOK = 0;
        while((kOK == 0) && (k < naxis))
        {
            if(coord[k] == data.image[ID].md[0].size[k])
            {
                coord[k] = 0;
                coord[k + 1]++;
            }
            else
            {
                kOK = 1;
            }
            k++;
        }
    }
    fclose(fpout);

    free(coord);

    return RETURN_SUCCESS;
}
