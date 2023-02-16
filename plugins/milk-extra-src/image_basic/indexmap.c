/** @file indexmap.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID image_basic_indexmap(const char *__restrict ID_index_name,
                             const char *__restrict ID_values_name,
                             const char *__restrict IDout_name);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t image_basic_indexmap_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 4) + CLI_checkarg(3, 3) == 0)
    {
        image_basic_indexmap(data.cmdargtoken[1].val.string,
                             data.cmdargtoken[2].val.string,
                             data.cmdargtoken[3].val.string);
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

errno_t __attribute__((cold)) indexmap_addCLIcmd()
{

    RegisterCLIcommand("imindexmap",
                       __FILE__,
                       image_basic_indexmap_cli,
                       "map input values to output image unsing index map",
                       "imindexmap <indexmap> <values> <output>",
                       "imindexmap imap imval outmap",
                       "long image_basic_indexmap(char *ID_index_name, char "
                       "*ID_values_name, char *IDout_name)");

    return RETURN_SUCCESS;
}

imageID image_basic_indexmap(const char *__restrict ID_index_name,
                             const char *__restrict ID_values_name,
                             const char *__restrict IDout_name)
{
    imageID IDindex, IDvalues;
    imageID IDout;
    long    xsize, ysize, xysize;
    long    val_xsize, val_ysize, val_xysize;
    uint8_t datatype;
    uint8_t val_datatype;
    long    ii, i;

    IDindex  = image_ID(ID_index_name);
    IDvalues = image_ID(ID_values_name);

    xsize    = data.image[IDindex].md[0].size[0];
    ysize    = data.image[IDindex].md[0].size[1];
    xysize   = xsize * ysize;
    datatype = data.image[IDindex].md[0].datatype;

    val_xsize    = data.image[IDvalues].md[0].size[0];
    val_ysize    = data.image[IDvalues].md[0].size[1];
    val_xysize   = val_xsize * val_ysize;
    val_datatype = data.image[IDindex].md[0].datatype;

    create_2Dimage_ID(IDout_name, xsize, ysize, &IDout);

    if(val_datatype == _DATATYPE_FLOAT)
    {
        for(ii = 0; ii < xysize; ii++)
        {
            i = (long)(data.image[IDindex].array.F[ii] + 0.1);
            if((i > -1) && (i < val_xysize))
            {
                data.image[IDout].array.F[ii] = data.image[IDvalues].array.F[i];
            }
        }
    }
    else
    {
        float *arrayf = (float *) malloc(sizeof(float) * val_xysize);
        if(arrayf == NULL)
        {
            PRINT_ERROR("malloc returns NULL pointer");
            abort();
        }

        for(i = 0; i < val_xysize; i++)
        {
            arrayf[i] = (float) data.image[IDvalues].array.D[i];
        }

        switch(datatype)
        {

            case _DATATYPE_DOUBLE:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long)(data.image[IDindex].array.D[ii] + 0.1);
                    if((i > -1) && (i < val_xysize))
                    {
                        data.image[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_UINT8:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) data.image[IDindex].array.UI8[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        data.image[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_INT8:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) data.image[IDindex].array.SI8[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        data.image[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_UINT16:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) data.image[IDindex].array.UI16[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        data.image[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_INT16:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) data.image[IDindex].array.SI16[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        data.image[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_UINT32:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) data.image[IDindex].array.UI32[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        data.image[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_INT32:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) data.image[IDindex].array.SI32[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        data.image[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_UINT64:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) data.image[IDindex].array.UI64[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        data.image[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            case _DATATYPE_INT64:
                for(ii = 0; ii < xysize; ii++)
                {
                    i = (long) data.image[IDindex].array.SI64[ii];
                    if((i > -1) && (i < val_xysize))
                    {
                        data.image[IDout].array.F[ii] = arrayf[i];
                    }
                }
                break;

            default:
                printf("ERROR: datatype not supported\n");
                free(arrayf);
                return EXIT_FAILURE;
                break;
        }
        free(arrayf);
    }

    return (IDout);
}
