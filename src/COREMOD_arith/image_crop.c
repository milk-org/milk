/**
 * @file    image_crop.c
 * @brief   crop functions
 *
 *
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID arith_image_crop(const char *ID_name,
                         const char *ID_out,
                         long       *start,
                         long       *end,
                         long        cropdim);

imageID arith_image_extract2D(const char *in_name,
                              const char *out_name,
                              long        size_x,
                              long        size_y,
                              long        xstart,
                              long        ystart);

imageID arith_image_extract3D(const char *in_name,
                              const char *out_name,
                              long        size_x,
                              long        size_y,
                              long        size_z,
                              long        xstart,
                              long        ystart,
                              long        zstart);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t arith_image_extract2D__cli()
{
    if(0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(3, CLIARG_INT64) + CLI_checkarg(4, CLIARG_INT64) +
            CLI_checkarg(5, CLIARG_INT64) + CLI_checkarg(6, CLIARG_INT64) ==
            0)
    {
        arith_image_extract2D(data.cmdargtoken[1].val.string,
                              data.cmdargtoken[2].val.string,
                              data.cmdargtoken[3].val.numl,
                              data.cmdargtoken[4].val.numl,
                              data.cmdargtoken[5].val.numl,
                              data.cmdargtoken[6].val.numl);

        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

static errno_t arith_image_extract3D__cli()
{
    if(0 + CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(3, CLIARG_INT64) + CLI_checkarg(4, CLIARG_INT64) +
            CLI_checkarg(5, CLIARG_INT64) + CLI_checkarg(6, CLIARG_INT64) +
            CLI_checkarg(7, CLIARG_INT64) + CLI_checkarg(8, CLIARG_INT64) ==
            0)
    {
        arith_image_extract3D(data.cmdargtoken[1].val.string,
                              data.cmdargtoken[2].val.string,
                              data.cmdargtoken[3].val.numl,
                              data.cmdargtoken[4].val.numl,
                              data.cmdargtoken[5].val.numl,
                              data.cmdargtoken[6].val.numl,
                              data.cmdargtoken[7].val.numl,
                              data.cmdargtoken[8].val.numl);

        return CLICMD_SUCCESS;
    }
    else
    {
        return 1;
    }
}

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t image_crop_addCLIcmd()
{

    RegisterCLIcommand(
        "extractim",
        __FILE__,
        arith_image_extract2D__cli,
        "crop 2D image",
        "<input image> <output image> <sizex> <sizey> <xstart> <ystart>",
        "extractim im ime 256 256 100 100",
        "int arith_image_extract2D(const char *in_name, const char *out_name, "
        "long size_x, long size_y, "
        "long xstart, long ystart)");

    RegisterCLIcommand("extract3Dim",
                       __FILE__,
                       arith_image_extract3D__cli,
                       "crop 3D image",
                       "<input image> <output image> <sizex> <sizey> <sizez> "
                       "<xstart> <ystart> <zstart>",
                       "extractim im ime 256 256 5 100 100 0",
                       "int arith_image_extract3D(const char *in_name, const "
                       "char *out_name, long size_x, long size_y, "
                       "long size_z, long xstart, long ystart, long zstart)");

    return RETURN_SUCCESS;
}

imageID arith_image_crop(const char *ID_name,
                         const char *ID_out,
                         long       *start,
                         long       *end,
                         long        cropdim)
{
    long      naxis;
    imageID   IDin;
    imageID   IDout;
    long      i;
    uint32_t *naxes    = NULL;
    uint32_t *naxesout = NULL;
    uint8_t   datatype;

    long start_c[3];
    long end_c[3];

    for(i = 0; i < 3; i++)
    {
        start_c[i] = 0;
        end_c[i]   = 0;
    }

    IDin = image_ID(ID_name);
    if(IDin == -1)
    {
        PRINT_ERROR("Missing input image = %s", ID_name);
        list_image_ID();
        exit(0);
    }

    naxis = data.image[IDin].md[0].naxis;
    if(naxis < 1)
    {
        PRINT_ERROR("naxis < 1");
        exit(0);
    }
    naxes = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(naxes == NULL)
    {
        PRINT_ERROR("malloc() error, naxis = %ld", naxis);
        exit(0);
    }

    naxesout = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    if(naxesout == NULL)
    {
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    datatype = data.image[IDin].md[0].datatype;

    naxes[0]    = 0;
    naxesout[0] = 0;
    for(i = 0; i < naxis; i++)
    {
        naxes[i]    = data.image[IDin].md[0].size[i];
        naxesout[i] = end[i] - start[i];
    }
    create_image_ID(ID_out,
                    naxis,
                    naxesout,
                    datatype,
                    data.SHARED_DFT,
                    NB_KEYWNODE_MAX,
                    0,
                    &IDout);

    start_c[0] = start[0];
    if(start_c[0] < 0)
    {
        start_c[0] = 0;
    }
    end_c[0] = end[0];
    if(end_c[0] > naxes[0])
    {
        end_c[0] = naxes[0];
    }
    if(naxis > 1)
    {
        start_c[1] = start[1];
        if(start_c[1] < 0)
        {
            start_c[1] = 0;
        }
        end_c[1] = end[1];
        if(end_c[1] > naxes[1])
        {
            end_c[1] = naxes[1];
        }
    }
    if(naxis > 2)
    {
        start_c[2] = start[2];
        if(start_c[2] < 0)
        {
            start_c[2] = 0;
        }
        end_c[2] = end[2];
        if(end_c[2] > naxes[2])
        {
            end_c[2] = naxes[2];
        }
    }

    printf("CROP: \n");
    for(i = 0; i < 3; i++)
    {
        printf("axis %ld: %ld -> %ld\n", i, start_c[i], end_c[i]);
    }

    if(cropdim != naxis)
    {
        printf(
            "Error (arith_image_crop): cropdim [%ld] and naxis [%ld] are "
            "different\n",
            cropdim,
            naxis);
    }

    if(naxis == 1)
    {
        if(datatype == _DATATYPE_FLOAT)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                data.image[IDout].array.F[ii - start[0]] =
                    data.image[IDin].array.F[ii];
            }
        }
        else if(datatype == _DATATYPE_DOUBLE)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                data.image[IDout].array.D[ii - start[0]] =
                    data.image[IDin].array.D[ii];
            }
        }
        else if(datatype == _DATATYPE_COMPLEX_FLOAT)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                data.image[IDout].array.CF[ii - start[0]].re =
                    data.image[IDin].array.CF[ii].re;
                data.image[IDout].array.CF[ii - start[0]].im =
                    data.image[IDin].array.CF[ii].im;
            }
        }
        else if(datatype == _DATATYPE_COMPLEX_DOUBLE)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                data.image[IDout].array.CD[ii - start[0]].re =
                    data.image[IDin].array.CD[ii].re;
                data.image[IDout].array.CD[ii - start[0]].im =
                    data.image[IDin].array.CD[ii].im;
            }
        }
        else if(datatype == _DATATYPE_UINT8)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                data.image[IDout].array.UI8[ii - start[0]] =
                    data.image[IDin].array.UI8[ii];
            }
        }
        else if(datatype == _DATATYPE_UINT16)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                data.image[IDout].array.UI16[ii - start[0]] =
                    data.image[IDin].array.UI16[ii];
            }
        }
        else if(datatype == _DATATYPE_UINT32)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                data.image[IDout].array.UI32[ii - start[0]] =
                    data.image[IDin].array.UI32[ii];
            }
        }
        else if(datatype == _DATATYPE_UINT64)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                data.image[IDout].array.UI64[ii - start[0]] =
                    data.image[IDin].array.UI64[ii];
            }
        }
        else if(datatype == _DATATYPE_INT8)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                data.image[IDout].array.SI8[ii - start[0]] =
                    data.image[IDin].array.SI8[ii];
            }
        }
        else if(datatype == _DATATYPE_INT16)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                data.image[IDout].array.SI16[ii - start[0]] =
                    data.image[IDin].array.SI16[ii];
            }
        }
        else if(datatype == _DATATYPE_INT32)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                data.image[IDout].array.SI32[ii - start[0]] =
                    data.image[IDin].array.SI32[ii];
            }
        }
        else if(datatype == _DATATYPE_INT64)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
            {
                data.image[IDout].array.SI64[ii - start[0]] =
                    data.image[IDin].array.SI64[ii];
            }
        }
        else
        {
            PRINT_ERROR("invalid data type");
            exit(0);
        }
    }
    if(naxis == 2)
    {
        if(datatype == _DATATYPE_FLOAT)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    data.image[IDout].array.F[(jj - start[1]) * naxesout[0] +
                                              (ii - start[0])] =
                                                  data.image[IDin].array.F[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_DOUBLE)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    data.image[IDout].array.D[(jj - start[1]) * naxesout[0] +
                                              (ii - start[0])] =
                                                  data.image[IDin].array.D[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_COMPLEX_FLOAT)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                {
                    data.image[IDout]
                    .array
                    .CF[(jj - start[1]) * naxesout[0] + (ii - start[0])]
                    .re = data.image[IDin].array.CF[jj * naxes[0] + ii].re;
                    data.image[IDout]
                    .array
                    .CF[(jj - start[1]) * naxesout[0] + (ii - start[0])]
                    .im = data.image[IDin].array.CF[jj * naxes[0] + ii].im;
                }
        }
        else if(datatype == _DATATYPE_COMPLEX_DOUBLE)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                {
                    data.image[IDout]
                    .array
                    .CD[(jj - start[1]) * naxesout[0] + (ii - start[0])]
                    .re = data.image[IDin].array.CD[jj * naxes[0] + ii].re;
                    data.image[IDout]
                    .array
                    .CD[(jj - start[1]) * naxesout[0] + (ii - start[0])]
                    .im = data.image[IDin].array.CD[jj * naxes[0] + ii].im;
                }
        }
        else if(datatype == _DATATYPE_UINT8)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    data.image[IDout].array.UI8[(jj - start[1]) * naxesout[0] +
                                                (ii - start[0])] =
                                                    data.image[IDin].array.UI8[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_UINT16)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    data.image[IDout].array.UI16[(jj - start[1]) * naxesout[0] +
                                                 (ii - start[0])] =
                                                     data.image[IDin].array.UI16[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_UINT32)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    data.image[IDout].array.UI32[(jj - start[1]) * naxesout[0] +
                                                 (ii - start[0])] =
                                                     data.image[IDin].array.UI32[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_UINT64)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    data.image[IDout].array.UI64[(jj - start[1]) * naxesout[0] +
                                                 (ii - start[0])] =
                                                     data.image[IDin].array.UI64[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_INT8)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    data.image[IDout].array.SI8[(jj - start[1]) * naxesout[0] +
                                                (ii - start[0])] =
                                                    data.image[IDin].array.SI8[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_INT16)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    data.image[IDout].array.SI16[(jj - start[1]) * naxesout[0] +
                                                 (ii - start[0])] =
                                                     data.image[IDin].array.SI16[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_INT32)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    data.image[IDout].array.SI32[(jj - start[1]) * naxesout[0] +
                                                 (ii - start[0])] =
                                                     data.image[IDin].array.SI32[jj * naxes[0] + ii];
        }
        else if(datatype == _DATATYPE_INT64)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    data.image[IDout].array.SI64[(jj - start[1]) * naxesout[0] +
                                                 (ii - start[0])] =
                                                     data.image[IDin].array.SI64[jj * naxes[0] + ii];
        }
        else
        {
            PRINT_ERROR("invalid data type");
            exit(0);
        }
    }
    if(naxis == 3)
    {
        //	printf("naxis = 3\n");
        if(datatype == _DATATYPE_FLOAT)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        data.image[IDout].array.F
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             data.image[IDin].array.F[kk * naxes[0] * naxes[1] +
                                                      jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_DOUBLE)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        data.image[IDout].array.D
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             data.image[IDin].array.D[kk * naxes[0] * naxes[1] +
                                                      jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_COMPLEX_FLOAT)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        data.image[IDout]
                        .array
                        .CF[(kk - start[2]) * naxesout[0] * naxesout[1] +
                                            (jj - start[1]) * naxesout[0] + (ii - start[0])]
                        .re = data.image[IDin]
                              .array
                              .CF[kk * naxes[0] * naxes[1] +
                                     jj * naxes[0] + ii]
                              .re;
                        data.image[IDout]
                        .array
                        .CF[(kk - start[2]) * naxesout[0] * naxesout[1] +
                                            (jj - start[1]) * naxesout[0] + (ii - start[0])]
                        .im = data.image[IDin]
                              .array
                              .CF[kk * naxes[0] * naxes[1] +
                                     jj * naxes[0] + ii]
                              .im;
                    }
        }
        else if(datatype == _DATATYPE_COMPLEX_DOUBLE)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        data.image[IDout]
                        .array
                        .CD[(kk - start[2]) * naxesout[0] * naxesout[1] +
                                            (jj - start[1]) * naxesout[0] + (ii - start[0])]
                        .re = data.image[IDin]
                              .array
                              .CD[kk * naxes[0] * naxes[1] +
                                     jj * naxes[0] + ii]
                              .re;
                        data.image[IDout]
                        .array
                        .CD[(kk - start[2]) * naxesout[0] * naxesout[1] +
                                            (jj - start[1]) * naxesout[0] + (ii - start[0])]
                        .im = data.image[IDin]
                              .array
                              .CD[kk * naxes[0] * naxes[1] +
                                     jj * naxes[0] + ii]
                              .im;
                    }
        }
        else if(datatype == _DATATYPE_UINT8)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        data.image[IDout].array.UI8
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             data.image[IDin]
                             .array.UI8[kk * naxes[0] * naxes[1] +
                                           jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_UINT16)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        data.image[IDout].array.UI16
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             data.image[IDin]
                             .array.UI16[kk * naxes[0] * naxes[1] +
                                            jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_UINT32)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        data.image[IDout].array.UI32
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             data.image[IDin]
                             .array.UI32[kk * naxes[0] * naxes[1] +
                                            jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_UINT64)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        data.image[IDout].array.UI64
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             data.image[IDin]
                             .array.UI64[kk * naxes[0] * naxes[1] +
                                            jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_INT8)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        data.image[IDout].array.SI8
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             data.image[IDin]
                             .array.SI8[kk * naxes[0] * naxes[1] +
                                           jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_INT16)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        data.image[IDout].array.SI16
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             data.image[IDin]
                             .array.SI16[kk * naxes[0] * naxes[1] +
                                            jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_INT32)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        data.image[IDout].array.SI32
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             data.image[IDin]
                             .array.SI32[kk * naxes[0] * naxes[1] +
                                            jj * naxes[0] + ii];
                    }
        }
        else if(datatype == _DATATYPE_INT64)
        {
            for(uint32_t ii = start_c[0]; ii < end_c[0]; ii++)
                for(uint32_t jj = start_c[1]; jj < end_c[1]; jj++)
                    for(uint32_t kk = start_c[2]; kk < end_c[2]; kk++)
                    {
                        data.image[IDout].array.SI64
                        [(kk - start[2]) * naxesout[0] * naxesout[1] +
                                         (jj - start[1]) * naxesout[0] + (ii - start[0])] =
                             data.image[IDin]
                             .array.SI64[kk * naxes[0] * naxes[1] +
                                            jj * naxes[0] + ii];
                    }
        }
        else
        {
            PRINT_ERROR("invalid data type");
            exit(0);
        }
    }

    free(naxesout);
    free(naxes);

    return IDout;
}

imageID arith_image_extract2D(const char *in_name,
                              const char *out_name,
                              long        size_x,
                              long        size_y,
                              long        xstart,
                              long        ystart)
{
    long        *start = NULL;
    long        *end   = NULL;
    int          naxis;
    imageID      ID;
    imageID      IDout;
    uint_fast8_t k;

    ID    = image_ID(in_name);
    naxis = data.image[ID].md[0].naxis;

    start = (long *) malloc(sizeof(long) * naxis);
    if(start == NULL)
    {
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    end = (long *) malloc(sizeof(long) * naxis);
    if(end == NULL)
    {
        PRINT_ERROR("malloc() error");
        exit(0);
    }

    for(k = 0; k < naxis; k++)
    {
        start[k] = 0;
        end[k]   = data.image[ID].md[0].size[k];
    }

    start[0] = xstart;
    start[1] = ystart;
    end[0]   = xstart + size_x;
    end[1]   = ystart + size_y;
    IDout    = arith_image_crop(in_name, out_name, start, end, naxis);

    free(start);
    free(end);

    return IDout;
}

imageID arith_image_extract3D(const char *in_name,
                              const char *out_name,
                              long        size_x,
                              long        size_y,
                              long        size_z,
                              long        xstart,
                              long        ystart,
                              long        zstart)
{
    imageID IDout;
    long   *start = NULL;
    long   *end   = NULL;

    start = (long *) malloc(sizeof(long) * 3);
    if(start == NULL)
    {
        PRINT_ERROR("malloc() error");
        printf("params: %s %s %ld %ld %ld %ld %ld %ld \n",
               in_name,
               out_name,
               size_x,
               size_y,
               size_z,
               xstart,
               ystart,
               zstart);
        exit(0);
    }

    end = (long *) malloc(sizeof(long) * 3);
    if(end == NULL)
    {
        PRINT_ERROR("malloc() error");
        printf("params: %s %s %ld %ld %ld %ld %ld %ld \n",
               in_name,
               out_name,
               size_x,
               size_y,
               size_z,
               xstart,
               ystart,
               zstart);
        exit(0);
    }

    start[0] = xstart;
    start[1] = ystart;
    start[2] = zstart;
    end[0]   = xstart + size_x;
    end[1]   = ystart + size_y;
    end[2]   = zstart + size_z;
    IDout    = arith_image_crop(in_name, out_name, start, end, 3);

    free(start);
    free(end);

    return IDout;
}
