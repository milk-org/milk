/** @file image_add.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID basic_add(const char *__restrict ID_name1,
                  const char *__restrict ID_name2,
                  const char *__restrict ID_name_out,
                  long off1,
                  long off2);

imageID basic_add3D(const char *__restrict ID_name1,
                    const char *__restrict ID_name2,
                    const char *__restrict ID_name_out,
                    long off1,
                    long off2,
                    long off3);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t image_basic_add_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 4) + CLI_checkarg(3, 3) +
            CLI_checkarg(4, 2) + CLI_checkarg(5, 2) ==
            0)
    {
        basic_add(data.cmdargtoken[1].val.string,
                  data.cmdargtoken[2].val.string,
                  data.cmdargtoken[3].val.string,
                  data.cmdargtoken[4].val.numl,
                  data.cmdargtoken[5].val.numl);
        return CLICMD_SUCCESS;
    }
    else
    {
        return CLICMD_INVALID_ARG;
    }
}

static errno_t image_basic_add3D_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 4) + CLI_checkarg(3, 3) +
            CLI_checkarg(4, 2) + CLI_checkarg(5, 2) + CLI_checkarg(6, 2) ==
            0)
    {
        basic_add3D(data.cmdargtoken[1].val.string,
                    data.cmdargtoken[2].val.string,
                    data.cmdargtoken[3].val.string,
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

// ==========================================
// Register CLI command(s)
// ==========================================

errno_t __attribute__((cold)) image_add_addCLIcmd()
{

    RegisterCLIcommand(
        "addim",
        __FILE__,
        image_basic_add_cli,
        "add two 2D images of different size",
        "<im1> <im2> <outim> <offsetx> <offsety>",
        "addim im1 im2 outim 23 201",
        "long basic_add(const char *ID_name1, const char *ID_name2, const char "
        "*ID_name_out, long off1, long off2)");

    RegisterCLIcommand("addim3D",
                       __FILE__,
                       image_basic_add3D_cli,
                       "add two 3D images of different size",
                       "<im1> <im2> <outim> <offsetx> <offsety> <offsetz>",
                       "addim3D im1 im2 outim 23 201 0",
                       "long basic_add3D(const char *ID_name1, const char "
                       "*ID_name2, const char *ID_name_out, long "
                       "off1, long off2, long off3)");

    return RETURN_SUCCESS;
}

imageID basic_add(const char *__restrict ID_name1,
                  const char *__restrict ID_name2,
                  const char *__restrict ID_name_out,
                  long off1,
                  long off2)
{
    imageID ID1, ID2; /* ID for the 2 images added */
    imageID ID_out;   /* ID for the output image */
    long    ii, jj;
    long    naxes1[2], naxes2[2], naxes[2];
    long    xmin, ymin, xmax, ymax; /* extrema in the ID1 coordinates */
    uint8_t datatype1, datatype2, datatype;
    int     datatypeOK;

    ID1       = image_ID(ID_name1);
    ID2       = image_ID(ID_name2);
    naxes1[0] = data.image[ID1].md[0].size[0];
    naxes1[1] = data.image[ID1].md[0].size[1];
    naxes2[0] = data.image[ID2].md[0].size[0];
    naxes2[1] = data.image[ID2].md[0].size[1];

    datatype1 = data.image[ID1].md[0].datatype;
    datatype2 = data.image[ID2].md[0].datatype;

    datatypeOK = 0;

    if((datatype1 == _DATATYPE_FLOAT) && (datatype2 == _DATATYPE_FLOAT))
    {
        datatype   = _DATATYPE_FLOAT;
        datatypeOK = 1;
    }
    if((datatype1 == _DATATYPE_DOUBLE) && (datatype2 == _DATATYPE_DOUBLE))
    {
        datatype   = _DATATYPE_DOUBLE;
        datatypeOK = 1;
    }

    if(datatypeOK == 0)
    {
        printf("ERROR in basic_add: data type combination not supported\n");
        exit(EXIT_FAILURE);
    }

    /*  if(data.quiet==0)*/
    /* printf("add called with %s ( %ld x %ld ) %s ( %ld x %ld ) and offset ( %ld x %ld )\n",ID_name1,naxes1[0],naxes1[1],ID_name2,naxes2[0],naxes2[1],off1,off2);*/
    xmin = 0;
    if(off1 < 0)
    {
        xmin = off1;
    }
    ymin = 0;
    if(off2 < 0)
    {
        ymin = off2;
    }
    xmax = naxes1[0];
    if((naxes2[0] + off1) > naxes1[0])
    {
        xmax = (naxes2[0] + off1);
    }
    ymax = naxes1[1];
    if((naxes2[1] + off2) > naxes1[1])
    {
        ymax = (naxes2[1] + off2);
    }

    if(datatype == _DATATYPE_FLOAT)
    {
        create_2Dimage_ID(ID_name_out, (xmax - xmin), (ymax - ymin), &ID_out);
        naxes[0] = data.image[ID_out].md[0].size[0];
        naxes[1] = data.image[ID_out].md[0].size[1];

        for(jj = 0; jj < naxes[1]; jj++)
            for(ii = 0; ii < naxes[0]; ii++)
            {
                {
                    data.image[ID_out].array.F[jj * naxes[0] + ii] = 0;
                    /* if pixel is in ID1 */
                    if(((ii + xmin) >= 0) && ((ii + xmin) < naxes1[0]))
                        if(((jj + ymin) >= 0) && ((jj + ymin) < naxes1[1]))
                        {
                            data.image[ID_out].array.F[jj * naxes[0] + ii] +=
                                data.image[ID1]
                                .array
                                .F[(jj + ymin) * naxes1[0] + (ii + xmin)];
                        }
                    /* if pixel is in ID2 */
                    if(((ii + xmin - off1) >= 0) &&
                            ((ii + xmin - off1) < naxes2[0]))
                        if(((jj + ymin - off2) >= 0) &&
                                ((jj + ymin - off2) < naxes2[1]))
                        {
                            data.image[ID_out].array.F[jj * naxes[0] + ii] +=
                                data.image[ID2]
                                .array.F[(jj + ymin - off2) * naxes2[0] +
                                                            (ii + xmin - off1)];
                        }
                }
            }
    }

    if(datatype == _DATATYPE_DOUBLE)
    {
        create_2Dimage_ID_double(ID_name_out,
                                 (xmax - xmin),
                                 (ymax - ymin),
                                 &ID_out);
        naxes[0] = data.image[ID_out].md[0].size[0];
        naxes[1] = data.image[ID_out].md[0].size[1];

        for(jj = 0; jj < naxes[1]; jj++)
            for(ii = 0; ii < naxes[0]; ii++)
            {
                {
                    data.image[ID_out].array.D[jj * naxes[0] + ii] = 0;
                    /* if pixel is in ID1 */
                    if(((ii + xmin) >= 0) && ((ii + xmin) < naxes1[0]))
                        if(((jj + ymin) >= 0) && ((jj + ymin) < naxes1[1]))
                        {
                            data.image[ID_out].array.D[jj * naxes[0] + ii] +=
                                data.image[ID1]
                                .array
                                .D[(jj + ymin) * naxes1[0] + (ii + xmin)];
                        }
                    /* if pixel is in ID2 */
                    if(((ii + xmin - off1) >= 0) &&
                            ((ii + xmin - off1) < naxes2[0]))
                        if(((jj + ymin - off2) >= 0) &&
                                ((jj + ymin - off2) < naxes2[1]))
                        {
                            data.image[ID_out].array.D[jj * naxes[0] + ii] +=
                                data.image[ID2]
                                .array.D[(jj + ymin - off2) * naxes2[0] +
                                                            (ii + xmin - off1)];
                        }
                }
            }
    }

    return (ID_out);
}

imageID basic_add3D(const char *__restrict ID_name1,
                    const char *__restrict ID_name2,
                    const char *__restrict ID_name_out,
                    long off1,
                    long off2,
                    long off3)
{
    imageID  ID1, ID2; /* ID for the 2 images added */
    imageID  ID_out;   /* ID for the output image */
    uint32_t naxes1[3], naxes2[3], naxes[3];
    long     xmin, ymin, zmin, xmax, ymax,
             zmax; /* extrema in the ID1 coordinates */
    uint8_t datatype1, datatype2, datatype;
    int     datatypeOK;

    ID1       = image_ID(ID_name1);
    ID2       = image_ID(ID_name2);
    naxes1[0] = data.image[ID1].md[0].size[0];
    naxes1[1] = data.image[ID1].md[0].size[1];
    naxes1[2] = data.image[ID1].md[0].size[2];

    naxes2[0] = data.image[ID2].md[0].size[0];
    naxes2[1] = data.image[ID2].md[0].size[1];
    naxes2[2] = data.image[ID2].md[0].size[2];

    datatype1 = data.image[ID1].md[0].datatype;
    datatype2 = data.image[ID2].md[0].datatype;

    datatypeOK = 0;

    if((datatype1 == _DATATYPE_FLOAT) && (datatype2 == _DATATYPE_FLOAT))
    {
        datatype   = _DATATYPE_FLOAT;
        datatypeOK = 1;
    }
    if((datatype1 == _DATATYPE_DOUBLE) && (datatype2 == _DATATYPE_DOUBLE))
    {
        datatype   = _DATATYPE_DOUBLE;
        datatypeOK = 1;
    }

    if(datatypeOK == 0)
    {
        printf("ERROR in basic_add: data type combination not supported\n");
        exit(0);
    }

    /*  if(data.quiet==0)*/
    /* printf("add called with %s ( %ld x %ld ) %s ( %ld x %ld ) and offset ( %ld x %ld )\n",ID_name1,naxes1[0],naxes1[1],ID_name2,naxes2[0],naxes2[1],off1,off2);*/
    xmin = 0;
    if(off1 < 0)
    {
        xmin = off1;
    }

    ymin = 0;
    if(off2 < 0)
    {
        ymin = off2;
    }

    zmin = 0;
    if(off3 < 0)
    {
        zmin = off3;
    }

    xmax = naxes1[0];
    if((naxes2[0] + off1) > naxes1[0])
    {
        xmax = (naxes2[0] + off1);
    }

    ymax = naxes1[1];
    if((naxes2[1] + off2) > naxes1[1])
    {
        ymax = (naxes2[1] + off2);
    }

    zmax = naxes1[2];
    if((naxes2[2] + off3) > naxes1[2])
    {
        zmax = (naxes2[2] + off3);
    }

    if(datatype == _DATATYPE_FLOAT)
    {
        create_3Dimage_ID(ID_name_out,
                          (xmax - xmin),
                          (ymax - ymin),
                          (zmax - zmin),
                          &ID_out);
        naxes[0] = data.image[ID_out].md[0].size[0];
        naxes[1] = data.image[ID_out].md[0].size[1];
        naxes[2] = data.image[ID_out].md[0].size[2];

        for(uint32_t kk = 0; kk < naxes[2]; kk++)
            for(uint32_t jj = 0; jj < naxes[1]; jj++)
                for(uint32_t ii = 0; ii < naxes[0]; ii++)
                {
                    {
                        data.image[ID_out].array.F[kk * naxes[1] * naxes[0] +
                                                   jj * naxes[0] + ii] = 0;
                        /* if pixel is in ID1 */

                        if(((ii + xmin) >= 0) && ((ii + xmin) < naxes1[0]))
                            if(((jj + ymin) >= 0) && ((jj + ymin) < naxes1[1]))
                                if(((kk + zmin) >= 0) &&
                                        ((kk + zmin) < naxes1[2]))
                                {
                                    data.image[ID_out]
                                    .array.F[kk * naxes[1] * naxes[0] +
                                                jj * naxes[0] + ii] +=
                                                 data.image[ID1]
                                                 .array.F[(kk + zmin) * naxes1[1] *
                                                                      naxes1[0] +
                                                                      (jj + ymin) * naxes1[0] +
                                                                      (ii + xmin)];
                                }
                        /* if pixel is in ID2 */
                        if(((ii + xmin - off1) >= 0) &&
                                ((ii + xmin - off1) < naxes2[0]))
                            if(((jj + ymin - off2) >= 0) &&
                                    ((jj + ymin - off2) < naxes2[1]))
                                if(((kk + zmin - off3) >= 0) &&
                                        ((kk + zmin - off3) < naxes2[2]))
                                {
                                    data.image[ID_out]
                                    .array.F[kk * naxes[1] * naxes[0] +
                                                jj * naxes[0] + ii] +=
                                                 data.image[ID2]
                                                 .array
                                                 .F[(kk + zmin - off3) * naxes2[1] *
                                                                       naxes2[0] +
                                                                       (jj + ymin - off2) * naxes2[0] +
                                                                       (ii + xmin - off1)];
                                }
                    }
                }
    }

    if(datatype == _DATATYPE_DOUBLE)
    {
        create_3Dimage_ID_double(ID_name_out,
                                 (xmax - xmin),
                                 (ymax - ymin),
                                 (zmax - zmin),
                                 &ID_out);
        naxes[0] = data.image[ID_out].md[0].size[0];
        naxes[1] = data.image[ID_out].md[0].size[1];
        naxes[2] = data.image[ID_out].md[0].size[2];

        for(uint32_t kk = 0; kk < naxes[2]; kk++)
            for(uint32_t jj = 0; jj < naxes[1]; jj++)
                for(uint32_t ii = 0; ii < naxes[0]; ii++)
                {
                    {
                        data.image[ID_out].array.D[kk * naxes[1] * naxes[0] +
                                                   jj * naxes[0] + ii] = 0;
                        /* if pixel is in ID1 */
                        if(((ii + xmin) >= 0) && ((ii + xmin) < naxes1[0]))
                            if(((jj + ymin) >= 0) && ((jj + ymin) < naxes1[1]))
                                if(((kk + zmin) >= 0) &&
                                        ((kk + zmin) < naxes1[2]))
                                {
                                    data.image[ID_out]
                                    .array.D[kk * naxes[1] * naxes[0] +
                                                jj * naxes[0] + ii] +=
                                                 data.image[ID1]
                                                 .array.D[(kk + zmin) * naxes1[1] *
                                                                      naxes1[0] +
                                                                      (jj + ymin) * naxes1[0] +
                                                                      (ii + xmin)];
                                }
                        /* if pixel is in ID2 */
                        if(((ii + xmin - off1) >= 0) &&
                                ((ii + xmin - off1) < naxes2[0]))
                            if(((jj + ymin - off2) >= 0) &&
                                    ((jj + ymin - off2) < naxes2[1]))
                                if(((kk + zmin - off3) >= 0) &&
                                        ((kk + zmin - off3) < naxes2[2]))
                                {
                                    data.image[ID_out]
                                    .array.D[kk * naxes[1] * naxes[0] +
                                                jj * naxes[0] + ii] +=
                                                 data.image[ID2]
                                                 .array
                                                 .D[(kk + zmin - off3) * naxes2[1] *
                                                                       naxes2[0] +
                                                                       (jj + ymin - off2) * naxes2[0] +
                                                                       (ii + xmin - off1)];
                                }
                    }
                }
    }

    return (ID_out);
}
