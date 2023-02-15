/** @file imresize.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

// ==========================================
// Forward declaration(s)
// ==========================================

long basic_resizeim(const char *imname_in,
                    const char *imname_out,
                    long        xsizeout,
                    long        ysizeout);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t image_basic_resize_cli()
{
    if(CLI_checkarg(1, 4) + CLI_checkarg(2, 3) + CLI_checkarg(3, 2) +
            CLI_checkarg(4, 2) ==
            0)
    {
        basic_resizeim(data.cmdargtoken[1].val.string,
                       data.cmdargtoken[2].val.string,
                       data.cmdargtoken[3].val.numl,
                       data.cmdargtoken[4].val.numl);
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

errno_t imresize_addCLIcmd()
{

    RegisterCLIcommand("resizeim",
                       __FILE__,
                       image_basic_resize_cli,
                       "resize 2D image",
                       "<image in> <output image> <new x size> <new y size>",
                       "resizeim im1 im2 230 200",
                       "long basic_resizeim(const char *imname_in, const char "
                       "*imname_out, long xsizeout, long ysizeout)");

    return RETURN_SUCCESS;
}

/* ----------------------------------------------------------------------
 *
 * resize image using bilinear interpolation
 *
 *
 * ---------------------------------------------------------------------- */

long basic_resizeim(const char *imname_in,
                    const char *imname_out,
                    long        xsizeout,
                    long        ysizeout)
{
    imageID  ID, IDout;
    long     naxis = 2;
    uint32_t naxes[2];
    uint32_t naxesout[2];
    float    xf, yf, xf1, yf1, uf, tf, v00f, v01f, v10f, v11f;
    double   xd, yd, xd1, yd1, ud, td, v00d, v01d, v10d, v11d;
    uint8_t  datatype;
    long     ii, jj, ii1, jj1;

    ID          = image_ID(imname_in);
    datatype    = data.image[ID].md[0].datatype;
    naxes[0]    = data.image[ID].md[0].size[0];
    naxes[1]    = data.image[ID].md[0].size[1];
    naxesout[0] = xsizeout;
    naxesout[1] = ysizeout;

    if(datatype == _DATATYPE_FLOAT)
    {
        create_image_ID(imname_out, naxis, naxesout, datatype, 0, 0, 0, &IDout);
        for(ii = 0; ii < naxesout[0]; ii++)
            for(jj = 0; jj < naxesout[1]; jj++)
            {
                xf  = (float)(1.0 * ii / naxesout[0]);
                yf  = (float)(1.0 * jj / naxesout[1]);
                xf1 = xf * (float) naxes[0];
                yf1 = yf * (float) naxes[1];
                ii1 = (long) xf1;
                jj1 = (long) yf1;
                uf  = xf1 - (float) ii1;
                tf  = yf1 - (float) jj1;
                if((ii1 > -1) && (ii1 + 1 < naxes[0]) && (jj1 > -1) &&
                        (jj1 + 1 < naxes[1]))
                {
                    v00f = data.image[ID].array.F[jj1 * naxes[0] + ii1];
                    v01f = data.image[ID].array.F[(jj1 + 1) * naxes[0] + ii1];
                    v10f = data.image[ID].array.F[jj1 * naxes[0] + ii1 + 1];
                    v11f =
                        data.image[ID].array.F[(jj1 + 1) * naxes[0] + ii1 + 1];
                    data.image[IDout].array.F[jj * naxesout[0] + ii] =
                        (float)(v00f * (1.0 - uf) * (1.0 - tf) +
                                v10f * uf * (1.0 - tf) +
                                v01f * (1.0 - uf) * tf + v11f * uf * tf);
                }
            }
    }
    else if(datatype == _DATATYPE_DOUBLE)
    {
        create_image_ID(imname_out, naxis, naxesout, datatype, 0, 0, 0, &IDout);
        for(ii = 0; ii < naxesout[0] - 1; ii++)
            for(jj = 0; jj < naxesout[1] - 1; jj++)
            {
                xd   = 1.0 * ii / naxesout[0];
                yd   = 1.0 * jj / naxesout[1];
                xd1  = xd * naxes[0];
                yd1  = yd * naxes[1];
                ii1  = (long) xd1;
                jj1  = (long) yd1;
                ud   = xd1 - (float) ii1;
                td   = yd1 - (float) jj1;
                v00d = data.image[ID].array.D[jj1 * naxes[0] + ii1];
                v01d = data.image[ID].array.D[(jj1 + 1) * naxes[0] + ii1];
                v10d = data.image[ID].array.D[jj1 * naxes[0] + ii1 + 1];
                v11d = data.image[ID].array.D[(jj1 + 1) * naxes[0] + ii1 + 1];
                data.image[IDout].array.D[jj * naxesout[0] + ii] =
                    (double)(v00d * (1.0 - ud) * (1.0 - td) +
                             v10d * ud * (1.0 - td) + v01d * (1.0 - ud) * td +
                             v11d * ud * td);
            }
    }
    else
    {
        PRINT_ERROR("Wrong image type(s)\n");
        exit(0);
    }

    return (0);
}
