/** @file imstretch.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"

imageID basic_stretch(const char *__restrict name_in,
                      const char *__restrict name_out,
                      float coeff,
                      long  Xcenter,
                      long  Ycenter)
{
    uint32_t naxes[2];
    imageID  IDin;
    imageID  IDout;
    long     i, j;

    IDin     = image_ID(name_in);
    naxes[0] = data.image[IDin].md[0].size[0];
    naxes[1] = data.image[IDin].md[0].size[1];

    create_2Dimage_ID(name_out, naxes[0], naxes[1], &IDout);

    for(uint32_t ii = 0; ii < naxes[0]; ii++)
        for(uint32_t jj = 0; jj < naxes[0]; jj++)
        {
            i = Xcenter + (long)(1.0 * (ii - Xcenter) * coeff);
            j = Ycenter + (long)(1.0 * (jj - Ycenter) * coeff);
            if((i < naxes[0]) && (j < naxes[1]) && (i > -1) && (j > -1))
            {
                data.image[IDout].array.F[jj * naxes[0] + ii] +=
                    data.image[IDin].array.F[j * naxes[0] + i] / coeff / coeff;
            }
        }

    arith_image_cstmult_inplace(name_out,
                                arith_image_total(name_in) /
                                arith_image_total(name_out));

    return IDout;
}

imageID basic_stretch_range(const char *__restrict name_in,
                            const char *__restrict name_out,
                            float coeff1,
                            float coeff2,
                            long  Xcenter,
                            long  Ycenter,
                            long  NBstep,
                            float ApoCoeff)
{
    DEBUG_TRACE_FSTART();

    // ApoCoeff should be between 0 and 1
    uint32_t naxes[2];
    imageID  IDin, IDout;
    long     i, j;
    float    coeff;
    long     step;
    float    mcoeff;
    float    x, y;
    float    eps = 1.0e-5;
    float    u, t, tmp;

    IDin     = image_ID(name_in);
    naxes[0] = data.image[IDin].md[0].size[0];
    naxes[1] = data.image[IDin].md[0].size[1];

    FUNC_CHECK_RETURN(create_2Dimage_ID(name_out, naxes[0], naxes[1], &IDout));

    for(step = 0; step < NBstep; step++)
    {
        fprintf(stdout, ".");
        fflush(stdout);
        coeff = coeff1 + (coeff2 - coeff1) * (1.0 * step / (NBstep - 1));
        x     = (coeff - (coeff1 + coeff2) / 2.0) / ((coeff2 - coeff1) / 2.0);
        // x goes from -1 to 1
        if(ApoCoeff > eps)
        {
            mcoeff =
                pow((1.0 - pow((fabs(x) - (1.0 - ApoCoeff)) / ApoCoeff, 2.0)),
                    4.0);
        }
        else
        {
            mcoeff = 1.0;
        }

        if((1.0 - x * x) < eps)
        {
            mcoeff = 0.0;
        }
        if(fabs(x) < ApoCoeff)
        {
            mcoeff = 1.0;
        }
        //      fprintf(stdout,"(%f %f %f %f %f)",coeff,coeff1,coeff2,x,mcoeff);

        for(uint32_t ii = 0; ii < naxes[0]; ii++)
            for(uint32_t jj = 0; jj < naxes[1]; jj++)
            {
                x = (1.0 * (ii - Xcenter) * coeff) + Xcenter;
                y = (1.0 * (jj - Ycenter) * coeff) + Ycenter;
                i = (long) x;
                j = (long) y;
                u = x - i;
                t = y - j;
                if((i < naxes[0] - 1) && (j < naxes[1] - 1) && (i > -1) &&
                        (j > -1))
                {
                    tmp = (1.0 - u) * (1.0 - t) *
                          data.image[IDin].array.F[j * naxes[0] + i];
                    tmp += (1.0 - u) * t *
                           data.image[IDin].array.F[(j + 1) * naxes[0] + i];
                    tmp += u * (1.0 - t) *
                           data.image[IDin].array.F[j * naxes[0] + i + 1];
                    tmp += u * t *
                           data.image[IDin].array.F[(j + 1) * naxes[0] + i + 1];
                    data.image[IDout].array.F[jj * naxes[0] + ii] +=
                        mcoeff * tmp / coeff / coeff;
                }
            }
    }

    fprintf(stdout, "\n");
    arith_image_cstmult_inplace(name_out,
                                arith_image_total(name_in) /
                                arith_image_total(name_out));

    DEBUG_TRACE_FEXIT();
    return IDout;
}

imageID basic_stretchc(const char *__restrict name_in,
                       const char *__restrict name_out,
                       float coeff)
{
    DEBUG_TRACE_FSTART();

    uint32_t naxes[2];
    imageID  IDin;
    imageID  IDout;
    long     i, j;
    long     Xcenter, Ycenter;

    IDin     = image_ID(name_in);
    naxes[0] = data.image[IDin].md[0].size[0];
    naxes[1] = data.image[IDin].md[0].size[1];
    Xcenter  = naxes[0] / 2;
    Ycenter  = naxes[1] / 2;

    FUNC_CHECK_RETURN(create_2Dimage_ID(name_out, naxes[0], naxes[1], &IDout));

    for(uint32_t ii = 0; ii < naxes[0]; ii++)
        for(uint32_t jj = 0; jj < naxes[0]; jj++)
        {
            i = Xcenter + (long)(1.0 * (ii - Xcenter) * coeff);
            j = Ycenter + (long)(1.0 * (jj - Ycenter) * coeff);
            if((i < naxes[0]) && (j < naxes[1]) && (i > -1) && (j > -1))
            {
                data.image[IDout].array.F[jj * naxes[0] + ii] +=
                    data.image[IDin].array.F[j * naxes[0] + i] / coeff / coeff;
            }
        }

    /*  basic_mult(name_out,arith_image_total(name_in)/arith_image_total(name_out));*/

    DEBUG_TRACE_FEXIT();
    return IDout;
}
