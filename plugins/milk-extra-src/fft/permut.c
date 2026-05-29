/**
 * @file permut.c
 * @brief Permut image quadrants
 */

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "CLIcore.h"
#endif
#include "fps.h"

#include "COREMOD_memory/COREMOD_memory.h"

#define SWAPf(x, y)              \
    do                           \
    {                            \
        float swaptmp = x;       \
        x             = y;       \
        y             = swaptmp; \
    } while (0)

#define SWAPd(x, y)               \
    do                            \
    {                             \
        double swaptmp = x;       \
        x              = y;       \
        y              = swaptmp; \
    } while (0)

#define CSWAPcf(x, y)            \
    do                           \
    {                            \
        float swaptmp = x.re;    \
        x.re          = y.re;    \
        y.re          = swaptmp; \
        swaptmp       = x.im;    \
        x.im          = y.im;    \
        y.im          = swaptmp; \
    } while (0)

#define CSWAPcd(x, y)             \
    do                            \
    {                             \
        double swaptmp = (x.re);  \
        x.re           = y.re;    \
        y.re           = swaptmp; \
        swaptmp        = x.im;    \
        x.im           = y.im;    \
        y.im           = swaptmp; \
    } while (0)

// Forward declaration
int permut(const char *ID_name);

/* =========================================
 *  V2 PARAMS
 * ======================================= */

static char p_imname[FUNCTION_PARAMETER_STRMAXLEN] = "im1";

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "permut",
    .cmdkey           = "permut",
    .description      = "permut image quadrants",
    .description_long = "Permute image quadrants to shift the DC component to the center. Swaps "
                        "the four quadrants of a 2D image for standard FFT display convention."
};

#define FPS_PARAMS(X) X(".in_name", p_imname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "image")

FPS_V2_SECTION5(FPS_PARAMS)

static MILK_HOT errno_t compute_function()
{
    permut(p_imname);
    return RETURN_SUCCESS;
}

#ifndef FPS_STANDALONE
static errno_t __attribute__((unused)) CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_milkfft__permut()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


int permut(const char *ID_name)
{
    long    naxes0, naxes1, naxes2;
    imageID ID;
    long    xhalf, yhalf;
    long    ii, jj, kk;
    long    naxis;
    uint8_t datatype;
    int     OK = 0;

    //  printf("permut image %s ...", ID_name);
    // fflush(stdout);

    ID = image_ID(ID_name, dcimg, dcnimg);
    if (ID == -1)
    {
        PRINT_ERROR("Image \"%s\" not found", ID_name);
        return RETURN_FAILURE;
    }
    naxis = dcimg[ID].md[0].naxis;

    naxes0 = dcimg[ID].md[0].size[0];
    if (naxis > 1)
    {
        naxes1 = dcimg[ID].md[0].size[1];
    }
    if (naxis > 2)
    {
        naxes2 = dcimg[ID].md[0].size[2];
    }
    else
    {
        naxes2 = 1;
    }

    //  printf(" [%ld %ld %ld] ", naxes0, naxes1, naxes2);

    datatype = dcimg[ID].md[0].datatype;

    if (datatype == _DATATYPE_FLOAT)
    {
        if (naxis == 1)
        {
            OK    = 1;
            xhalf = (long) (naxes0 / 2);
            for (ii = 0; ii < xhalf; ii++)
            {
                SWAPf(dcimg[ID].array.F[ii], dcimg[ID].array.F[ii + xhalf]);
            }
        }
        if (naxis == 2)
        {
            OK    = 1;
            xhalf = (long) (naxes0 / 2);
            yhalf = (long) (naxes1 / 2);
            for (jj = 0; jj < yhalf; jj++)
            {
                for (ii = 0; ii < xhalf; ii++)
                {
                    SWAPf(dcimg[ID].array.F[jj * naxes0 + ii],
                          dcimg[ID].array.F[(jj + yhalf) * naxes0 + (ii + xhalf)]);
                }
            }
            for (jj = yhalf; jj < naxes1; jj++)
            {
                for (ii = 0; ii < xhalf; ii++)
                {
                    SWAPf(dcimg[ID].array.F[jj * naxes0 + ii],
                          dcimg[ID].array.F[(jj - yhalf) * naxes0 + (ii + xhalf)]);
                }
            }
        }
        if (naxis == 3)
        {
            OK    = 1;
            xhalf = (long) (naxes0 / 2);
            yhalf = (long) (naxes1 / 2);
            for (jj = 0; jj < yhalf; jj++)
            {
                for (ii = 0; ii < xhalf; ii++)
                {
                    for (kk = 0; kk < naxes2; kk++)
                    {
                        SWAPf(dcimg[ID].array.F[kk * naxes0 * naxes1 + jj * naxes0 + ii],
                              dcimg[ID].array.F[kk * naxes0 * naxes1 + (jj + yhalf) * naxes0 +
                                                (ii + xhalf)]);
                    }
                }
            }
            for (jj = yhalf; jj < naxes1; jj++)
            {
                for (ii = 0; ii < xhalf; ii++)
                {
                    for (kk = 0; kk < naxes2; kk++)
                    {
                        SWAPf(dcimg[ID].array.F[kk * naxes0 * naxes1 + jj * naxes0 + ii],
                              dcimg[ID].array.F[kk * naxes0 * naxes1 + (jj - yhalf) * naxes0 +
                                                (ii + xhalf)]);
                    }
                }
            }
        }
    }

    if (datatype == _DATATYPE_DOUBLE)
    {
        if (naxis == 1)
        {
            OK    = 1;
            xhalf = (long) (naxes0 / 2);
            for (ii = 0; ii < xhalf; ii++)
            {
                SWAPd(dcimg[ID].array.D[ii], dcimg[ID].array.D[ii + xhalf]);
            }
        }
        if (naxis == 2)
        {
            OK    = 1;
            xhalf = (long) (naxes0 / 2);
            yhalf = (long) (naxes1 / 2);
            for (jj = 0; jj < yhalf; jj++)
            {
                for (ii = 0; ii < xhalf; ii++)
                {
                    SWAPd(dcimg[ID].array.D[jj * naxes0 + ii],
                          dcimg[ID].array.D[(jj + yhalf) * naxes0 + (ii + xhalf)]);
                }
            }
            for (jj = yhalf; jj < naxes1; jj++)
            {
                for (ii = 0; ii < xhalf; ii++)
                {
                    SWAPd(dcimg[ID].array.D[jj * naxes0 + ii],
                          dcimg[ID].array.D[(jj - yhalf) * naxes0 + (ii + xhalf)]);
                }
            }
        }
        if (naxis == 3)
        {
            OK    = 1;
            xhalf = (long) (naxes0 / 2);
            yhalf = (long) (naxes1 / 2);
            for (jj = 0; jj < yhalf; jj++)
            {
                for (ii = 0; ii < xhalf; ii++)
                {
                    for (kk = 0; kk < naxes2; kk++)
                    {
                        SWAPd(dcimg[ID].array.D[kk * naxes0 * naxes1 + jj * naxes0 + ii],
                              dcimg[ID].array.D[kk * naxes0 * naxes1 + (jj + yhalf) * naxes0 +
                                                (ii + xhalf)]);
                    }
                }
            }
            for (jj = yhalf; jj < naxes1; jj++)
            {
                for (ii = 0; ii < xhalf; ii++)
                {
                    for (kk = 0; kk < naxes2; kk++)
                    {
                        SWAPd(dcimg[ID].array.D[kk * naxes0 * naxes1 + jj * naxes0 + ii],
                              dcimg[ID].array.D[kk * naxes0 * naxes1 + (jj - yhalf) * naxes0 +
                                                (ii + xhalf)]);
                    }
                }
            }
        }
    }

    if (datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        if (naxis == 1)
        {
            OK    = 1;
            xhalf = (long) (naxes0 / 2);
            for (ii = 0; ii < xhalf; ii++)
            {
                CSWAPcf(dcimg[ID].array.CF[ii], dcimg[ID].array.CF[ii + xhalf]);
            }
        }
        if (naxis == 2)
        {
            OK    = 1;
            xhalf = (long) (naxes0 / 2);
            yhalf = (long) (naxes1 / 2);
            for (jj = 0; jj < yhalf; jj++)
            {
                for (ii = 0; ii < xhalf; ii++)
                {
                    CSWAPcf(dcimg[ID].array.CF[jj * naxes0 + ii],
                            dcimg[ID].array.CF[(jj + yhalf) * naxes0 + (ii + xhalf)]);
                }
            }
            for (jj = yhalf; jj < naxes1; jj++)
            {
                for (ii = 0; ii < xhalf; ii++)
                {
                    CSWAPcf(dcimg[ID].array.CF[jj * naxes0 + ii],
                            dcimg[ID].array.CF[(jj - yhalf) * naxes0 + (ii + xhalf)]);
                }
            }
        }
        if (naxis == 3)
        {
            OK    = 1;
            xhalf = (long) (naxes0 / 2);
            yhalf = (long) (naxes1 / 2);
            for (kk = 0; kk < naxes2; kk++)
            {
                for (jj = 0; jj < yhalf; jj++)
                {
                    for (ii = 0; ii < xhalf; ii++)
                    {
                        CSWAPcf(dcimg[ID].array.CF[kk * naxes0 * naxes1 + jj * naxes0 + ii],
                                dcimg[ID].array.CF[kk * naxes0 * naxes1 + (jj + yhalf) * naxes0 +
                                                   (ii + xhalf)]);
                    }
                }
            }
            printf(" - ");
            fflush(stdout);

            for (kk = 0; kk < naxes2; kk++)
            {
                for (jj = yhalf; jj < naxes1; jj++)
                {
                    for (ii = 0; ii < xhalf; ii++)
                    {
                        CSWAPcf(dcimg[ID].array.CF[kk * naxes0 * naxes1 + jj * naxes0 + ii],
                                dcimg[ID].array.CF[kk * naxes0 * naxes1 + (jj - yhalf) * naxes0 +
                                                   (ii + xhalf)]);
                    }
                }
            }
        }
    }

    if (datatype == _DATATYPE_COMPLEX_DOUBLE)
    {
        if (naxis == 1)
        {
            OK    = 1;
            xhalf = (long) (naxes0 / 2);
            for (ii = 0; ii < xhalf; ii++)
            {
                CSWAPcd(dcimg[ID].array.CD[ii], dcimg[ID].array.CD[ii + xhalf]);
            }
        }
        if (naxis == 2)
        {
            OK    = 1;
            xhalf = (long) (naxes0 / 2);
            yhalf = (long) (naxes1 / 2);
            for (jj = 0; jj < yhalf; jj++)
            {
                for (ii = 0; ii < xhalf; ii++)
                {
                    CSWAPcd(dcimg[ID].array.CD[jj * naxes0 + ii],
                            dcimg[ID].array.CD[(jj + yhalf) * naxes0 + (ii + xhalf)]);
                }
            }
            for (jj = yhalf; jj < naxes1; jj++)
            {
                for (ii = 0; ii < xhalf; ii++)
                {
                    CSWAPcd(dcimg[ID].array.CD[jj * naxes0 + ii],
                            dcimg[ID].array.CD[(jj - yhalf) * naxes0 + (ii + xhalf)]);
                }
            }
        }
        if (naxis == 3)
        {
            OK    = 1;
            xhalf = (long) (naxes0 / 2);
            yhalf = (long) (naxes1 / 2);
            for (kk = 0; kk < naxes2; kk++)
            {
                for (jj = 0; jj < yhalf; jj++)
                {
                    for (ii = 0; ii < xhalf; ii++)
                    {
                        CSWAPcd(dcimg[ID].array.CD[kk * naxes0 * naxes1 + jj * naxes0 + ii],
                                dcimg[ID].array.CD[kk * naxes0 * naxes1 + (jj + yhalf) * naxes0 +
                                                   (ii + xhalf)]);
                    }
                }
            }
            printf(" - ");
            fflush(stdout);

            for (kk = 0; kk < naxes2; kk++)
            {
                for (jj = yhalf; jj < naxes1; jj++)
                {
                    for (ii = 0; ii < xhalf; ii++)
                    {
                        CSWAPcd(dcimg[ID].array.CD[kk * naxes0 * naxes1 + jj * naxes0 + ii],
                                dcimg[ID].array.CD[kk * naxes0 * naxes1 + (jj - yhalf) * naxes0 +
                                                   (ii + xhalf)]);
                    }
                }
            }
        }
    }

    if (OK == 0)
    {
        printf("Error : data format not supported by permut\n");
    }

    //  printf(" done\n");
    // fflush(stdout);

    return (0);
}
