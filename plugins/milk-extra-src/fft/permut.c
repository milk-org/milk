/** @file permut.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#define SWAPf(x, y)                                                            \
    do                                                                         \
    {                                                                          \
        float swaptmp = x;                                                     \
        x             = y;                                                     \
        y             = swaptmp;                                               \
    } while (0)

#define SWAPd(x, y)                                                            \
    do                                                                         \
    {                                                                          \
        double swaptmp = x;                                                    \
        x              = y;                                                    \
        y              = swaptmp;                                              \
    } while (0)

#define CSWAPcf(x, y)                                                          \
    do                                                                         \
    {                                                                          \
        float swaptmp = x.re;                                                  \
        x.re          = y.re;                                                  \
        y.re          = swaptmp;                                               \
        swaptmp       = x.im;                                                  \
        x.im          = y.im;                                                  \
        y.im          = swaptmp;                                               \
    } while (0)

#define CSWAPcd(x, y)                                                          \
    do                                                                         \
    {                                                                          \
        double swaptmp = (x.re);                                               \
        x.re           = y.re;                                                 \
        y.re           = swaptmp;                                              \
        swaptmp        = x.im;                                                 \
        x.im           = y.im;                                                 \
        y.im           = swaptmp;                                              \
    } while (0)

// ==========================================
// Forward declaration(s)
// ==========================================

int permut(const char *ID_name);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t fft_permut_cli()
{
    if(CLI_checkarg(1, CLIARG_IMG) == 0)
    {
        permut(data.cmdargtoken[1].val.string);
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

errno_t permut_addCLIcmd()
{
    RegisterCLIcommand("permut",
                       __FILE__,
                       fft_permut_cli,
                       "permut image quadrants",
                       "<image>",
                       "permut im1",
                       "int permut(const char *ID_name)");

    return RETURN_SUCCESS;
}

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

    ID    = image_ID(ID_name);
    naxis = data.image[ID].md[0].naxis;

    naxes0 = data.image[ID].md[0].size[0];
    if(naxis > 1)
    {
        naxes1 = data.image[ID].md[0].size[1];
    }
    if(naxis > 2)
    {
        naxes2 = data.image[ID].md[0].size[2];
    }
    else
    {
        naxes2 = 1;
    }

    //  printf(" [%ld %ld %ld] ", naxes0, naxes1, naxes2);

    datatype = data.image[ID].md[0].datatype;

    if(datatype == _DATATYPE_FLOAT)
    {
        if(naxis == 1)
        {
            OK    = 1;
            xhalf = (long)(naxes0 / 2);
            for(ii = 0; ii < xhalf; ii++)
            {
                SWAPf(data.image[ID].array.F[ii],
                      data.image[ID].array.F[ii + xhalf]);
            }
        }
        if(naxis == 2)
        {
            OK    = 1;
            xhalf = (long)(naxes0 / 2);
            yhalf = (long)(naxes1 / 2);
            for(jj = 0; jj < yhalf; jj++)
                for(ii = 0; ii < xhalf; ii++)
                {
                    SWAPf(data.image[ID].array.F[jj * naxes0 + ii],
                          data.image[ID]
                          .array.F[(jj + yhalf) * naxes0 + (ii + xhalf)]);
                }
            for(jj = yhalf; jj < naxes1; jj++)
                for(ii = 0; ii < xhalf; ii++)
                {
                    SWAPf(data.image[ID].array.F[jj * naxes0 + ii],
                          data.image[ID]
                          .array.F[(jj - yhalf) * naxes0 + (ii + xhalf)]);
                }
        }
        if(naxis == 3)
        {
            OK    = 1;
            xhalf = (long)(naxes0 / 2);
            yhalf = (long)(naxes1 / 2);
            for(jj = 0; jj < yhalf; jj++)
                for(ii = 0; ii < xhalf; ii++)
                {
                    for(kk = 0; kk < naxes2; kk++)
                    {
                        SWAPf(data.image[ID].array.F[kk * naxes0 * naxes1 +
                                                     jj * naxes0 + ii],
                              data.image[ID].array.F[kk * naxes0 * naxes1 +
                                                     (jj + yhalf) * naxes0 +
                                                     (ii + xhalf)]);
                    }
                }
            for(jj = yhalf; jj < naxes1; jj++)
                for(ii = 0; ii < xhalf; ii++)
                {
                    for(kk = 0; kk < naxes2; kk++)
                    {
                        SWAPf(data.image[ID].array.F[kk * naxes0 * naxes1 +
                                                     jj * naxes0 + ii],
                              data.image[ID].array.F[kk * naxes0 * naxes1 +
                                                     (jj - yhalf) * naxes0 +
                                                     (ii + xhalf)]);
                    }
                }
        }
    }

    if(datatype == _DATATYPE_DOUBLE)
    {
        if(naxis == 1)
        {
            OK    = 1;
            xhalf = (long)(naxes0 / 2);
            for(ii = 0; ii < xhalf; ii++)
            {
                SWAPd(data.image[ID].array.D[ii],
                      data.image[ID].array.D[ii + xhalf]);
            }
        }
        if(naxis == 2)
        {
            OK    = 1;
            xhalf = (long)(naxes0 / 2);
            yhalf = (long)(naxes1 / 2);
            for(jj = 0; jj < yhalf; jj++)
                for(ii = 0; ii < xhalf; ii++)
                {
                    SWAPd(data.image[ID].array.D[jj * naxes0 + ii],
                          data.image[ID]
                          .array.D[(jj + yhalf) * naxes0 + (ii + xhalf)]);
                }
            for(jj = yhalf; jj < naxes1; jj++)
                for(ii = 0; ii < xhalf; ii++)
                {
                    SWAPd(data.image[ID].array.D[jj * naxes0 + ii],
                          data.image[ID]
                          .array.D[(jj - yhalf) * naxes0 + (ii + xhalf)]);
                }
        }
        if(naxis == 3)
        {
            OK    = 1;
            xhalf = (long)(naxes0 / 2);
            yhalf = (long)(naxes1 / 2);
            for(jj = 0; jj < yhalf; jj++)
                for(ii = 0; ii < xhalf; ii++)
                {
                    for(kk = 0; kk < naxes2; kk++)
                    {
                        SWAPd(data.image[ID].array.D[kk * naxes0 * naxes1 +
                                                     jj * naxes0 + ii],
                              data.image[ID].array.D[kk * naxes0 * naxes1 +
                                                     (jj + yhalf) * naxes0 +
                                                     (ii + xhalf)]);
                    }
                }
            for(jj = yhalf; jj < naxes1; jj++)
                for(ii = 0; ii < xhalf; ii++)
                {
                    for(kk = 0; kk < naxes2; kk++)
                    {
                        SWAPd(data.image[ID].array.D[kk * naxes0 * naxes1 +
                                                     jj * naxes0 + ii],
                              data.image[ID].array.D[kk * naxes0 * naxes1 +
                                                     (jj - yhalf) * naxes0 +
                                                     (ii + xhalf)]);
                    }
                }
        }
    }

    if(datatype == _DATATYPE_COMPLEX_FLOAT)
    {
        if(naxis == 1)
        {
            OK    = 1;
            xhalf = (long)(naxes0 / 2);
            for(ii = 0; ii < xhalf; ii++)
            {
                CSWAPcf(data.image[ID].array.CF[ii],
                        data.image[ID].array.CF[ii + xhalf]);
            }
        }
        if(naxis == 2)
        {
            OK    = 1;
            xhalf = (long)(naxes0 / 2);
            yhalf = (long)(naxes1 / 2);
            for(jj = 0; jj < yhalf; jj++)
                for(ii = 0; ii < xhalf; ii++)
                {
                    CSWAPcf(data.image[ID].array.CF[jj * naxes0 + ii],
                            data.image[ID].array.CF[(jj + yhalf) * naxes0 +
                                                    (ii + xhalf)]);
                }
            for(jj = yhalf; jj < naxes1; jj++)
                for(ii = 0; ii < xhalf; ii++)
                {
                    CSWAPcf(data.image[ID].array.CF[jj * naxes0 + ii],
                            data.image[ID].array.CF[(jj - yhalf) * naxes0 +
                                                    (ii + xhalf)]);
                }
        }
        if(naxis == 3)
        {
            OK    = 1;
            xhalf = (long)(naxes0 / 2);
            yhalf = (long)(naxes1 / 2);
            for(kk = 0; kk < naxes2; kk++)
                for(jj = 0; jj < yhalf; jj++)
                    for(ii = 0; ii < xhalf; ii++)
                    {
                        CSWAPcf(data.image[ID].array.CF[kk * naxes0 * naxes1 +
                                                        jj * naxes0 + ii],
                                data.image[ID].array.CF[kk * naxes0 * naxes1 +
                                                        (jj + yhalf) * naxes0 +
                                                        (ii + xhalf)]);
                    }
            printf(" - ");
            fflush(stdout);

            for(kk = 0; kk < naxes2; kk++)
                for(jj = yhalf; jj < naxes1; jj++)
                    for(ii = 0; ii < xhalf; ii++)
                    {
                        CSWAPcf(data.image[ID].array.CF[kk * naxes0 * naxes1 +
                                                        jj * naxes0 + ii],
                                data.image[ID].array.CF[kk * naxes0 * naxes1 +
                                                        (jj - yhalf) * naxes0 +
                                                        (ii + xhalf)]);
                    }
        }
    }

    if(datatype == _DATATYPE_COMPLEX_DOUBLE)
    {
        if(naxis == 1)
        {
            OK    = 1;
            xhalf = (long)(naxes0 / 2);
            for(ii = 0; ii < xhalf; ii++)
            {
                CSWAPcd(data.image[ID].array.CD[ii],
                        data.image[ID].array.CD[ii + xhalf]);
            }
        }
        if(naxis == 2)
        {
            OK    = 1;
            xhalf = (long)(naxes0 / 2);
            yhalf = (long)(naxes1 / 2);
            for(jj = 0; jj < yhalf; jj++)
                for(ii = 0; ii < xhalf; ii++)
                {
                    CSWAPcd(data.image[ID].array.CD[jj * naxes0 + ii],
                            data.image[ID].array.CD[(jj + yhalf) * naxes0 +
                                                    (ii + xhalf)]);
                }
            for(jj = yhalf; jj < naxes1; jj++)
                for(ii = 0; ii < xhalf; ii++)
                {
                    CSWAPcd(data.image[ID].array.CD[jj * naxes0 + ii],
                            data.image[ID].array.CD[(jj - yhalf) * naxes0 +
                                                    (ii + xhalf)]);
                }
        }
        if(naxis == 3)
        {
            OK    = 1;
            xhalf = (long)(naxes0 / 2);
            yhalf = (long)(naxes1 / 2);
            for(kk = 0; kk < naxes2; kk++)
                for(jj = 0; jj < yhalf; jj++)
                    for(ii = 0; ii < xhalf; ii++)
                    {
                        CSWAPcd(data.image[ID].array.CD[kk * naxes0 * naxes1 +
                                                        jj * naxes0 + ii],
                                data.image[ID].array.CD[kk * naxes0 * naxes1 +
                                                        (jj + yhalf) * naxes0 +
                                                        (ii + xhalf)]);
                    }
            printf(" - ");
            fflush(stdout);

            for(kk = 0; kk < naxes2; kk++)
                for(jj = yhalf; jj < naxes1; jj++)
                    for(ii = 0; ii < xhalf; ii++)
                    {
                        CSWAPcd(data.image[ID].array.CD[kk * naxes0 * naxes1 +
                                                        jj * naxes0 + ii],
                                data.image[ID].array.CD[kk * naxes0 * naxes1 +
                                                        (jj - yhalf) * naxes0 +
                                                        (ii + xhalf)]);
                    }
        }
    }

    if(OK == 0)
    {
        printf("Error : data format not supported by permut\n");
    }

    //  printf(" done\n");
    // fflush(stdout);

    return (0);
}
