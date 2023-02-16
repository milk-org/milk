/** @file fconvolve.c
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "fft/fft.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID fconvolve(const char *__restrict name_in,
                  const char *__restrict name_ke,
                  const char *__restrict name_out);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

static errno_t fconvolve_cli()
{
    if(0 + CLI_checkarg(1, 4) + CLI_checkarg(2, 4) + CLI_checkarg(3, 3) == 0)
    {
        fconvolve(data.cmdargtoken[1].val.string,
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

errno_t fconvolve_addCLIcmd()
{

    RegisterCLIcommand("fconv",
                       __FILE__,
                       fconvolve_cli,
                       "Fourier-based convolution",
                       "<input image> <kernel> <output image>",
                       "fconv imin kernim imout",
                       "long fconvolve(const char *ID_in, const char *ID_ke, "
                       "const char *ID_out)");

    return RETURN_SUCCESS;
}

imageID fconvolve(const char *__restrict name_in,
                  const char *__restrict name_ke,
                  const char *__restrict name_out)
{
    imageID IDin;
    imageID IDke;
    long    naxes[2];
    imageID IDout;

    IDin     = image_ID(name_in);
    naxes[0] = data.image[IDin].md[0].size[0];
    naxes[1] = data.image[IDin].md[0].size[1];
    IDke     = image_ID(name_ke);
    if((naxes[0] != data.image[IDke].md[0].size[0]) ||
            (naxes[1] != data.image[IDke].md[0].size[1]))
    {
        fprintf(stderr,
                "ERROR in function fconvolve: image and kernel have different "
                "sizes\n");
        exit(0);
    }
    //  save_fl_fits(name_in,"test1.fits");
    // save_fl_fits(name_ke,"test2.fits");

    do2drfft(name_in, "infft");
    do2drfft(name_ke, "kefft");

    arith_image_Cmult("infft", "kefft", "outfft");
    delete_image_ID("infft", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("kefft", DELETE_IMAGE_ERRMODE_WARNING);
    do2dffti("outfft", "outfft1");
    delete_image_ID("outfft", DELETE_IMAGE_ERRMODE_WARNING);
    mk_reim_from_complex("outfft1", "tmpre", "tmpim", 0);

    //  save_fl_fits("tmpre","tmpre.fits");
    // save_fl_fits("tmpim","tmpim.fits");

    delete_image_ID("outfft1", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmpim", DELETE_IMAGE_ERRMODE_WARNING);
    arith_image_cstmult("tmpre", 1.0 / naxes[0] / naxes[1], name_out);
    delete_image_ID("tmpre", DELETE_IMAGE_ERRMODE_WARNING);
    permut(name_out);

    IDout = image_ID(name_out);

    return IDout;
}

// to avoid edge effects
imageID fconvolve_padd(const char *__restrict name_in,
                       const char *__restrict name_ke,
                       long paddsize,
                       const char *__restrict name_out)
{
    imageID IDin;
    imageID IDke;
    imageID ID1;
    imageID ID2;
    imageID ID3;
    imageID IDout;
    long    naxes[2];
    long    naxespadd[2];
    long    ii, jj;

    IDin     = image_ID(name_in);
    naxes[0] = data.image[IDin].md[0].size[0];
    naxes[1] = data.image[IDin].md[0].size[1];
    IDke     = image_ID(name_ke);
    if((naxes[0] != data.image[IDke].md[0].size[0]) ||
            (naxes[1] != data.image[IDke].md[0].size[1]))
    {
        fprintf(stderr,
                "ERROR in function fconvolve: image and kernel have different "
                "sizes\n");
        exit(0);
    }

    naxespadd[0] = naxes[0] + 2 * paddsize;
    naxespadd[1] = naxes[1] + 2 * paddsize;

    // printf("new axes : %ld %ld\n",naxespadd[0],naxespadd[1]);

    create_2Dimage_ID("tmpimpadd", naxespadd[0], naxespadd[1], &ID1);
    create_2Dimage_ID("tmpkepadd", naxespadd[0], naxespadd[1], &ID2);
    create_2Dimage_ID("tmpim1padd", naxespadd[0], naxespadd[1], &ID3);

    for(ii = 0; ii < naxes[0]; ii++)
        for(jj = 0; jj < naxes[1]; jj++)
        {
            data.image[ID1]
            .array.F[(jj + paddsize) * naxespadd[0] + (ii + paddsize)] =
                data.image[IDin].array.F[jj * naxes[0] + ii];
            data.image[ID2]
            .array.F[(jj + paddsize) * naxespadd[0] + (ii + paddsize)] =
                data.image[IDke].array.F[jj * naxes[0] + ii];
            data.image[ID3]
            .array.F[(jj + paddsize) * naxespadd[0] + (ii + paddsize)] =
                1.0;
        }

    //  list_image_ID();
    //  printf("Doing convolutions...");
    //  fflush(stdout);

    fconvolve("tmpimpadd", "tmpkepadd", "tmpconv1");
    fconvolve("tmpim1padd", "tmpkepadd", "tmpconv2");

    //  printf(" done\n");
    // fflush(stdout);

    delete_image_ID("tmpimpadd", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmpkepadd", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmpim1padd", DELETE_IMAGE_ERRMODE_WARNING);

    ID1 = image_ID("tmpconv1");
    ID2 = image_ID("tmpconv2");
    create_2Dimage_ID(name_out, naxes[0], naxes[1], &IDout);

    for(ii = 0; ii < naxes[0]; ii++)
        for(jj = 0; jj < naxes[1]; jj++)
        {
            data.image[IDout].array.F[jj * naxes[0] + ii] =
                data.image[ID1]
                .array.F[(jj + paddsize) * naxespadd[0] + (ii + paddsize)] /
                data.image[ID2]
                .array.F[(jj + paddsize) * naxespadd[0] + (ii + paddsize)];
        }
    delete_image_ID("tmpconv1", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmpconv2", DELETE_IMAGE_ERRMODE_WARNING);

    return IDout;
}

imageID fconvolve_1(const char *__restrict name_in,
                    const char *__restrict kefft,
                    const char *__restrict name_out)
{
    /* FFT of kernel has already been done */
    imageID IDin;
    long    naxes[2];

    IDin     = image_ID(name_in);
    naxes[0] = data.image[IDin].md[0].size[0];
    naxes[1] = data.image[IDin].md[0].size[1];

    do2drfft(name_in, "infft");

    arith_image_Cmult("infft", kefft, "outfft");
    delete_image_ID("infft", DELETE_IMAGE_ERRMODE_WARNING);
    do2dffti("outfft", "outfft1");
    delete_image_ID("outfft", DELETE_IMAGE_ERRMODE_WARNING);
    mk_reim_from_complex("outfft1", "tmpre", "tmpim", 0);
    delete_image_ID("outfft1", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmpim", DELETE_IMAGE_ERRMODE_WARNING);
    arith_image_cstmult("tmpre", 1.0 / naxes[0] / naxes[1], name_out);
    delete_image_ID("tmpre", DELETE_IMAGE_ERRMODE_WARNING);
    permut(name_out);
    imageID IDout = image_ID(name_out);

    return IDout;
}

// if blocksize = 512, for images > 512x512, break image in 512x512 overlapping blocks
// kernel image must be blocksize
imageID fconvolveblock(const char *__restrict name_in,
                       const char *__restrict name_ke,
                       const char *__restrict name_out,
                       long blocksize)
{
    imageID IDin;
    imageID IDout;
    imageID IDtmp;
    imageID IDtmpout;
    imageID IDcnt;
    long    xsize, ysize;
    long    overlap;
    long    ii, jj, ii0, jj0;
    float   gain;
    float   alpha = 4.0;

    overlap = (long)(blocksize / 10);
    IDin    = image_ID(name_in);
    xsize   = data.image[IDin].md[0].size[0];
    ysize   = data.image[IDin].md[0].size[1];

    create_2Dimage_ID(name_out, xsize, ysize, &IDout);

    create_2Dimage_ID("tmpblock", blocksize, blocksize, &IDtmp);

    create_2Dimage_ID("tmpcnt", xsize, ysize, &IDcnt);

    for(ii = 0; ii < xsize * ysize; ii++)
    {
        data.image[IDcnt].array.F[ii] = 0.0;
    }

    for(ii0 = 0; ii0 < xsize - overlap; ii0 += blocksize - overlap)
        for(jj0 = 0; jj0 < ysize - overlap; jj0 += blocksize - overlap)
        {
            for(ii = 0; ii < blocksize; ii++)
                for(jj = 0; jj < blocksize; jj++)
                {
                    if((ii0 + ii < xsize) && (jj0 + jj < ysize))
                    {
                        data.image[IDtmp].array.F[jj * blocksize + ii] =
                            data.image[IDin]
                            .array.F[(jj0 + jj) * xsize + (ii0 + ii)];
                    }
                    else
                    {
                        data.image[IDtmp].array.F[jj * blocksize + ii] = 0.0;
                    }
                }
            fconvolve("tmpblock", name_ke, "tmpblockc");
            IDtmpout = image_ID("tmpblockc");
            for(ii = 0; ii < blocksize; ii++)
                for(jj = 0; jj < blocksize; jj++)
                {
                    if((ii0 + ii < xsize) && (jj0 + jj < ysize))
                    {
                        gain = 1.0;
                        if(ii < overlap)
                        {
                            gain *= pow(1.0 * (1.0 * ii / overlap), alpha);
                        }
                        if(jj < overlap)
                        {
                            gain *= pow(1.0 * (1.0 * jj / overlap), alpha);
                        }
                        if(ii > blocksize - overlap)
                        {
                            gain *=
                                pow(1.0 * (1.0 * (blocksize - ii) / overlap),
                                    alpha);
                        }
                        if(jj > blocksize - overlap)
                        {
                            gain *=
                                pow(1.0 * (1.0 * (blocksize - jj) / overlap),
                                    alpha);
                        }

                        data.image[IDout]
                        .array.F[(jj0 + jj) * xsize + (ii0 + ii)] +=
                            gain *
                            data.image[IDtmpout].array.F[jj * blocksize + ii];
                        data.image[IDcnt]
                        .array.F[(jj0 + jj) * xsize + (ii0 + ii)] +=
                            gain * 1.0;
                    }
                }
        }
    //  save_fl_fits("tmpcnt","tmpcnt.fits");
    // exit(0);
    for(ii = 0; ii < xsize * ysize; ii++)
    {
        data.image[IDout].array.F[ii] /= data.image[IDcnt].array.F[ii] + 1.0e-8;
    }

    delete_image_ID("tmpcnt", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmpblock", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("tmpblockc", DELETE_IMAGE_ERRMODE_WARNING);

    return IDout;
}
