/** @file fftcorrelation.c
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "dofft.h"
#include "permut.h"

// ==========================================
// Forward declaration(s)
// ==========================================

imageID fft_correlation(const char *ID_name1,
                        const char *ID_name2,
                        const char *ID_nameout);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t fft_correlation_cli()
{
    if(CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_IMG) +
            CLI_checkarg(3, CLIARG_STR_NOT_IMG) ==
            0)
    {
        fft_correlation(data.cmdargtoken[1].val.string,
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

errno_t fftcorrelation_addCLIcmd()
{
    RegisterCLIcommand("fcorrel",
                       __FILE__,
                       fft_correlation_cli,
                       "correlate two images",
                       "<imagein1> <imagein2> <correlout>",
                       "fcorrel im1 im2 outim",
                       "long fft_correlation(const char *ID_name1, const char "
                       "*ID_name2, const char *ID_nameout)");

    return RETURN_SUCCESS;
}

imageID fft_correlation(const char *ID_name1,
                        const char *ID_name2,
                        const char *ID_nameout)
{
    imageID ID1;
    imageID IDout;
    long    nelement;

    char ft1name[STRINGMAXLEN_IMGNAME];
    char ft2name[STRINGMAXLEN_IMGNAME];
    char fta1name[STRINGMAXLEN_IMGNAME];
    char fta2name[STRINGMAXLEN_IMGNAME];
    char ftp1name[STRINGMAXLEN_IMGNAME];
    char ftp2name[STRINGMAXLEN_IMGNAME];
    char fta12name[STRINGMAXLEN_IMGNAME];
    char ftp12name[STRINGMAXLEN_IMGNAME];
    char fftname[STRINGMAXLEN_IMGNAME];
    char fft1name[STRINGMAXLEN_IMGNAME];
    char fft1pname[STRINGMAXLEN_IMGNAME];

    ID1      = image_ID(ID_name1);
    nelement = data.image[ID1].md[0].nelement;

    WRITE_IMAGENAME(ft1name, "_ft1_%d", (int) getpid());
    do2drfft(ID_name1, ft1name);

    WRITE_IMAGENAME(ft2name, "_ft2_%d", (int) getpid());
    do2drfft(ID_name2, ft2name);

    WRITE_IMAGENAME(fta1name, "_%s_a_%d", ft1name, (int) getpid());
    WRITE_IMAGENAME(ftp1name, "_%s_p_%d", ft1name, (int) getpid());
    WRITE_IMAGENAME(fta2name, "_%s_a_%d", ft2name, (int) getpid());
    WRITE_IMAGENAME(ftp2name, "_%s_p_%d", ft2name, (int) getpid());
    WRITE_IMAGENAME(fta12name, "_%s_12a_%d", ft1name, (int) getpid());
    WRITE_IMAGENAME(ftp12name, "_%s_12p_%d", ft1name, (int) getpid());

    mk_amph_from_complex(ft1name, fta1name, ftp1name, 0);
    mk_amph_from_complex(ft2name, fta2name, ftp2name, 0);

    delete_image_ID(ft1name, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(ft2name, DELETE_IMAGE_ERRMODE_WARNING);

    arith_image_mult(fta1name, fta2name, fta12name);
    arith_image_sub(ftp1name, ftp2name, ftp12name);
    delete_image_ID(fta1name, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(fta2name, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(ftp1name, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(ftp2name, DELETE_IMAGE_ERRMODE_WARNING);

    arith_image_cstmult_inplace(fta12name,
                                1.0 / sqrt(nelement) / (1.0 * nelement));

    WRITE_IMAGENAME(fftname, "_fft_%d", (int) getpid());

    mk_complex_from_amph(fta12name, ftp12name, fftname, 0);
    delete_image_ID(fta12name, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(ftp12name, DELETE_IMAGE_ERRMODE_WARNING);

    WRITE_IMAGENAME(fft1name, "_fft1_%d", (int) getpid());

    do2dfft(fftname, fft1name);
    delete_image_ID(fftname, DELETE_IMAGE_ERRMODE_WARNING);

    WRITE_IMAGENAME(fft1pname, "_fft1p_%d", (int) getpid());

    mk_amph_from_complex(fft1name, ID_nameout, fft1pname, 0);
    permut(ID_nameout);

    delete_image_ID(fft1name, DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID(fft1pname, DELETE_IMAGE_ERRMODE_WARNING);

    IDout = image_ID(ID_nameout);

    return IDout;
}
