/** @file ffttranslate.c
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

int fft_image_translate(const char *ID_name,
                        const char *ID_out,
                        double      xtransl,
                        double      ytransl);

// ==========================================
// Command line interface wrapper function(s)
// ==========================================

errno_t fft_image_translate_cli()
{
    if(CLI_checkarg(1, CLIARG_IMG) + CLI_checkarg(2, CLIARG_STR_NOT_IMG) +
            CLI_checkarg(3, CLIARG_FLOAT64) + CLI_checkarg(4, CLIARG_FLOAT64) ==
            0)
    {
        fft_image_translate(data.cmdargtoken[1].val.string,
                            data.cmdargtoken[2].val.string,
                            data.cmdargtoken[3].val.numf,
                            data.cmdargtoken[4].val.numf);

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

errno_t ffttranslate_addCLIcmd()
{

    RegisterCLIcommand("transl",
                       __FILE__,
                       fft_image_translate_cli,
                       "translate image",
                       "<imagein> <imageout> <xtransl> <ytransl>",
                       "transl im1 im2 2.3 -2.1",
                       "int fft_image_translate(const char *ID_name, const "
                       "char *ID_out, double xtransl, double ytransl)");

    return RETURN_SUCCESS;
}

/*^-----------------------------------------------------------------------------
|
| COMMENT:  Inclusion of this routine requires inclusion of modules:
|           fft, gen_image
* DOES NOT WORK ON STREAM
+-----------------------------------------------------------------------------*/
int fft_image_translate(const char *ID_name,
                        const char *ID_out,
                        double      xtransl,
                        double      ytransl)
{
    long ID;
    long naxes[2];
    //  int n0,n1;

    ID       = image_ID(ID_name);
    naxes[0] = data.image[ID].md[0].size[0];
    naxes[1] = data.image[ID].md[0].size[1];

    //  fprintf( stdout, "[arith_image_translate %ld %ld %ld     %f %f]\n", ID, naxes[0], naxes[1], xtransl, ytransl);

    // n0 = (int) ((log10(naxes[0])/log10(2))+0.01);
    // n1 = (int) ((log10(naxes[0])/log10(2))+0.01);

    //  if ((n0==n1)&&(naxes[0]==(int) pow(2,n0))&&(naxes[1]==(int) pow(2,n1)))
    // {

    do2drfft(ID_name, "ffttmp1");
    mk_amph_from_complex("ffttmp1", "amptmp", "phatmp", 0);

    delete_image_ID("ffttmp1", DELETE_IMAGE_ERRMODE_WARNING);
    arith_make_slopexy("sltmp",
                       naxes[0],
                       naxes[1],
                       xtransl * 2.0 * M_PI / naxes[0],
                       ytransl * 2.0 * M_PI / naxes[1]);
    permut("sltmp");

    arith_image_add("phatmp", "sltmp", "phatmp1");
    delete_image_ID("phatmp", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("sltmp", DELETE_IMAGE_ERRMODE_WARNING);

    mk_complex_from_amph("amptmp", "phatmp1", "ffttmp2", 0);
    delete_image_ID("amptmp", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("phatmp1", DELETE_IMAGE_ERRMODE_WARNING);
    do2dffti("ffttmp2", "ffttmp3");
    delete_image_ID("ffttmp2", DELETE_IMAGE_ERRMODE_WARNING);
    mk_reim_from_complex("ffttmp3", "retmp", "imtmp", 0);
    arith_image_cstmult("retmp", 1.0 / naxes[0] / naxes[1], ID_out);
    delete_image_ID("ffttmp3", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("retmp", DELETE_IMAGE_ERRMODE_WARNING);
    delete_image_ID("imtmp", DELETE_IMAGE_ERRMODE_WARNING);
    // }
    // else
    //{
    // printf("Error: image size does not allow translation\n");
    //}

    return (0);
}
