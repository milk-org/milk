/** @file extract_RGGBchan.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"




static char *inim;
static long  fpi_inim;

static char *outimR;
static long  fpi_outimR;

static char *outimG1;
static long  fpi_outimG1;

static char *outimG2;
static long  fpi_outimG2;

static char *outimB;
static long  fpi_outimB;




static CLICMDARGDEF farg[] = {{
        CLIARG_STR,
        ".inim",
        "input RGGB image",
        "inim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inim,
        &fpi_inim
    },
    {
        CLIARG_STR,
        ".outimR",
        "output R image",
        "inim",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimR,
        &fpi_outimR
    },
    {
        CLIARG_STR,
        ".outimG1",
        "output G1 image",
        "outimG1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimG1,
        &fpi_outimG1
    },
    {
        CLIARG_STR,
        ".outimG2",
        "output G2 image",
        "outimG2",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimG2,
        &fpi_outimG2
    },
    {
        CLIARG_STR,
        ".outimB",
        "output B image",
        "outimB",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimB,
        &fpi_outimB
    }
};




static CLICMDDATA CLIcmddata = {"extractRGGBchan",
                                "extract RGGB channels from color image",
                                CLICMD_FIELDS_DEFAULTS
                               };



// detailed help
static errno_t help_function()
{
    return RETURN_SUCCESS;
}




/*
    IMGID imgoutR,
    IMGID imgoutG1,
    IMGID imgoutG2,
    IMGID imgoutB
*/

//
// separates a single RGB image into its 4 channels
// output written in im_r, im_g1, im_g2 and im_b
//
errno_t image_format_extract_RGGBchan(
    IMGID imgin, IMGID imgoutR, IMGID imgoutG1, IMGID imgoutG2, IMGID imgoutB)
{
    DEBUG_TRACE_FSTART();

    // input image is required
    resolveIMGID(&imgin, ERRMODE_ABORT);



    copyIMGID(&imgin, &imgoutR);
    imgoutR.size[0] = imgin.size[0] / 2;
    imgoutR.size[1] = imgin.size[1] / 2;

    copyIMGID(&imgoutR, &imgoutG1);
    copyIMGID(&imgoutR, &imgoutG2);
    copyIMGID(&imgoutR, &imgoutB);

    createimagefromIMGID(&imgoutR);
    createimagefromIMGID(&imgoutG1);
    createimagefromIMGID(&imgoutG2);
    createimagefromIMGID(&imgoutB);

    uint32_t xsize = imgin.size[0];

    list_image_ID();



    switch(imgin.datatype)
    {

        case _DATATYPE_FLOAT:
            for(uint32_t ii = 0; ii < imgoutR.size[0]; ii++)
                for(uint32_t jj = 0; jj < imgoutR.size[1]; jj++)
                {
                    uint32_t ii1  = 2 * ii;
                    uint32_t jj1  = 2 * jj;
                    uint64_t pixi = jj * imgoutR.size[0] + ii;

                    imgoutR.im->array.F[pixi] =
                        imgin.im->array.F[(jj1 + 1) * xsize + ii1];
                    imgoutG1.im->array.F[pixi] =
                        imgin.im->array.F[jj1 * xsize + ii1];
                    imgoutG2.im->array.F[pixi] =
                        imgin.im->array.F[(jj1 + 1) * xsize + (ii1 + 1)];
                    imgoutB.im->array.F[pixi] =
                        imgin.im->array.F[jj1 * xsize + (ii1 + 1)];
                }
            break;

        case _DATATYPE_DOUBLE:
            for(uint32_t ii = 0; ii < imgoutR.size[0]; ii++)
                for(uint32_t jj = 0; jj < imgoutR.size[1]; jj++)
                {
                    uint32_t ii1  = 2 * ii;
                    uint32_t jj1  = 2 * jj;
                    uint64_t pixi = jj * imgoutR.size[0] + ii;

                    imgoutR.im->array.D[pixi] =
                        imgin.im->array.D[(jj1 + 1) * xsize + ii1];
                    imgoutG1.im->array.D[pixi] =
                        imgin.im->array.D[jj1 * xsize + ii1];
                    imgoutG2.im->array.D[pixi] =
                        imgin.im->array.D[(jj1 + 1) * xsize + (ii1 + 1)];
                    imgoutB.im->array.D[pixi] =
                        imgin.im->array.D[jj1 * xsize + (ii1 + 1)];
                }
            break;


        case _DATATYPE_UINT16:
            for(uint32_t ii = 0; ii < imgoutR.size[0]; ii++)
                for(uint32_t jj = 0; jj < imgoutR.size[1]; jj++)
                {
                    uint32_t ii1  = 2 * ii;
                    uint32_t jj1  = 2 * jj;
                    uint64_t pixi = jj * imgoutR.size[0] + ii;

                    imgoutR.im->array.UI16[pixi] =
                        imgin.im->array.UI16[(jj1 + 1) * xsize + ii1];
                    imgoutG1.im->array.UI16[pixi] =
                        imgin.im->array.UI16[jj1 * xsize + ii1];
                    imgoutG2.im->array.UI16[pixi] =
                        imgin.im->array.UI16[(jj1 + 1) * xsize + (ii1 + 1)];
                    imgoutB.im->array.UI16[pixi] =
                        imgin.im->array.UI16[jj1 * xsize + (ii1 + 1)];
                }
            break;
    }


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




/**
 * @brief Wrapper function, used by all CLI calls
 *
 * INSERT_STD_PROCINFO statements enable processinfo support
 */
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();



    INSERT_STD_PROCINFO_COMPUTEFUNC_START

    image_format_extract_RGGBchan(mkIMGID_from_name(inim),
                                  mkIMGID_from_name(outimR),
                                  mkIMGID_from_name(outimG1),
                                  mkIMGID_from_name(outimG2),
                                  mkIMGID_from_name(outimB));


    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_image_format__extractRGGBchan()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
