/**
 * @file    image_merge3D.c
 * @brief   merge 3D images
 *
 *
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"






// input image names
static char *inimname0;
static char *inimname1;

static char *outimname;



static uint32_t *mergeaxis;
static long      fpi_mergeaxis = -1;




static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".in0name",
        "input image 0",
        "im0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname0,
        NULL
    },
    {
        CLIARG_IMG,
        ".in1name",
        "input image 1",
        "im1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inimname1,
        NULL
    },
    {
        CLIARG_STR,
        ".outname",
        "output image",
        "im0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimname,
        NULL
    },
    {
        CLIARG_UINT32,
        ".axis",
        "merge axis",
        "0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &mergeaxis,
        &fpi_mergeaxis
    },
};





static CLICMDDATA CLIcmddata =
{
    "immerge",
    "merge images along axis",
    CLICMD_FIELDS_DEFAULTS
};



// detailed help
static errno_t help_function()
{
    printf("Merge two images along specified axis\n");

    return RETURN_SUCCESS;
}



errno_t image_marge(
    IMGID inimg0,
    IMGID inimg1,
    IMGID *outimg,
    uint8_t mergeaxis
)
{
    DEBUG_TRACE_FSTART();

    resolveIMGID(&inimg0, ERRMODE_ABORT);
    resolveIMGID(&inimg1, ERRMODE_ABORT);


    resolveIMGID(outimg, ERRMODE_NULL);
    if( outimg->ID == -1)
    {
        copyIMGID(&inimg0, outimg);
    }

    if ( mergeaxis < 3)
    {
        uint32_t size0;
        uint32_t size1;

        size0 = (inimg0.md->size[mergeaxis] == 0) ? 1 : inimg0.md->size[mergeaxis];
        size1 = (inimg1.md->size[mergeaxis] == 0) ? 1 : inimg1.md->size[mergeaxis];
        outimg->size[mergeaxis] = size0 + size1;
    }
    else
    {
        PRINT_ERROR("mergeaxis %u not supported", mergeaxis);
        abort();
    }



    outimg->naxis = 1;
    if ( outimg->size[1] > 1 )
    {
        outimg->naxis = 2;
    }
    if ( outimg->size[2] > 1 )
    {
        outimg->naxis = 3;
    }

    createimagefromIMGID(outimg);


    if ( mergeaxis == outimg->naxis-1 )
    {
        // we can simply memcpy

        switch ( outimg->datatype )
        {

        case _DATATYPE_UINT8 :
            memcpy(&outimg->im->array.UI8[0], &inimg0.im->array.UI8[0], sizeof(uint8_t)*inimg0.md->nelement);
            memcpy(&outimg->im->array.UI8[inimg0.md->nelement], &inimg1.im->array.UI8, sizeof(uint8_t)*inimg1.md->nelement);
            break;

        case _DATATYPE_INT8 :
            memcpy(&outimg->im->array.SI8[0], &inimg0.im->array.SI8[0], sizeof(int8_t)*inimg0.md->nelement);
            memcpy(&outimg->im->array.SI8[inimg0.md->nelement], &inimg1.im->array.SI8, sizeof(int8_t)*inimg1.md->nelement);
            break;


        case _DATATYPE_UINT16 :
            memcpy(&outimg->im->array.UI16[0], &inimg0.im->array.UI16[0], sizeof(uint16_t)*inimg0.md->nelement);
            memcpy(&outimg->im->array.UI16[inimg0.md->nelement], &inimg1.im->array.UI16, sizeof(uint16_t)*inimg1.md->nelement);
            break;

        case _DATATYPE_INT16 :
            memcpy(&outimg->im->array.SI16[0], &inimg0.im->array.SI16[0], sizeof(int16_t)*inimg0.md->nelement);
            memcpy(&outimg->im->array.SI16[inimg0.md->nelement], &inimg1.im->array.SI16, sizeof(int16_t)*inimg1.md->nelement);
            break;


        case _DATATYPE_UINT32 :
            memcpy(&outimg->im->array.UI32[0], &inimg0.im->array.UI32[0], sizeof(uint32_t)*inimg0.md->nelement);
            memcpy(&outimg->im->array.UI32[inimg0.md->nelement], &inimg1.im->array.UI32, sizeof(uint32_t)*inimg1.md->nelement);
            break;

        case _DATATYPE_INT32 :
            memcpy(&outimg->im->array.SI32[0], &inimg0.im->array.SI32[0], sizeof(int32_t)*inimg0.md->nelement);
            memcpy(&outimg->im->array.SI32[inimg0.md->nelement], &inimg1.im->array.SI32, sizeof(int32_t)*inimg1.md->nelement);
            break;


        case _DATATYPE_UINT64 :
            memcpy(&outimg->im->array.UI64[0], &inimg0.im->array.UI64[0], sizeof(uint64_t)*inimg0.md->nelement);
            memcpy(&outimg->im->array.UI64[inimg0.md->nelement], &inimg1.im->array.UI64, sizeof(uint64_t)*inimg1.md->nelement);
            break;

        case _DATATYPE_INT64 :
            memcpy(&outimg->im->array.SI64[0], &inimg0.im->array.SI64[0], sizeof(int64_t)*inimg0.md->nelement);
            memcpy(&outimg->im->array.SI64[inimg0.md->nelement], &inimg1.im->array.SI64, sizeof(int64_t)*inimg1.md->nelement);
            break;

        case _DATATYPE_FLOAT :
            printf("datatype FLOAT\n");
            memcpy(&outimg->im->array.F[0], &inimg0.im->array.F[0], sizeof(float)*inimg0.md->nelement);
            memcpy(&outimg->im->array.F[inimg0.md->nelement], &inimg1.im->array.F[0], sizeof(float)*inimg1.md->nelement);
            break;

        case _DATATYPE_DOUBLE :
            memcpy(&outimg->im->array.D[0], &inimg0.im->array.D[0], sizeof(double)*inimg0.md->nelement);
            memcpy(&outimg->im->array.D[inimg0.md->nelement], &inimg1.im->array.D, sizeof(double)*inimg1.md->nelement);
            break;

        default:
            PRINT_ERROR("datatype %u not supported", outimg->datatype );
            abort();
        }
    }
    else
    {
        // block size for memcpy in number of pixel

        uint64_t blocksize_out = outimg->size[0];
        uint64_t blocksize_in0 = inimg0.size[0];
        uint64_t blocksize_in1 = inimg1.size[0];

        if (mergeaxis == 1)
        {
            blocksize_out *= outimg->size[1];
            blocksize_in0 *= inimg0.size[1];
            blocksize_in1 *= inimg1.size[1];
        }

        uint64_t pixiout = 0;
        uint64_t pixiin0 = 0;
        uint64_t pixiin1 = 0;

        switch ( outimg->datatype )
        {

        case _DATATYPE_UINT8 :
            while ( pixiout < outimg->md->nelement)
            {
                memcpy(&outimg->im->array.UI8[pixiout], &inimg0.im->array.UI8[pixiin0],
                       sizeof(uint8_t)*blocksize_in0);
                pixiin0 += blocksize_in0;
                pixiout += blocksize_in0;

                memcpy(&outimg->im->array.UI8[pixiout], &inimg1.im->array.UI8[pixiin1],
                       sizeof(uint8_t)*blocksize_in1);
                pixiin1 += blocksize_in1;
                pixiout += blocksize_in1;
            }
            break;

        case _DATATYPE_INT8 :
            while ( pixiout < outimg->md->nelement)
            {
                memcpy(&outimg->im->array.SI8[pixiout], &inimg0.im->array.SI8[pixiin0],
                       sizeof(int8_t)*blocksize_in0);
                pixiin0 += blocksize_in0;
                pixiout += blocksize_in0;

                memcpy(&outimg->im->array.SI8[pixiout], &inimg1.im->array.SI8[pixiin1],
                       sizeof(int8_t)*blocksize_in1);
                pixiin1 += blocksize_in1;
                pixiout += blocksize_in1;
            }
            break;

        case _DATATYPE_UINT16 :
            while ( pixiout < outimg->md->nelement)
            {
                memcpy(&outimg->im->array.UI16[pixiout], &inimg0.im->array.UI16[pixiin0],
                       sizeof(uint16_t)*blocksize_in0);
                pixiin0 += blocksize_in0;
                pixiout += blocksize_in0;

                memcpy(&outimg->im->array.UI16[pixiout], &inimg1.im->array.UI16[pixiin1],
                       sizeof(uint16_t)*blocksize_in1);
                pixiin1 += blocksize_in1;
                pixiout += blocksize_in1;
            }
            break;

        case _DATATYPE_INT16 :
            while ( pixiout < outimg->md->nelement)
            {
                memcpy(&outimg->im->array.SI16[pixiout], &inimg0.im->array.SI16[pixiin0],
                       sizeof(int16_t)*blocksize_in0);
                pixiin0 += blocksize_in0;
                pixiout += blocksize_in0;

                memcpy(&outimg->im->array.SI16[pixiout], &inimg1.im->array.SI16[pixiin1],
                       sizeof(int16_t)*blocksize_in1);
                pixiin1 += blocksize_in1;
                pixiout += blocksize_in1;
            }
            break;

        case _DATATYPE_UINT32 :
            while ( pixiout < outimg->md->nelement)
            {
                memcpy(&outimg->im->array.UI32[pixiout], &inimg0.im->array.UI32[pixiin0],
                       sizeof(uint32_t)*blocksize_in0);
                pixiin0 += blocksize_in0;
                pixiout += blocksize_in0;

                memcpy(&outimg->im->array.UI32[pixiout], &inimg1.im->array.UI32[pixiin1],
                       sizeof(uint32_t)*blocksize_in1);
                pixiin1 += blocksize_in1;
                pixiout += blocksize_in1;
            }
            break;

        case _DATATYPE_INT32 :
            while ( pixiout < outimg->md->nelement)
            {
                memcpy(&outimg->im->array.SI32[pixiout], &inimg0.im->array.SI32[pixiin0],
                       sizeof(int32_t)*blocksize_in0);
                pixiin0 += blocksize_in0;
                pixiout += blocksize_in0;

                memcpy(&outimg->im->array.SI32[pixiout], &inimg1.im->array.SI32[pixiin1],
                       sizeof(int32_t)*blocksize_in1);
                pixiin1 += blocksize_in1;
                pixiout += blocksize_in1;
            }
            break;


        case _DATATYPE_UINT64 :
            while ( pixiout < outimg->md->nelement)
            {
                memcpy(&outimg->im->array.UI64[pixiout], &inimg0.im->array.UI64[pixiin0],
                       sizeof(uint64_t)*blocksize_in0);
                pixiin0 += blocksize_in0;
                pixiout += blocksize_in0;

                memcpy(&outimg->im->array.UI64[pixiout], &inimg1.im->array.UI64[pixiin1],
                       sizeof(uint64_t)*blocksize_in1);
                pixiin1 += blocksize_in1;
                pixiout += blocksize_in1;
            }
            break;

        case _DATATYPE_INT64 :
            while ( pixiout < outimg->md->nelement)
            {
                memcpy(&outimg->im->array.SI64[pixiout], &inimg0.im->array.SI64[pixiin0],
                       sizeof(int64_t)*blocksize_in0);
                pixiin0 += blocksize_in0;
                pixiout += blocksize_in0;

                memcpy(&outimg->im->array.SI64[pixiout], &inimg1.im->array.SI64[pixiin1],
                       sizeof(int64_t)*blocksize_in1);
                pixiin1 += blocksize_in1;
                pixiout += blocksize_in1;
            }
            break;


        case _DATATYPE_FLOAT :
            while ( pixiout < outimg->md->nelement)
            {
                memcpy(&outimg->im->array.F[pixiout], &inimg0.im->array.F[pixiin0],
                       sizeof(float)*blocksize_in0);
                pixiin0 += blocksize_in0;
                pixiout += blocksize_in0;

                memcpy(&outimg->im->array.F[pixiout], &inimg1.im->array.F[pixiin1],
                       sizeof(float)*blocksize_in1);
                pixiin1 += blocksize_in1;
                pixiout += blocksize_in1;
            }
            break;

        case _DATATYPE_DOUBLE :
            while ( pixiout < outimg->md->nelement)
            {
                memcpy(&outimg->im->array.D[pixiout], &inimg0.im->array.D[pixiin0],
                       sizeof(double)*blocksize_in0);
                pixiin0 += blocksize_in0;
                pixiout += blocksize_in0;

                memcpy(&outimg->im->array.D[pixiout], &inimg1.im->array.D[pixiin1],
                       sizeof(double)*blocksize_in1);
                pixiin1 += blocksize_in1;
                pixiout += blocksize_in1;
            }
            break;


        default:
            PRINT_ERROR("datatype %u not supported", outimg->datatype );
            abort();
        }

    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}





static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID inimg0 = mkIMGID_from_name(inimname0);
    resolveIMGID(&inimg0, ERRMODE_ABORT);

    IMGID inimg1 = mkIMGID_from_name(inimname1);
    resolveIMGID(&inimg1, ERRMODE_ABORT);


    IMGID outimg = mkIMGID_from_name(outimname);



    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        image_marge(
            inimg0,
            inimg1,
            &outimg,
            *mergeaxis
        );

        processinfo_update_output_stream(processinfo, outimg.ID);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}



INSERT_STD_FPSCLIfunctions



// Register function in CLI
errno_t
CLIADDCMD_COREMOD_arith__image_merge()
{
    //CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    //CLIcmddata.FPS_customCONFcheck = customCONFcheck;

    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
