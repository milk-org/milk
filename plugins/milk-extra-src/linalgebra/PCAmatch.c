#include "ImageStreamIO/ImageStruct.h"
/**
 * @file PCAmatch.c
 *
 * @brief match two PCA decompositions
 *
 * Find corresponding linear combination across two basis
 *
 */

#include <math.h>

#include "CLIcore.h"

#include "SGEMM.h"



static char *modesA;
static long  fpi_modesA;

static char *modesB;
static long  fpi_modesB;

static char *outcoeffA;
static long  fpi_outcoeffA;

static char *outcoeffB;
static long  fpi_outcoeffB;

static char *outimA;
static long  fpi_outimA;

static char *outimB;
static long  fpi_outimB;


static int32_t *GPUdevice;
static long     fpi_GPUdevice;






static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".modesA",
        "input modes A",
        "inmA",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &modesA,
        &fpi_modesA
    },
    {
        CLIARG_IMG,
        ".modesB",
        "input modes B",
        "inmB",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &modesB,
        &fpi_modesB
    },
    {
        CLIARG_STR,
        ".outcoeffA",
        "output coeffs A",
        "outcA",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outcoeffA,
        &fpi_outcoeffA
    },
    {
        CLIARG_STR,
        ".outcoeffB",
        "output coeffs B",
        "outcB",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outcoeffB,
        &fpi_outcoeffB
    },
    {
        CLIARG_STR,
        ".outimA",
        "output image A",
        "outcA",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimA,
        &fpi_outimA
    },
    {
        CLIARG_STR,
        ".outimB",
        "output image B",
        "outcB",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outimB,
        &fpi_outimB
    },
    {
        // using GPU (99 : no GPU, otherwise GPU device)
        CLIARG_INT32,
        ".GPUdevice",
        "GPU device, 99 for CPU",
        "-1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &GPUdevice,
        &fpi_GPUdevice
    }
};





static CLICMDDATA CLIcmddata =
{
    "PCAmatch", "find matching linear combination across two modal bases", CLICMD_FIELDS_DEFAULTS
};



static errno_t help_function()
{


    return RETURN_SUCCESS;
}




/**
 * @brief Find matching linear combinations across two bases
 *
 * @param imgmodesA  basis A
 * @param imgmodesB  basis B
 * @param imgoutcA   output coeffs A
 * @param imgoutcB   output coeffs B
 * @param imgoutimA  output image A
 * @param imgoutimB  output image B
 * @param GPUdev     GPU device
 * @return errno_t
 */
errno_t PCAmatch(
    IMGID    imgmodesA,
    IMGID    imgmodesB,
    IMGID    *imgoutcA,
    IMGID    *imgoutcB,
    IMGID    *imgoutimA,
    IMGID    *imgoutimB,
    int      GPUdev
)
{
    DEBUG_TRACE_FSTART();

    uint32_t NBmodesA = imgmodesA.md->size[2];
    uint32_t NBmodesB = imgmodesA.md->size[2];

    printf("NBmodesA = %u\n", NBmodesA);
    printf("NBmodesB = %u\n", NBmodesB);
    fflush(stdout);

    // output vectors

    imgoutcA->mdt->datatype = _DATATYPE_FLOAT;
    imgoutcA->mdt->naxis = 2;
    imgoutcA->mdt->size[0] = NBmodesA;
    imgoutcA->mdt->size[1] = 1;
    printf("CREATING %s\n", imgoutcA->name);
    fflush(stdout);
    createimagefromIMGID(imgoutcA);


    imgoutcB->mdt->datatype = _DATATYPE_FLOAT;
    imgoutcB->mdt->naxis = 2;
    imgoutcB->mdt->size[0] = NBmodesA;
    imgoutcB->mdt->size[1] = 1;
    printf("CREATING %s\n", imgoutcB->name);
    fflush(stdout);
    createimagefromIMGID(imgoutcB);







    // A->B coeff remapping matrix
    IMGID imgAtoB;
    strcpy(imgAtoB.name, "matAtoB");
    computeSGEMM(
        imgmodesB,
        imgmodesA,
        &imgAtoB,
        1, 0,
        GPUdev
    );

    // B->A coeff remapping matrix
    IMGID imgBtoA;
    strcpy(imgBtoA.name, "matBtoA");
    computeSGEMM(
        imgmodesA,
        imgmodesB,
        &imgBtoA,
        1, 0,
        GPUdev
    );


    // Initialization
    imgoutcA->im->array.F[0] = 1.0;
    for(uint32_t mode=1; mode < NBmodesA; mode++)
    {
        imgoutcA->im->array.F[mode] = 0.0;
    }

    imgoutcB->im->array.F[0] = 1.0;
    for(uint32_t mode=1; mode < NBmodesB; mode++)
    {
        imgoutcB->im->array.F[mode] = 0.0;
    }




    // residual0
    IMGID imgimres0  = imgid_make_from_name("imres0");
    imgimres0.mdt->naxis   = 2;
    imgimres0.mdt->size[0] = imgmodesA.md->size[0];
    imgimres0.mdt->size[1] = imgmodesA.md->size[1];
    createimagefromIMGID(&imgimres0);

    double resim0 = 0.0;
    for(uint64_t ii=0; ii< imgmodesA.md->size[0]*imgmodesA.md->size[1]; ii++)
    {
        double vA = imgmodesA.im->array.F[ii];
        double vB = imgmodesB.im->array.F[ii];
        double vdiff = vA-vB;
        resim0 += vdiff*vdiff;
        imgimres0.im->array.F[ii] =  vdiff;
    }





    // project to B
    computeSGEMM(
        imgAtoB,
        *imgoutcA,
        imgoutcB,
        0, 0,
        GPUdev
    );


    int NBiter = 1000;
    for(int iter=0; iter<NBiter; iter++)
    {
        // project to A
        computeSGEMM(
            imgBtoA,
            *imgoutcB,
            imgoutcA,
            0, 0,
            GPUdev
        );

        // attenuate non-average terms
        //imgoutcA->im->array.F[0] = 1.0;
        for(uint32_t mode=1; mode < NBmodesA; mode++)
        {
            imgoutcA->im->array.F[mode] *= 0.999;
        }

        // normalize vector A
        {
            double norm = 0.0;
            for(uint32_t mode=0; mode < NBmodesA; mode++)
            {
                double val = imgoutcA->im->array.F[mode];
                norm += val*val;
            }
            norm = sqrt(norm);
            printf("   A norm = %f\n", norm);

            for(uint32_t mode=0; mode < NBmodesA; mode++)
            {
                imgoutcA->im->array.F[mode] /= norm;
            }
        }



        // project to B
        computeSGEMM(
            imgAtoB,
            *imgoutcA,
            imgoutcB,
            0, 0,
            GPUdev
        );


        printf("[%5d] coeffs A :  ", iter);
        for(uint32_t mode=0; mode < NBmodesA; mode++)
        {
            if(mode < 16)
            {
                printf("%+8.6f  ", imgoutcA->im->array.F[mode]);
            }
        }
        printf("\n");

        printf("[%5d] coeffs B :  ", iter);
        for(uint32_t mode=0; mode < NBmodesB; mode++)
        {
            if(mode < 16)
            {
                printf("%+8.6f  ", imgoutcB->im->array.F[mode]);
            }
        }
        printf("\n");

        printf("\n");

    }





    // compute output images
    computeSGEMM(
        imgmodesA,
        *imgoutcA,
        imgoutimA,
        0, 0,
        GPUdev
    );

    computeSGEMM(
        imgmodesB,
        *imgoutcB,
        imgoutimB,
        0, 0,
        GPUdev
    );



    IMGID imgimres  = imgid_make_from_name("imres");
    imgimres.mdt->naxis   = 2;
    imgimres.mdt->size[0] = imgmodesA.md->size[0];
    imgimres.mdt->size[1] = imgmodesA.md->size[1];
    createimagefromIMGID(&imgimres);

    double resim = 0.0;
    for(uint64_t ii=0; ii< imgmodesA.md->size[0]*imgmodesA.md->size[1]; ii++)
    {
        double vA = imgoutimA->im->array.F[ii];
        double vB = imgoutimB->im->array.F[ii];
        double vdiff = vA-vB;
        resim += vdiff*vdiff;
        imgimres.im->array.F[ii] =  vdiff;
    }

    printf("RESIDUAL %g -> %g\n", resim0, resim);
    printf("GAIN = %f\n", resim0/resim);
    printf("\n");

    imgid_free(&imgAtoB);
    imgid_free(&imgBtoA);
    imgid_free(&imgimres0);
    imgid_free(&imgimres);



    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}






static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgmodesA = imgid_make_from_name(modesA);
    resolveIMGID(&imgmodesA, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);

    IMGID imgmodesB = imgid_make_from_name(modesB);
    resolveIMGID(&imgmodesB, ERRMODE_ABORT, data.image, data.NB_MAX_IMAGE);

    printf("Modes images IDs : %ld %ld\n", imgmodesA.ID, imgmodesB.ID);
    fflush(stdout);


    printf("outcoeffA = %s\n", outcoeffA);
    fflush(stdout);
    IMGID imgoutcA  = imgid_make_from_name(outcoeffA);

    printf("outcoeffB = %s\n", outcoeffB);
    fflush(stdout);
    IMGID imgoutcB  = imgid_make_from_name(outcoeffB);


    printf("imgoutimA = %s\n", outimA);
    fflush(stdout);
    IMGID imgoutimA  = imgid_make_from_name(outimA);

    printf("imgoutimB = %s\n", outimB);
    fflush(stdout);
    IMGID imgoutimB  = imgid_make_from_name(outimB);


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        PCAmatch(
            imgmodesA,
            imgmodesB,
            &imgoutcA,
            &imgoutcB,
            &imgoutimA,
            &imgoutimB,
            *GPUdevice
        );

        processinfo_update_output_stream(processinfo, imgoutcA.im, NULL);
        processinfo_update_output_stream(processinfo, imgoutcB.im, NULL);
        //processinfo_update_output_stream(processinfo, imgoutimA.im, NULL);
        //processinfo_update_output_stream(processinfo, imgoutimB.im, NULL);

    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imgmodesA);
    imgid_free(&imgmodesB);
    imgid_free(&imgoutcA);
    imgid_free(&imgoutcB);
    imgid_free(&imgoutimA);
    imgid_free(&imgoutimB);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions


errno_t CLIADDCMD_linalgebra__PCAmatch()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

