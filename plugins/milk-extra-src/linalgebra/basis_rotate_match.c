/**
 * @file SGEMM.c
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_iofits/COREMOD_iofits.h"




static char *inmatAB;
static long  fpi_inmatAB;

static char *outmatArot;
static long  fpi_outmatArot;


static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".matAB",
        "input decomposition of modes B in basis A",
        "matA",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &inmatAB,
        &fpi_inmatAB
    },
    {
        CLIARG_STR,
        ".matArot",
        "output rotation matrix",
        "matA",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outmatArot,
        &fpi_outmatArot
    }
};


static CLICMDDATA CLIcmddata =
{
    "basisrotmatch", "rotate modal basis to fit modes", CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    printf("Force modal basis A to match set of modes B a much as possible\n");
    printf("basis A is assumed to be orthonormal\n");
    printf("set of modes B has no constraint\n");
    printf("The imput to this function is the modal decomposition of vectors B on the modal basis A\n");
    printf("Match is enforced by rotations that preserve basis A orthonormality\n");

    return RETURN_SUCCESS;
}


errno_t compute_basis_rotate_match(
    IMGID imginAB,
    IMGID *imgArot
)
{
    DEBUG_TRACE_FSTART();

    int Adim = imginAB.md->size[1];
    int Bdim = imginAB.md->size[0];


    // internal Arot array, double for improved prescision
    //
    double * __restrict Arot = (double *) malloc(sizeof(double) * Adim * Adim );
    // Initialize: set Arot to identity matrix
    //
    for(uint64_t ii; ii<Adim*Adim; ii++)
    {
        Arot[ii] = 0.0;
    }
    for(uint32_t ii; ii<Adim; ii++)
    {
        Arot[Adim*ii + ii] = 1.0;
    }



    for( int iB = 0; iB < Bdim; iB++)
    {
        //printf("Processing vector %d / %d\n", iB, Bdim);
        // start from last mode
        int modei = Adim-1;

        while(modei > iB)
        {
            // to be maximized
            int aindex = iB*Bdim + iB;
            double vala = imginAB.im->array.F[aindex];

            // to be minimized
            int bindex = modei*Bdim + iB;
            double valb = imginAB.im->array.F[bindex];


            // rotation angle
            //
            double theta = atan2(-valb, vala);

            //printf("    modes %5u %5d rot = %f\n", iB, modei, theta);

            //printf("    IN  [%f %f]\n", vala, valb);
            //double valar = vala * cos(theta) - valb * sin(theta);
            //double valbr = vala * sin(theta) + valb * cos(theta);
            //printf("    OUT [%f %f]\n", valar, valbr);


            // apply rotation between modes numbers modei and iB
            //
            printf("rotation %d %d  angle %f\n", iB, modei, theta);
            for(uint32_t ii=0; ii<Bdim; ii++)
            {
                // modei
                double va = imginAB.im->array.F[iB*Bdim    + ii];
                double vb = imginAB.im->array.F[modei*Bdim + ii];

                double var = va * cos(theta) - vb * sin(theta);
                double vbr = va * sin(theta) + vb * cos(theta);

                imginAB.im->array.F[iB*Bdim    + ii] = var;
                imginAB.im->array.F[modei*Bdim + ii] = vbr;
            }

            for(uint32_t ii=0; ii<Adim; ii++)
            {
                // apply rotation to rotation matrix
                double va = Arot[iB*Adim + ii];
                double vb = Arot[modei*Adim + ii];
                double var = va * cos(theta) - vb * sin(theta);
                double vbr = va * sin(theta) + vb * cos(theta);
                Arot[iB*Adim + ii] = var;
                Arot[modei*Adim + ii] = vbr;
            }

            //valb = imginAB.im->array.F[bindex];
            //printf("       %f\n", valb);

            modei --;
        }

    }


    // Create output
    //
    imgArot->naxis = 2;
    imgArot->size[0] = Adim;
    imgArot->size[1] = Adim;
    imgArot->datatype = _DATATYPE_FLOAT;
    createimagefromIMGID(imgArot);
    for(uint64_t ii = 0; ii < Adim*Adim; ii++ )
    {
        imgArot->im->array.F[ii] = Arot[ii];
    }

    free(Arot);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}






static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginAB = mkIMGID_from_name(inmatAB);
    resolveIMGID(&imginAB, ERRMODE_ABORT);


    IMGID imgoutArot  = mkIMGID_from_name(outmatArot);


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {


        compute_basis_rotate_match(imginAB, &imgoutArot);


    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END



    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions




// Register function in CLI
errno_t
CLIADDCMD_linalgebra__basis_rotate_match()
{

    //CLIcmddata.FPS_customCONFsetup = customCONFsetup;
    //CLIcmddata.FPS_customCONFcheck = customCONFcheck;
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

