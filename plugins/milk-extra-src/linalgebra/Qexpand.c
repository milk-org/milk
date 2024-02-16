/**
 * @file PCAmatch.c
 *
 * @brief match two PCA decompositions
 *
 * Find corresponding linear combination across two basis
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"

#include "SGEMM.h"



static char *incoeffM;
static long  fpi_incoeffM;

static char *outcoeffM;
static long  fpi_outcoeffM;

static int32_t *axis;
static long     fpi_axis;






static CLICMDARGDEF farg[] =
{
    {
        CLIARG_IMG,
        ".incoeffM",
        "input coeffs matrix",
        "inM",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &incoeffM,
        &fpi_incoeffM
    },
    {
        CLIARG_STR,
        ".outcoeffM",
        "output coeffs matrix",
        "outcA",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &outcoeffM,
        &fpi_outcoeffM
    },
    {
        CLIARG_INT32,
        ".axis",
        "axis",
        "0",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &axis,
        &fpi_axis
    }
};





static CLICMDDATA CLIcmddata =
{
    "Qexpand", "quadractic expansion of vector or matrix coeffs", CLICMD_FIELDS_DEFAULTS
};



static errno_t help_function()
{


    return RETURN_SUCCESS;
}




/**
 * @brief Quadratic expansion of vector(s)
 *
 *
 * Adds constant in front of vector, and quadractic terms behind.
 * For example, vector [a,b] -> [1,a,b,aa,ab,bb]
 *
 * @param incoeffM   input coeffs
 * @param outcoeffM  output coeffs
 * @param axis       expansion axis
 * @return errno_t
 *
 * Only handles 2D images
 */
errno_t Qexpand(
    IMGID    imgincoeffM,
    IMGID    *imgoutcoeffM,
    int      axis
)
{
    DEBUG_TRACE_FSTART();

    int vecindexaxis = 1;
    if ( axis == 1 )
    {
        vecindexaxis = 0;
    }
    uint32_t vecdim = imgincoeffM.md->size[axis];
    uint32_t NBvec = imgincoeffM.md->size[vecindexaxis];

    printf("axis = %d\n", axis);
    printf("vecdim = %u\n", vecdim);
    printf("NBvec = %u\n", NBvec);
    fflush(stdout);


    // vecdimout is output vector dimension
    //
    uint32_t vecdimout = 1 + vecdim + vecdim*(vecdim+1)/2;




    // output vectors
    imgoutcoeffM->datatype = _DATATYPE_FLOAT;
    imgoutcoeffM->naxis = 2;

    imgoutcoeffM->size[axis] = vecdimout;
    imgoutcoeffM->size[vecindexaxis] = NBvec;

    printf("CREATING %s\n", imgoutcoeffM->name);
    fflush(stdout);
    createimagefromIMGID(imgoutcoeffM);

    {
        FILE *fp = fopen("Qmode.def.txt", "w");
        fprintf(fp, "0   \n");
        for(uint32_t ii=0; ii<vecdim; ii++)
        {
            fprintf(fp, "%05u   %05u\n", ii+1, ii);
        }
        uint32_t index = vecdim + 1;
        for(uint32_t ii=0; ii<vecdim; ii++)
        {
            for(uint32_t jj=ii; jj<vecdim; jj++)
            {
                fprintf(fp, "%05d   %05u  %05u\n", index, ii, jj);
                index ++;
            }
        }
        fclose(fp);
    }



    for( uint32_t vec=0; vec<NBvec; vec++)
    {
        // copy linear part
        if(axis == 0)
        {
            imgoutcoeffM->im->array.F[vec*vecdimout] = 1.0;
            for(uint32_t ii=0; ii<vecdim; ii++)
            {
                imgoutcoeffM->im->array.F[vec*vecdimout + ii + 1] = imgincoeffM.im->array.F[vec*vecdim + ii];
            }
        }
        else
        {
            imgoutcoeffM->im->array.F[vec] = 1.0;
            for(uint32_t ii=0; ii<vecdim; ii++)
            {
                imgoutcoeffM->im->array.F[(ii+1)*NBvec + vec] = imgincoeffM.im->array.F[ii*NBvec + vec];
            }
        }

        uint32_t index = vecdim + 1;

        for(uint32_t ii=0; ii<vecdim; ii++)
        {
            for(uint32_t jj=ii; jj<vecdim; jj++)
            {
                if(axis == 0)
                {
                    float valii = imgincoeffM.im->array.F[vec*vecdim + ii];
                    float valjj = imgincoeffM.im->array.F[vec*vecdim + jj];
                    imgoutcoeffM->im->array.F[vec*vecdimout + index] = valii * valjj;
                }
                else
                {
                    float valii = imgincoeffM.im->array.F[ii*NBvec + vec];
                    float valjj = imgincoeffM.im->array.F[jj*NBvec + vec];
                    imgoutcoeffM->im->array.F[index*NBvec + vec] = valii * valjj;
                }
                index ++;
            }
        }
    }



    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}






static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imgincoeffM = mkIMGID_from_name(incoeffM);
    resolveIMGID(&imgincoeffM, ERRMODE_ABORT);


    fflush(stdout);
    IMGID imgoutcoeffM  = mkIMGID_from_name(outcoeffM);




    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {

        Qexpand(
            imgincoeffM,
            &imgoutcoeffM,
            *axis
        );

        processinfo_update_output_stream(processinfo, imgoutcoeffM.ID);
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END



    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




INSERT_STD_FPSCLIfunctions


errno_t CLIADDCMD_linalgebra__Qexpand()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}

