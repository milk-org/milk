/**
 * @file SingularValueDecomp.c
 *
 */

#include <math.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits/COREMOD_iofits.h"

#include "timeutils.h"

#include "SingularValueDecomp.h"
#include "SGEMM.h"


// CPU mode: Use MKL if available
// Otherwise use openBLAS
//

#ifdef HAVE_MKL
#    include "mkl.h"
#    include "mkl_lapacke.h"
#    define BLASLIB "IntelMKL"
#else
#    ifdef HAVE_OPENBLAS
#        include <cblas.h>
#        include <lapacke.h>
#        define BLASLIB "OpenBLAS"
#    else
#        include <lapacke.h>
#        define BLASLIB "Lapacke standalone"
#    endif
#endif


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "compSVD",
    .cmdkey           = "compSVD",
    .description      = "compute SVD",
    .description_long = "Compute the Singular Value Decomposition (SVD) of a matrix. Factorizes M "
                        "= U * S * V^T using LAPACK routines."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *inM       = NULL;
static char     *outU      = NULL;
static char     *outS      = NULL;
static char     *outV      = NULL;
static uint32_t *Vdim0     = NULL;
static float    *svdlim    = NULL;
static uint32_t *maxNBmode = NULL;
static int32_t  *GPUdevice = NULL;
static uint64_t *compmode  = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".outS", &outS, FPTYPE_STRING, 1, FPFLAG_DEFAULT_INPUT, "output ingular values")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

/**
 * @brief Compute SVD of indimM x indimN matrix
 *
 * Decompose matrix imgin as:
 * imgU imgeigenval imgV^T
 *
 * Using column-major indexing
 *
 *
 *
 * compSVDmode flags:
 * COMPSVD_SKIP_BIGMAT  skip big (U of V) matrix computation
 */
errno_t compute_SVD(IMGID    imgin,
                    IMGID   *imgU,
                    IMGID   *imgS,
                    IMGID   *imgV,
                    uint32_t Vdim0,
                    float    SVlimit,
                    uint32_t SVDmaxNBmode,
                    int      GPUdev,
                    uint64_t compSVDmode,
                    char    *SVDunmodesname,
                    char    *SVDvnmodesname)
{
    DEBUG_TRACE_FSTART();

    // check if images already exist
    //
    resolveIMGID(&imgin, ERRMODE_WARN, dcimg, dcnimg);
    resolveIMGID(imgU, ERRMODE_NULL, dcimg, dcnimg);
    if (imgin.ID == -1)
    {
        return RETURN_FAILURE;
    }
    resolveIMGID(imgS, ERRMODE_NULL, dcimg, dcnimg);
    resolveIMGID(imgV, ERRMODE_NULL, dcimg, dcnimg);


    // input dimensions
    // input matrix is inMdim x inNdim, column-major
    //
    int inNdim, inNdim0, inNdim1;
    int inMdim, inMdim0, inMdim1;

    if (imgin.md->naxis == 3)
    {
        printf("inMdim   : %d x %d\n", imgin.md->size[0], imgin.md->size[1]);
        inMdim  = imgin.md->size[0] * imgin.md->size[1];
        inMdim0 = imgin.md->size[0];
        inMdim1 = imgin.md->size[1];

        printf("inNdim    : %d\n", imgin.md->size[2]);
        inNdim  = imgin.md->size[2];
        inNdim0 = imgin.md->size[2];
        inNdim1 = 1;
    }
    else
    {
        printf("inMdim   : %d\n", imgin.md->size[0]);
        inMdim  = imgin.md->size[0];
        inMdim0 = imgin.md->size[0];
        inMdim1 = 1;

        printf("inNdim    : %d\n", imgin.md->size[1]);
        inNdim  = imgin.md->size[1];
        inNdim0 = imgin.md->size[1];
        inNdim1 = 1;
    }


    // Orient matrix such that it is tall (M > N)
    //


    enum matrix_shape
    {
        inMshape_tall,
        inMshape_wide
    } mshape;

    uint32_t Mdim                          = 0;
    uint32_t Mdim0 __attribute__((unused)) = 0;
    uint32_t Mdim1 __attribute__((unused)) = 0;

    uint32_t Ndim                          = 0;
    uint32_t Ndim0 __attribute__((unused)) = 0;
    uint32_t Ndim1 __attribute__((unused)) = 0;

    if (inNdim < inMdim)
    {
        // input matrix is tall
        // this is the default
        // notations follow this case
        //
        DEBUG_TRACEPOINT("CASE inNdim < inMdim (tall)\n");
        mshape = inMshape_tall;

        Mdim  = inMdim;
        Mdim0 = inMdim0;
        Mdim1 = inMdim1;

        Ndim  = inNdim;
        Ndim0 = inNdim0;
        Ndim1 = inNdim1;
    }
    else
    {
        DEBUG_TRACEPOINT("CASE inNdim > inMdim (wide)\n");
        mshape = inMshape_wide;

        Mdim  = inNdim;
        Mdim0 = inNdim0;
        Mdim1 = inNdim1;

        Ndim  = inMdim;
        Ndim0 = inMdim0;
        Ndim1 = inMdim1;
    }

    DEBUG_TRACEPOINT("inNdim               = %d  (%d x %d)\n", inNdim, inNdim0, inNdim1);
    DEBUG_TRACEPOINT("inMdim               = %d  (%d x %d)\n", inMdim, inMdim0, inMdim1);

    //printf("  Ndim               = %d  (%d x %d)\n",   Ndim, Ndim0, Ndim1);
    //printf("  Mdim               = %d  (%d x %d)\n",   Mdim, Mdim0, Mdim1);


    // from here on, Mdim > Ndim

    // create eigenvalues array if needed
    if (imgS->ID == -1)
    {
        imgS->mdt->naxis   = 2;
        imgS->mdt->size[0] = Ndim;
        imgS->mdt->size[1] = 1;
        createimagefromIMGID(imgS);
    }


    IMGID imgATA;
    {
        // create ATA
        // note that this is AAT if inNdim > inMdim (inMshape_wide)
        //
        int TranspA = 1;
        int TranspB = 0;
        if (mshape == inMshape_wide)
        {
            TranspA = 0;
            TranspB = 1;
        }
        snprintf(imgATA.name, sizeof(imgATA.name), "%s", "ATA");
        computeSGEMM(imgin, imgin, &imgATA, TranspA, TranspB, GPUdev);
    }


    // singular vectors array, small dimension
    // matrix (U or V) is square
    //
    IMGID *imgmNsvec;
    float  svalmax;
    long   NBmode = 0;
    {
        // eigendecomposition
        //
        float *__restrict d = (float *) malloc(sizeof(float) * Ndim);
        float *__restrict e = (float *) malloc(sizeof(float) * Ndim);
        float *__restrict t = (float *) malloc(sizeof(float) * Ndim);

#ifdef HAVE_MKL
        mkl_set_interface_layer(MKL_INTERFACE_ILP64);
#endif

        LAPACKE_ssytrd(LAPACK_COL_MAJOR, 'U', Ndim, (float *) imgATA.im->array.F, Ndim, d, e, t);

        // Assemble Q matrix
        LAPACKE_sorgtr(LAPACK_COL_MAJOR, 'U', Ndim, imgATA.im->array.F, Ndim, t);

        // compute all eigenvalues and eivenvectors -> imgmV
        //
        //memcpy(imgmNsvec->im->array.F, imgATA.im->array.F, sizeof(float)*Ndim*Ndim);
        LAPACKE_ssteqr(LAPACK_COL_MAJOR, 'V', Ndim, d, e, imgATA.im->array.F, Ndim);


        // How many modes to keep ?
        svalmax = sqrt(d[Ndim - 1]);
        {
            long modecnt = 0;
            for (int k = 0; k < Ndim; k++)
            {
                if (sqrt(d[k]) > SVlimit * svalmax)
                {
                    modecnt++;
                }
            }
            NBmode = modecnt;
            if (modecnt > SVDmaxNBmode)
            {
                NBmode = SVDmaxNBmode;
            }
        }
        printf("KEEPING %ld MODES\n", NBmode);


        if (mshape == inMshape_tall)
        {
            imgmNsvec = imgV;

            if (imgV->ID == -1)
            {
                if (Vdim0 == 0)
                {
                    imgV->mdt->naxis   = 2;
                    imgV->mdt->size[0] = inNdim;
                    imgV->mdt->size[1] = NBmode; //inNdim;
                }
                else
                {
                    imgV->mdt->naxis   = 3;
                    imgV->mdt->size[0] = Vdim0;
                    imgV->mdt->size[1] = inNdim / Vdim0;
                    imgV->mdt->size[2] = NBmode; //inNdim;
                }
                createimagefromIMGID(imgV);
            }
        }
        else
        {
            imgmNsvec = imgU;

            if (imgU->ID == -1)
            {
                imgU->mdt->naxis = imgin.md->naxis;
                if (imgin.md->naxis == 3)
                {
                    imgU->mdt->size[0] = inMdim0;
                    imgU->mdt->size[1] = inMdim1;
                    imgU->mdt->size[2] = NBmode; //inMdim;
                }
                else
                {
                    imgU->mdt->size[0] = inMdim;
                    imgU->mdt->size[1] = NBmode; //inMdim;
                }
                createimagefromIMGID(imgU);
                printf("[%d] imgU Created ==============================\n", __LINE__);
                printf("[%d] imgU %s\n", __LINE__, imgU->name);
                printf("[%d] imgU %s\n", __LINE__, imgU->md->name);
            }
        }


        // re-order from largest to smallest
        for (int k = 0; k < NBmode; k++)
        {
            char *ptr0 = (char *) &imgATA.im->array.F[(Ndim - k - 1) * Ndim];
            char *ptr1 = (char *) &imgmNsvec->im->array.F[k * Ndim];

            memcpy(ptr1, ptr0, sizeof(float) * Ndim);

            imgS->im->array.F[k] = sqrtf(d[Ndim - k - 1]);
        }


        // store singular values
        delete_image_ID("SV", DELETE_IMAGE_ERRMODE_IGNORE);
        IMGID imgSV         = imgid_make_from_name("SV");
        imgSV.mdt->naxis    = 2;
        imgSV.mdt->datatype = _DATATYPE_FLOAT;
        imgSV.mdt->size[0]  = NBmode;
        imgSV.mdt->size[1]  = 1;
        createimagefromIMGID(&imgSV);
        for (int k = 0; k < NBmode; k++)
        {
            float sval           = imgS->im->array.F[k];
            imgSV.im->array.F[k] = sval;
        }


        // store inv of singular values
        delete_image_ID("SVinv", DELETE_IMAGE_ERRMODE_IGNORE);
        IMGID imgSVinv         = imgid_make_from_name("SVinv");
        imgSVinv.mdt->naxis    = 2;
        imgSVinv.mdt->datatype = _DATATYPE_FLOAT;
        imgSVinv.mdt->size[0]  = NBmode;
        imgSVinv.mdt->size[1]  = 1;
        createimagefromIMGID(&imgSVinv);
        for (int k = 0; k < NBmode; k++)
        {
            //float normfact = 0.0;
            float sval  = imgS->im->array.F[k];
            float svaln = sval / svalmax;
            if (svaln > SVlimit)
            {
                imgSVinv.im->array.F[k] = 1.0 / sval;
            }
            else
            {
                imgSVinv.im->array.F[k] = 0.0;
            }
        }

        free(d);
        free(e);
        free(t);
        imgid_free(&imgSV);
        imgid_free(&imgSVinv);

        // imgmNsvec is matV if inMshape_tall, matU if inMshape_wide
    }
    delete_image(&imgATA, DELETE_IMAGE_ERRMODE_EXIT);


    if (!(compSVDmode & COMPSVD_SKIP_BIGMAT))
    {
        // create mU (if inMshape_tall)
        // create mV (if inMshape_wide)
        // (only non-zero part allocated)
        //


        // Compute mU (only non-zero part allocated)
        //
        IMGID *imgmMsvec;
        {
            int TranspA = 0;
            int TranspB = 0;
            if (mshape == inMshape_wide)
            {
                TranspA = 1;
            }

            if (mshape == inMshape_tall)
            {
                computeSGEMM(imgin, *imgmNsvec, imgU, TranspA, TranspB, GPUdev);
                imgmMsvec = imgU;
            }
            else
            {
                computeSGEMM(imgin, *imgmNsvec, imgV, TranspA, TranspB, GPUdev);
                imgmMsvec = imgV;
            }
        }


        // normalize cols of imgmMsvec
        // Report number of modes kept
        //
        long SVkeptcnt = 0;
        for (uint32_t jj = 0; jj < NBmode; jj++)
        {
            float normfact = 0.0;
            float sval     = imgS->im->array.F[jj];
            float svaln    = sval / svalmax;
            if (svaln > SVlimit)
            {
                normfact = 1.0 / sval;
                SVkeptcnt++;
            }

            for (uint32_t ii = 0; ii < Mdim; ii++)
            {
                imgmMsvec->im->array.F[jj * Mdim + ii] *= normfact;
            }
        }
        printf("LIMIT = %g  - Keeping %ld / %u modes\n", SVlimit, SVkeptcnt, Ndim);


        // Compute pseudo-inverse
        //
        if ((compSVDmode & COMPSVD_COMP_PSINV))
        {
            // assumes tall matrix
            //
            IMGID imgmNsvec1 = imgid_make_from_name("matNtemp");
            if (imgmNsvec1.ID == -1)
            {
                imgmNsvec1.mdt->naxis = 2;

                imgmNsvec1.mdt->size[0] = Ndim;
                imgmNsvec1.mdt->size[1] = NBmode;

                createimagefromIMGID(&imgmNsvec1);
            }

            // multiply by inverse of singular values
            //
            for (uint32_t jj = 0; jj < NBmode; jj++)
            {
                float normfact = 0.0;
                float sval     = imgS->im->array.F[jj];
                float svaln    = sval / svalmax;
                if (svaln > SVlimit)
                {
                    normfact = 1.0 / sval;
                }

                for (uint32_t ii = 0; ii < Ndim; ii++)
                {
                    imgmNsvec1.im->array.F[jj * Ndim + ii] =
                        imgmNsvec->im->array.F[jj * Ndim + ii] * normfact;
                }
            }


            IMGID imgpsinv;
            {
                int TranspA = 0;
                int TranspB = 1;
                snprintf(imgpsinv.name, sizeof(imgpsinv.name), "%s", "psinv");
                computeSGEMM(imgmNsvec1, *imgmMsvec, &imgpsinv, TranspA, TranspB, GPUdev);

                delete_image(&imgmNsvec1, DELETE_IMAGE_ERRMODE_EXIT);
            }
            imgid_free(&imgmNsvec1);


            // Check inverse
            //
            if ((compSVDmode & COMPSVD_COMP_CHECKPSINV))
            {
                IMGID imgpsinvcheck = imgid_make_from_name("psinvcheck");
                if (mshape == inMshape_tall)
                {
                    // inNdim < inMdim
                    computeSGEMM(imgpsinv, imgin, &imgpsinvcheck, 0, 0, GPUdev);
                }
                imgid_free(&imgpsinvcheck);
            }
        }
    }

    // Compute un-normalized modes U
    // Singular Values included in modes U
    //
    if (imgU->ID != -1)
    {
        // un-normalized modes
        delete_image_ID(SVDunmodesname, DELETE_IMAGE_ERRMODE_IGNORE);
        IMGID imgunmodes         = imgid_make_from_name(SVDunmodesname);
        imgunmodes.mdt->naxis    = imgU->md->naxis;
        imgunmodes.mdt->datatype = imgU->md->datatype;
        imgunmodes.mdt->size[0]  = imgU->md->size[0];
        imgunmodes.mdt->size[1]  = imgU->md->size[1];
        imgunmodes.mdt->size[2]  = imgU->md->size[2];
        createimagefromIMGID(&imgunmodes);

        int  lastaxis  = imgunmodes.mdt->naxis - 1;
        long framesize = imgunmodes.mdt->size[0];
        if (lastaxis == 2)
        {
            framesize *= imgunmodes.mdt->size[1];
        }

        for (int kk = 0; kk < imgunmodes.mdt->size[lastaxis]; kk++)
        {
            float mfact = imgS->im->array.F[kk];
            for (long ii = 0; ii < framesize; ii++)
            {
                imgunmodes.im->array.F[kk * framesize + ii] =
                    imgU->im->array.F[kk * framesize + ii] * mfact;
            }
        }

        delete_image_ID("SVDinrec", DELETE_IMAGE_ERRMODE_IGNORE);
        IMGID iminrec = imgid_make_from_name("SVDinrec");
        computeSGEMM(imgunmodes, *imgV, &iminrec, 0, 1, GPUdev);
        imgid_free(&imgunmodes);
        imgid_free(&iminrec);
    }


    // Compute un-normalized modes V
    // Singular Values included in modes V
    //
    if (imgV->ID != -1)
    {
        // un-normalized modes
        delete_image_ID(SVDvnmodesname, DELETE_IMAGE_ERRMODE_IGNORE);
        IMGID imgvnmodes         = imgid_make_from_name(SVDvnmodesname);
        imgvnmodes.mdt->naxis    = imgV->md->naxis;
        imgvnmodes.mdt->datatype = imgV->md->datatype;
        imgvnmodes.mdt->size[0]  = imgV->md->size[0];
        imgvnmodes.mdt->size[1]  = imgV->md->size[1];
        imgvnmodes.mdt->size[2]  = imgV->md->size[2];
        createimagefromIMGID(&imgvnmodes);

        int  lastaxis  = imgvnmodes.mdt->naxis - 1;
        long framesize = imgvnmodes.mdt->size[0];
        if (lastaxis == 2)
        {
            framesize *= imgvnmodes.mdt->size[1];
        }

        for (int kk = 0; kk < imgvnmodes.mdt->size[lastaxis]; kk++)
        {
            float mfact = imgS->im->array.F[kk];
            //printf("mfact %4d = %f\n", kk, mfact);
            for (long ii = 0; ii < framesize; ii++)
            {
                imgvnmodes.im->array.F[kk * framesize + ii] =
                    imgV->im->array.F[kk * framesize + ii] * mfact;
            }
        }

        //IMGID iminrec = imgid_make_from_name("SVDinrec");
        //computeSGEMM(imgvnmodes, imgV, &iminrec, 0, 1, GPUdev);
        imgid_free(&imgvnmodes);
    }


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginM = imgid_make_from_name(inM);
    resolveIMGID(&imginM, ERRMODE_WARN, dcimg, dcnimg);


    IMGID imgU = imgid_make_from_name(outU);
    if (imginM.ID == -1)
    {
        return RETURN_FAILURE;
    }
    IMGID imgS = imgid_make_from_name(outS);
    IMGID imgV = imgid_make_from_name(outV);

    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {
        compute_SVD(imginM, &imgU, &imgS, &imgV, *Vdim0, *svdlim, *maxNBmode, *GPUdevice, *compmode,
                    "SVDunmodes", "SVDvnmodes");
    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imginM);
    imgid_free(&imgU);
    imgid_free(&imgS);
    imgid_free(&imgV);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(&FPS_app_info, farg, &CLIcmddata, my_bindings, nb_bindings,
                                        compute_function);
}

errno_t CLIADDCMD_linalgebra__compSVD()
{
    safe_fps_fill_farg_examples(farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(FPS_app_info, FPS_PARAMS, compute_function)
#endif
