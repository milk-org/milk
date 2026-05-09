/**
 * @file basis_rotate_match.c
 * @brief Basis rotate match module
 */

/**
 * @file SGEMM.c
 *
 */

#include <math.h>

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits/COREMOD_iofits.h"

#include "COREMOD_tools/COREMOD_tools.h"


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name    = "basisrotmatch",
    .cmdkey      = "basisrotmatch",
    .description = "rotate modal basis to fit modes",
    .description_long =
        "Rotate a modal basis to best fit a set of target modes using least-squares matching."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char * inmatAB = NULL;
static char * outmatArot = NULL;
static uint32_t * optmode = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X) \
    X(".matAB", &inmatAB, \
      FPTYPE_STREAMNAME, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "input decomposition of modes B in basis A") \
    X(".matArot", &outmatArot, \
      FPTYPE_STRING, 1, \
      FPFLAG_DEFAULT_INPUT, \
      "output rotation matrix")


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)

errno_t compute_basis_rotate_match(
    IMGID imginAB,
    IMGID *imgArot,
    int optmode
)
{
    DEBUG_TRACE_FSTART();

    int Adim = imginAB.md->size[1];
    int Bdim = imginAB.md->size[0];


    // internal Arot array, double for improved precision
    //
    double * Arot = (double *) malloc(sizeof(double) * Adim * Adim );


    // internal copy of imginAB, double for improved precision
    //
    double * matAB = (double *) malloc(sizeof(double) * Adim * Bdim);


    // loop stop condition: toggles to 0 when done
    int loopOK = 1;

    // diagonal values
    //
    double * diagVal = (double*) malloc(sizeof(double)*Adim);
    double diagVal_lim = 0.0;
    int loopiter = 0;

    int loopiterMax = 0;
    if(optmode == 0)
    {
        loopiterMax = 0;
    }
    else
    {
        loopiterMax = 10;
    }

    double diagVal_lim_step;


    // lower triangular
    //

    while( loopOK == 1 )
    {

        // Initialize: set Arot to identity matrix
        //
        for(uint64_t ii=0; ii<Adim*Adim; ii++)
        {
            Arot[ii] = 0.0;
        }
        for(uint32_t ii=0; ii<Adim; ii++)
        {
            Arot[Adim*ii + ii] = 1.0;
        }

        // copy input to internal buffer
        for(uint64_t ii=0; ii<Adim*Bdim; ii++)
        {
            matAB[ii] = imginAB.im->array.F[ii];
        }


        // counters
        int skipcnt = 0; // skipped
        int incrcnt = 0; // incremented
        int proccnt = 0; // processed


        // loop over target vectors
        int m1 = 0;
        for( int iB = 0; iB < Bdim; iB++)
        {
            //printf("   %5d  %5d    %5d  ", iB, m1, Adim-1-m1);
            // start from last mode
            int modei = Adim-1;

            // i0 is target vector index
            // m1 is goal mode

            // to be maximized
            // diagonal element if not skipping (m1 = i0)
            //
            int aindex;
            aindex = m1*Bdim + iB;


            int procflag = 0; // toggles to 1 if processed
            while(modei > m1)
            {
                procflag = 1;
                //printf(".");
                double vala = matAB[aindex];

                // to be minimized
                int bindex = modei*Bdim + iB;


                double valb = matAB[bindex];


                // rotation angle
                //
                double theta = atan2(-valb, vala);


                // apply rotation between modes numbers modei and i0
                //
                //printf("rotation %d %d  angle %f\n", iB, modei, theta);
                for(uint32_t ii=0; ii<Bdim; ii++)
                {
                    // modei
                    double va = matAB[m1*Bdim    + ii];
                    double vb = matAB[modei*Bdim + ii];

                    double var = va * cos(theta) - vb * sin(theta);
                    double vbr = va * sin(theta) + vb * cos(theta);

                    matAB[m1*Bdim    + ii] = var;
                    matAB[modei*Bdim + ii] = vbr;
                }


                for(uint32_t ii=0; ii<Adim; ii++)
                {
                    // apply rotation to rotation matrix
                    double va = Arot[m1*Adim + ii];
                    double vb = Arot[modei*Adim + ii];
                    double var = va * cos(theta) - vb * sin(theta);
                    double vbr = va * sin(theta) + vb * cos(theta);
                    Arot[m1*Adim + ii] = var;
                    Arot[modei*Adim + ii] = vbr;
                }


                modei --;
            }


            if( (procflag == 1) && (m1 < Adim-1) )
            {
                diagVal[m1] = matAB[aindex];
                proccnt ++;
            }

            if ( (fabs(matAB[aindex]) > diagVal_lim ) && (m1 < Adim-1) )
            {
                m1 ++;
                incrcnt ++;
            }
            else
            {
                //printf("   skip %3d   (%3d x %3d)   %f\n", skipcnt, m1, iB, matAB[aindex]);
                if( m1 < Adim-1)
                {
                    skipcnt ++;
                }
            }

        }


        printf("%9.6f  incremented %d, skipped %d  processed %d  (Bsize = %d) \n",
               diagVal_lim, incrcnt, skipcnt, proccnt, Bdim);
        if(loopiter == 0)
        {
            quick_sort_double(diagVal, incrcnt);
            printf("    median = %f\n", diagVal[incrcnt/2]);
            diagVal_lim = diagVal[incrcnt/2];
            diagVal_lim_step = 0.5*diagVal_lim;
        }

        if(proccnt < Bdim)
        {
            diagVal_lim += diagVal_lim_step;
        }
        else
        {
            diagVal_lim -= diagVal_lim_step;
        }


        diagVal_lim_step *= 0.6;


        loopiter ++;

        if(loopiter >= loopiterMax)
        {
            loopOK = 0;
        }
    }

    free(diagVal);


    if(optmode == 2)
    {
        loopiter = 0;
        loopiterMax = 100;

        double alphap = 1.0;

        double dangle = 1.0;
        double danglegain = 0.8;

        double danglemin = 0.001;
        double danglemfact = 0.7;

        double negSideAmp = 0.0;
        double posSideAmp = 1.0;

        // temp storate for vects to be swapped

        double *  n0arraypos = (double *) malloc(sizeof(double) * Bdim);
        double *  n1arraypos = (double *) malloc(sizeof(double) * Bdim);

        double *  n0arrayneg = (double *) malloc(sizeof(double) * Bdim);
        double *  n1arrayneg = (double *) malloc(sizeof(double) * Bdim);


        // effective B index of each A mode
        // tracks location of diagonal
        double * AmodeBeff = (double *) malloc(sizeof(double) * Adim);
        // initialized to be straight diagonal
        for( int ii=0; ii<Adim; ii++)
        {
            AmodeBeff[ii] = 1.0*ii;
        }


        while ( ( loopiter < loopiterMax ) && ( dangle > danglemin) )
        {
            long cntpos = 0;
            long cntneg = 0;
            long cntmid = 0;


            // measure quality metric (optall)
            //
            double optall = 0.0;
            for( int iia = 0; iia < Adim; iia++ )
            {
                // x0 generally > 1
                // x0 tracks diagonal
                double x0 = AmodeBeff[iia] / Adim;

                for( int iib = 0; iib < Bdim; iib++ )
                {
                    // dx0 is distance to "diagonal"
                    double x  = 1.0*iib / Adim;
                    double dx0 = x-x0;


                    double dcoeff = pow(dx0*dx0, alphap);
                    if( dx0 > 0.0 )
                    {
                        dcoeff *= posSideAmp;
                    }
                    else
                    {
                        dcoeff *= negSideAmp;
                    }
                    optall += dcoeff * matAB[iia*Bdim + iib] * dcoeff * matAB[iia*Bdim + iib];
                }
            }
            printf("iter %4d / %4d   dangle = %f / %f  val = %g\n", loopiter, loopiterMax, dangle, danglemin, optall);


            for ( int n0 = 0; n0 < Adim; n0++)
            {
                for ( int n1 = n0+1; n1 < Adim; n1++ )
                {
                    // testing rotation n0 n1, dangle

                    // ref value
                    // sum of optval0 and optval1 to be minimized
                    double optval0  = 0.0;
                    double optval1  = 0.0;

                    double optvalpos0  = 0.0;
                    double optvalpos1  = 0.0;

                    double optvalneg0  = 0.0;
                    double optvalneg1  = 0.0;


                    for(uint32_t ii=0; ii<Bdim; ii++)
                    {
                        double x  = 1.0*ii / Adim;
                        double x0 = AmodeBeff[n0] / Adim;
                        double x1 = AmodeBeff[n1] / Adim;


                        double dx0 = x-x0;
                        double dx1 = x-x1;


                        double dcoeff0 = pow(dx0*dx0, alphap);
                        double dcoeff1 = pow(dx1*dx1, alphap);

                        double v0 = matAB[n0*Bdim + ii];
                        double v1 = matAB[n1*Bdim + ii];

                        double wcoeff0 = 0.0;
                        double wcoeff1 = 0.0;

                        if( dx0 > 0.0 )
                        {
                            dcoeff0 *= posSideAmp;
                            wcoeff0 += posSideAmp * fabs(v0);
                        }
                        else
                        {
                            dcoeff0 *= negSideAmp;
                            wcoeff0 += negSideAmp * fabs(v0);
                        }

                        if( dx1 > 0.0 )
                        {
                            dcoeff1 *= posSideAmp;
                            wcoeff1 += posSideAmp * fabs(v1);
                        }
                        else
                        {
                            dcoeff1 *= negSideAmp;
                            wcoeff1 += negSideAmp * fabs(v1);
                        }

                        //wcoeff = pow(wcoeff, 4.0);
                        wcoeff0 = 1.0;
                        wcoeff1 = 1.0;


                        // optimization metric without rotation
                        optval0 += wcoeff0 * dcoeff0 * v0*v0;
                        optval1 += wcoeff1 * dcoeff1 * v1*v1;

                        // perform rotation, weite to n0array and n1array
                        n0arraypos[ii] = v0 * cos(dangle) - v1 * sin(dangle);
                        n1arraypos[ii] = v0 * sin(dangle) + v1 * cos(dangle);

                        n0arrayneg[ii] = v0 * cos(dangle) + v1 * sin(dangle);
                        n1arrayneg[ii] = - v0 * sin(dangle) + v1 * cos(dangle);

                        // optimization metric with positive rotation
                        optvalpos0 += wcoeff0 * dcoeff0 * n0arraypos[ii] * n0arraypos[ii];
                        optvalpos1 += wcoeff1 * dcoeff1 * n1arraypos[ii] * n1arraypos[ii];

                        // optimization metric with negative rotation
                        optvalneg0 += wcoeff0 * dcoeff0 * n0arrayneg[ii] * n0arrayneg[ii];
                        optvalneg1 += wcoeff1 * dcoeff1 * n1arrayneg[ii] * n1arrayneg[ii];

                    }
                    double optval = optval0 + optval1;
                    double optvalneg = optvalneg0 + optvalneg1;
                    double optvalpos = optvalpos0 + optvalpos1;


                    //printf("     [%3d - %3d]  %g  %g  %g\n", n0, n1, optvalneg, optval, optvalpos);

                    double optrotangle = 0.0;
                    if(optvalneg < optval)
                    {
                        // rotate neg
                        optrotangle = -dangle * danglegain;
                        cntneg++;
                    }
                    else if(optvalpos < optval)
                    {
                        // rotate pos
                        optrotangle = dangle * danglegain;
                        cntpos++;
                    }
                    else
                    {
                        // figure out optrotangle
                        // model: parabola
                        // input: 3 points, at x = -1, 0 and +1
                        // reference y=0 for point at x = 0
                        // -> 2 inputs, vpos and vneg
                        //
                        // parabola : y = a(x0-x)^2 + b
                        double vpos = optvalpos - optval;
                        double vneg = optvalneg - optval;

                        double a = (vpos+vneg)/2.0;
                        optrotangle = (vneg-vpos)/(4.0*a) * dangle;


                        if( optrotangle > dangle )
                        {
                            optrotangle = dangle * danglegain;
                        }
                        if( optrotangle < -dangle )
                        {
                            optrotangle = -dangle * danglegain;
                        }

                        //optrotangle = 0.0;
                        cntmid++;
                    }


                    // apply rotation between n0 and n1

                    for(uint32_t ii=0; ii<Bdim; ii++)
                    {
                        // modei
                        double va = matAB[n0*Bdim + ii];
                        double vb = matAB[n1*Bdim + ii];

                        double var = va * cos(optrotangle) - vb * sin(optrotangle);
                        double vbr = va * sin(optrotangle) + vb * cos(optrotangle);

                        matAB[n0*Bdim + ii] = var;
                        matAB[n1*Bdim + ii] = vbr;
                    }

                    for(uint32_t ii=0; ii<Adim; ii++)
                    {
                        // apply rotation to rotation matrix
                        double va = Arot[n0*Adim + ii];
                        double vb = Arot[n1*Adim + ii];
                        double var = va * cos(optrotangle) - vb * sin(optrotangle);
                        double vbr = va * sin(optrotangle) + vb * cos(optrotangle);
                        Arot[n0*Adim + ii] = var;
                        Arot[n1*Adim + ii] = vbr;
                    }
                }

            }
            printf("     [%5ld  %5ld  %5ld] %8.6f -> %8.6f\n", cntneg, cntmid, cntpos, 1.0*cntmid/(cntmid+cntpos+cntneg), dangle);
            if( 1.0*cntmid/(cntmid+cntpos+cntneg) > 0.999 )
            {
                dangle *= danglemfact;
            }


            // Measure, for each A mode, the effective index of B modes
            // AmodeBeff[iia] has to be > iia
            //
            long * iarray = (long *) malloc(sizeof(long) * Adim);
            for( int iia = 0; iia < Adim; iia++ )
            {
                iarray[iia] = iia;
                double Beff = 0.0;
                double Beffcnt = 0.0;
                for(uint32_t iib=0; iib<Bdim; iib++)
                {
                    double a = matAB[iia*Bdim + iib] * matAB[iia*Bdim + iib];
                    double ap = a*a;
                    Beff += 1.0*iib * ap;
                    Beffcnt += ap;
                }
                AmodeBeff[iia] = Beff/Beffcnt;
            }
            // sort by effective index
            quick_sort2l(AmodeBeff, iarray, Adim);


            {
                char fname[STRINGMAXLEN_FILENAME];
                WRITE_FILENAME(fname, "./compfCM/Beff.%04d.dat", loopiter);
                FILE * fpBeff = fopen(fname, "w");
                for( int iia = 0; iia < Adim; iia++ )
                {
                    if(AmodeBeff[iia] < iia)
                    {
                        AmodeBeff[iia] = iia;
                    }
                    fprintf(fpBeff, "%4d %16f  %4ld\n", iia, AmodeBeff[iia], iarray[iia]);
                }
                fclose(fpBeff);
            }


            if ( loopiter == loopiterMax-1 )
            {
                // re-order A modes
                // allocate temporary array
                double * tmpmatAB = (double *) malloc(sizeof(double)*Adim*Bdim);
                memcpy(tmpmatAB, matAB, sizeof(double)*Adim*Bdim);
                for( int iia = 0; iia < Adim; iia++ )
                {
                    memcpy( (char *) tmpmatAB + sizeof(double)*iia*Bdim,
                            (char *) matAB + sizeof(double)*iarray[iia]*Bdim,
                            sizeof(double)*Bdim);
                }
                memcpy(matAB, tmpmatAB, sizeof(double)*Adim*Bdim);
                free(tmpmatAB);
            }

            free(iarray);


            loopiter ++;
        }

        free(AmodeBeff);

        free(n0arraypos);
        free(n1arraypos);

        free(n0arrayneg);
        free(n1arrayneg);

    }


// Create output
//
    imgArot->mdt->naxis = 2;
    imgArot->mdt->size[0] = Adim;
    imgArot->mdt->size[1] = Adim;
    imgArot->mdt->datatype = _DATATYPE_FLOAT;
    createimagefromIMGID(imgArot);
    for(uint64_t ii = 0; ii < Adim*Adim; ii++ )
    {
        imgArot->im->array.F[ii] = Arot[ii];
    }

    free(Arot);

//copy matAB to ouput
    for(uint64_t ii = 0; ii<Adim*Bdim; ii++)
    {
        imginAB.im->array.F[ii] = matAB[ii];
    }

    free(matAB);


    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


static MILK_HOT errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    IMGID imginAB = imgid_make_from_name(inmatAB);
    resolveIMGID(&imginAB, ERRMODE_WARN, dcimg, dcnimg);


    IMGID imgoutArot  = imgid_make_from_name(outmatArot);
    if (imginAB.ID == -1) {
        return RETURN_FAILURE;
    }


    INSERT_STD_PROCINFO_COMPUTEFUNC_INIT


    INSERT_STD_PROCINFO_COMPUTEFUNC_LOOPSTART
    {


        compute_basis_rotate_match(imginAB, &imgoutArot, *optmode);


    }
    INSERT_STD_PROCINFO_COMPUTEFUNC_END

    imgid_free(&imginAB);
    imgid_free(&imgoutArot);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


/* ================================================================
 * 7.  MILK MODULE REGISTRATION
 * ============================================================= */

#ifndef FPS_STANDALONE
static errno_t CLIfunction(void)
{
    return safe_fps_generic_CLIfunction(
        &FPS_app_info, farg, &CLIcmddata,
        my_bindings, nb_bindings,
        compute_function);
}

errno_t
CLIADDCMD_linalgebra__basis_rotate_match()
{
    safe_fps_fill_farg_examples(
        farg, my_bindings, nb_bindings);
    INSERT_STD_CLIREGISTERFUNC
    return RETURN_SUCCESS;
}
#endif


/* ================================================================
 * 8.  STANDALONE ENTRY POINT
 * ============================================================= */

#ifdef FPS_STANDALONE
FPS_MAIN_STANDALONE_V2(
    FPS_app_info,
    FPS_PARAMS,
    compute_function)
#endif

