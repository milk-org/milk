/**
 * @file SGEMM.c
 *
 */

#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "COREMOD_iofits/COREMOD_iofits.h"

#include "COREMOD_tools/COREMOD_tools.h"


static char *inmatAB;
static long  fpi_inmatAB;

static char *outmatArot;
static long  fpi_outmatArot;

static uint32_t *optmode;
static long fpi_optmode;


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
    },
    {
        CLIARG_UINT32,
        ".optmode",
        "optimization mode",
        "0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &optmode,
        &fpi_optmode
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
    printf("\n");
    printf("NOTATIONS :\n");
    printf("modal basis A has N modes, (n = 0...N-1)\n");
    printf("vector set B has M modes, (m = 0...M-1)\n");
    printf("B decompositions on A : C matrix, coeffs c(n,m)");
    printf("\n");
    printf("Optimization Mode:\n");
    printf("   0 (default) : B mode #m is linear combination of first m modes A\n");
    printf("       c(n,m)=0 if n>m");
    printf("       C matrix is triangular\n");
    printf("       Assumes B modes have no null space\n");
    printf("   1 (skipmatch): skip null space in B\n");
    printf("   2 (linforce) try to force match between n/N and m/N\n");

    return RETURN_SUCCESS;
}


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

        int TriangMode = 0; // lower triang
        if(optmode == 3)
        {
            TriangMode = 1; // upper triang
        }





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
        loopiterMax = 1000;

        double alphap = 1.0;
        double dangle = 1.0;
        double posSideAmp = 10.0;

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


        while ( loopiter < loopiterMax )
        {
            long cntpos = 0;
            long cntneg = 0;
            long cntmid = 0;

            double optall = 0.0;
            for( int iia = 0; iia < Adim; iia++ )
            {
                double x0 = AmodeBeff[iia] / Adim;
                for( int iib = 0; iib < Bdim; iib++ )
                {
                    double x  = 1.0*iib / Adim;
                    double dx0 = x-x0;
                    double dcoeff = pow(dx0*dx0, alphap);
                    if( dx0 > 0.0 )
                    {
                        dcoeff *= posSideAmp;
                    }
                    optall += dcoeff * matAB[iia*Bdim + iib] * dcoeff * matAB[iia*Bdim + iib];
                }
            }
            printf("iter %4d / %4d   dangle = %f  val = %g\n", loopiter, loopiterMax, dangle, optall);

            for ( int n0 = 0; n0 < Adim; n0++)
            {
                for ( int n1 = n0+1; n1 < Adim; n1++ )
                {
                    // testing rotation n0 n1, dangle

                    // ref value
                    double optval0  = 0.0; // to be minimized
                    double optval1  = 0.0; // to be minimized

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

                        if( dx0 > 0.0 )
                        {
                            dcoeff0 *= posSideAmp;
                        }
                        if( dx1 > 0.0 )
                        {
                            dcoeff1 *= posSideAmp;
                        }



                        double v0 = matAB[n0*Bdim + ii];
                        double v1 = matAB[n1*Bdim + ii];

                        // optimization metric without rotation
                        optval0 += dcoeff0 * v0*v0;
                        optval1 += dcoeff1 * v1*v1;

                        // perform rotation, weite to n0array and n1array
                        n0arraypos[ii] = v0 * cos(dangle) - v1 * sin(dangle);
                        n1arraypos[ii] = v0 * sin(dangle) + v1 * cos(dangle);

                        n0arrayneg[ii] = v0 * cos(dangle) + v1 * sin(dangle);
                        n1arrayneg[ii] = - v0 * sin(dangle) + v1 * cos(dangle);

                        // optimization metric with positive rotation
                        optvalpos0 += dcoeff0 * n0arraypos[ii] * n0arraypos[ii];
                        optvalpos1 += dcoeff1 * n1arraypos[ii] * n1arraypos[ii];

                        // optimization metric with negative rotation
                        optvalneg0 += dcoeff0 * n0arrayneg[ii] * n0arrayneg[ii];
                        optvalneg1 += dcoeff1 * n1arrayneg[ii] * n1arrayneg[ii];

                    }
                    double optval = optval0 + optval1;
                    double optvalneg = optvalneg0 + optvalneg1;
                    double optvalpos = optvalpos0 + optvalpos1;


                    //printf("     [%3d - %3d]  %g  %g  %g\n", n0, n1, optvalneg, optval, optvalpos);

                    double optrotangle = 0.0;
                    if(optvalneg < optval)
                    {
                        // rotate neg
                        optrotangle = -dangle;
                        cntneg++;
                    }
                    else if(optvalpos < optval)
                    {
                        // rotate pos
                        optrotangle = dangle;
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
                            optrotangle = dangle;
                        }
                        if( optrotangle < -dangle )
                        {
                            optrotangle = -dangle;
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
            printf("     [%5ld  %5ld  %5ld]  %g\n", cntneg, cntmid, cntpos, dangle);
            if( cntmid > 100.0*(cntpos+cntneg) )
            {
                dangle *= 0.98;
            }





            // Measure, for each A mode, the effective index of B modes
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
                WRITE_FILENAME(fname, "Beff.%04d.dat", loopiter);
                FILE * fpBeff = fopen(fname, "w");
                for( int iia = 0; iia < Adim; iia++ )
                {
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

//copy matAB to ouput
    for(uint64_t ii = 0; ii<Adim*Bdim; ii++)
    {
        imginAB.im->array.F[ii] = matAB[ii];
    }

    free(matAB);


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


        compute_basis_rotate_match(imginAB, &imgoutArot, *optmode);


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

