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
    double * __restrict Arot = (double *) malloc(sizeof(double) * Adim * Adim );


    // internal copy of imginAB, double for improved precision
    //
    double * __restrict matAB = (double *) malloc(sizeof(double) * Adim * Bdim);


    // loop stop condition: toggles to 0 when done
    int loopOK = 1;

    // diagonal values
    //
    double * __restrict diagVal = (double*) malloc(sizeof(double)*Adim);
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
            //printf("\n");



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
        loopiterMax = 20000;

        double alphap = 2.0;
        double alphalim = 0.05;

        double dangle = 0.5;

        // temp storate for vects to be swapped

        double * __restrict n0arraypos = (double *) malloc(sizeof(double) * Bdim);
        double * __restrict n1arraypos = (double *) malloc(sizeof(double) * Bdim);

        double * __restrict n0arrayneg = (double *) malloc(sizeof(double) * Bdim);
        double * __restrict n1arrayneg = (double *) malloc(sizeof(double) * Bdim);


        while ( loopiter < loopiterMax)
        {

            long cntpos = 0;
            long cntneg = 0;
            long cntmid = 0;

            double optall = 0.0;
            for( int iia = 0; iia < Adim; iia++ )
            {
                double xa = 1.0*iia/Adim;
                for( int iib = 0; iib < Bdim; iib++ )
                {
                    double xb = 1.0*iib/Bdim;
                    double dcoeff = pow(fabs(xa-xb), alphap);
                    optall += dcoeff * fabs(matAB[iia*Bdim + iib]);
                }
            }
            printf("iter %4d  val = %g\n", loopiter, optall);

            for ( int n0 = 0; n0 < Adim; n0++) //Adim; n0++ )
            {
                for ( int n1 = n0+1; n1 < Adim; n1++ )
                {
                    // testing rotation n0 n1, dangle

                    // ref value
                    double optvalN0 = 0.0; // to be minimized
                    double optvalD0 = 0.0; // to be maximized
                    double optval0 = 0.0; // equal o optvalN/optvalD - ratio to be minimized

                    double optvalN1 = 0.0; // to be minimized
                    double optvalD1 = 0.0; // to be maximized
                    double optval1 = 0.0; // equal o optvalN/optvalD - ratio to be minimized



                    double optvalpos0 = 0.0;
                    double optvalNpos0 = 0.0;
                    double optvalDpos0 = 0.0;

                    double optvalpos1 = 0.0;
                    double optvalNpos1 = 0.0;
                    double optvalDpos1 = 0.0;



                    double optvalneg0 = 0.0;
                    double optvalNneg0 = 0.0;
                    double optvalDneg0 = 0.0;

                    double optvalneg1 = 0.0;
                    double optvalNneg1 = 0.0;
                    double optvalDneg1 = 0.0;




                    for(uint32_t ii=0; ii<Bdim; ii++)
                    {
                        double x = 1.0*ii/Bdim;
                        double y0 = 1.0*n0/Adim;
                        double y1 = 1.0*n1/Adim;

                        double v0 = matAB[n0*Bdim + ii];
                        double v1 = matAB[n1*Bdim + ii];

                        double dx0 = x-y0;
                        double dx1 = x-y1;

                        double dcoeff0 = 0.0;
                        //if ( ii > n0 )
                        if ( dx0 > 0 )
                        {
                            dcoeff0 = pow(dx0, alphap);
                            //dcoeff0 = ii-n0;
                        }
                        if(dcoeff0 > alphalim)
                        {
                            dcoeff0 = alphalim;
                        }


                        double dcoeff1 = 0.0;
                        //if ( ii > n1 )
                        if ( dx1 > 0 )
                        {
                            dcoeff1 = pow(dx1, alphap);
                            //dcoeff1 = ii-n1;
                        }
                        if(dcoeff1 > alphalim)
                        {
                            dcoeff1 = alphalim;
                        }



                        optvalN0 += dcoeff0 * v0*v0;
                        optvalN1 += dcoeff1 * v1*v1;

                        // maximize sum squared of coefficitents to the left (and including) diagonal
                        // this ensures the target modes are represented
                        //if(ii <= n0)
                        if ( dx0 < 0.0 )
                        {
                            optvalD0 += v0*v0;
                            //printf("    %4d  optval1D  += %g (%g)  -> %g\n", ii, v0*v0, v0, optvalD);
                        }
                        //if(ii <= n1)
                        if ( dx1 < 0.0 )
                        {
                            optvalD1 += v1*v1;
                            //printf("    %4d  optval1D  += %g (%g)  -> %g\n", ii, v1*v1, v1, optvalD);
                        }


                        // perform rotation, weite to n0array and n1array
                        n0arraypos[ii] = v0 * cos(dangle) - v1 * sin(dangle);
                        n1arraypos[ii] = v0 * sin(dangle) + v1 * cos(dangle);

                        n0arrayneg[ii] = v0 * cos(dangle) + v1 * sin(dangle);
                        n1arrayneg[ii] = - v0 * sin(dangle) + v1 * cos(dangle);

                        optvalNpos0 += dcoeff0 * n0arraypos[ii] * n0arraypos[ii];
                        optvalNpos1 += dcoeff1 * n1arraypos[ii] * n1arraypos[ii];
                        if ( dx0 < 0.0 )
                        {
                            optvalDpos0 += n0arraypos[ii] * n0arraypos[ii];
                        }
                        if ( dx1 < 0.0 )
                        {
                            optvalDpos1 += n1arraypos[ii] * n1arraypos[ii];
                        }


                        optvalNneg0 += dcoeff0 * n0arrayneg[ii] * n0arrayneg[ii];
                        optvalNneg1 += dcoeff1 * n1arrayneg[ii] * n1arrayneg[ii];
                        if ( dx0 < 0.0 )
                        {
                            optvalDneg0 += n0arrayneg[ii] * n0arrayneg[ii];
                        }
                        if ( dx1 < 0.0 )
                        {
                            optvalDneg1 += n1arrayneg[ii] * n1arrayneg[ii];
                        }
                    }
                    double epsN = 1e-8;
                    double epsD = 1e-16;// avoid division by zero

                    optval0 = (optvalN0 + epsN) / (optvalD0 + epsD);
                    optvalpos0 = (optvalNpos0 + epsN) / (optvalDpos0 + epsD);
                    optvalneg0 = (optvalNneg0 + epsN) / (optvalDneg0 + epsD);

                    optval1 = (optvalN1 + epsN) / (optvalD1 + epsD);
                    optvalpos1 = (optvalNpos1 + epsN) / (optvalDpos1 + epsD);
                    optvalneg1 = (optvalNneg1 + epsN) / (optvalDneg1 + epsD);

                    // quantity to be minimized
                    double n0coeff = 1.0; ///pow(1.0+n0, 2.0);
                    double n1coeff = 1.0; ///pow(1.0+n1, 2.0);

                    double optval = optval0*n0coeff;// + optval1*n1coeff;
                    double optvalneg = optvalneg0*n0coeff;// + optvalneg1*n1coeff;
                    double optvalpos = optvalpos0*n0coeff;// + optvalpos1*n1coeff;


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
                        // rotate neg
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
            if( cntmid > 30*(cntpos+cntneg) )
            {
                dangle *= 0.99;
            }


            loopiter ++;
        }

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

