/**
 * @file    PIAACMCsimul_achromFPMsol_eval_zonezderivative.c
 * @brief   PIAA-type coronagraph design, run
 *
 * Can design both APLCMC and PIAACMC coronagraphs
 *
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "CLIcore.h"
#include "coffee_compat.h"

errno_t PIAACMCsimul_achromFPMsol_eval_zonezderivative(long    zone,
        double *fpmresp_array,
        double *zonez_array,
        double *dphadz_array,
        double *outtmp_array,
        long    vsize,
        long    nbz,
        long    nbl)
{
    DEBUG_TRACE_FSTART();

    // axis 0: eval pts (ii)   size = data.core.image[IDfpmresp].md[0].size[0] -> vsize
    // axis 1: zones (mz)      size = data.core.image[piaacmcopticaldesign.zonezID].md[0].size[0]+1 = nbz+1
    // axis 3: lambda (k)      size = piaacmcopticaldesign.nblambda -> nbl
    //
    // indexing :  k*(data.core.image[piaacmcopticaldesign.zonezID].md[0].size[0]+1)*vsize + mz*vsize + ii

#ifdef PIAASIMUL_LOGFUNC1
    PIAACMCsimul_logFunctionCall("PIAACMCsimul.fcall.log",
                                 __FUNCTION__,
                                 __LINE__,
                                 "");
#endif

    for(long evalk = 0; evalk < nbl; evalk++)  // lambda loop
    {
        long evalki = evalk * (nbz + 1) * vsize;

        // outer zone
        //  for(evalii=0; evalii<vsize; evalii++)
        //    outtmp_array[evalk*vsize+evalii] = 0.0; //fpmresp_array[evalk*(nbz+1)*vsize+evalii];

        long evalmz = zone;

        // compute derivative as
        // dphadz is a function of wavelength
        // zonez is the current zone thickness
        // zonez*dphadz sets the phase gives the resulting phase

        // old sign convention (before 2017-12-23) :
        // -zonez*dphadz sets the phase gives the resulting phase
        // Re1 = Re * -dphadz*sin(-zonez*dphadz) - Im * dphadz*cos(-zonez*dphadz)
        // Im1 = Re * dphadz*cos(-zonez*dphadz) + Im * -dphadz*sin(-zonez*dphadz)

        // new sign convention (after 2017-12-23) :
        // zonez*dphadz sets the phase gives the resulting phase
        // Re1 = Re * dphadz * sin(zonez*dphadz) - Im * dphadz * cos(zonez*dphadz)
        // Im1 = Re * dphadz * cos(zonez*dphadz) + Im * dphadz * sin(zonez*dphadz)

        double evalpha =
            zonez_array[evalmz] *
            dphadz_array
            [evalk]; // CHANGED sign to + on 2017-12-23 to adopt new sign convention
        // !!! note that cos is sin and sin is cos !!!
        // this implements a 90 degree pre-rotation so that this is a
        // derivative
        double evalcosp =
            -sin(evalpha) *
            dphadz_array
            [evalk]; // z-derivative of cos(evalpha);   // CHANGED sign to - on 2017-12-23 to adopt new sign convention
        double evalsinp =
            cos(evalpha) *
            dphadz_array
            [evalk]; // z-derivative of sin(evalpha);  // CHANGED sign to + on 2017-12-23 to adopt new sign convention
        long evalki1 = evalki + (evalmz + 1) * vsize;
        long evalkv  = evalk * vsize;

        for(long evalii = 0; evalii < vsize / 2; evalii++)
        {
            long   evalii1 = 2 * evalii;
            long   evalii2 = 2 * evalii + 1;
            double evalre  = fpmresp_array[evalki1 + evalii1];
            double evalim  = fpmresp_array[evalki1 + evalii2];
            double evalre1 = evalre * evalcosp - evalim * evalsinp;
            double evalim1 = evalre * evalsinp + evalim * evalcosp;
            outtmp_array[evalkv + evalii1] = evalre1;
            outtmp_array[evalkv + evalii2] = evalim1;
        }
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
