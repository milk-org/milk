/**
 * @file leaf_addentry.c
 * @brief log all debug trace points to file
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#include "milkDebugTools.h"
#include "fps.h"
#include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "COREMOD_memory/COREMOD_memory.h"
#include "clustering_defs.h"

#include "addCF_to_CF.h"

// log all debug trace points to file
//#define DEBUGLOG

/*
 * Add entry to leaf
 */
errno_t leaf_addentry(
    CLUSTERTREE *ctree,
    double      *datavec,
    long double  ssqr,
    long         lCFindex,
    int         *addOK,
    double       distance
)
{
    DEBUG_TRACE_FSTART();

    // index of leaf to which point should be added
    long cfi = lCFindex;

#ifdef DEBUGPRINT
    printf("[%5d %s] trying to add vector to cfi %ld\n", __LINE__, __func__, cfi);
#endif


    // scan back to root, add vector to CF along the path
    int isleaf = 1; // will toggle to 0 when moving upstream of leaf
    while(cfi != -1)
    {
        CLUSTERING_CF CF;
        CF.datasumvec = datavec;
        CF.dataposvec = datavec;
        CF.datassq = ssqr;
        CF.N = 1;
        CF.pathcnt = 0.0;
        CF.posvecsourceID = cfi;
        addCF_to_CF(ctree, CF, cfi, addOK);

#ifdef DEBUGPRINT
        printf("[%5d %s] addOK = %d\n", __LINE__, __func__, *addOK);
#endif
        if(*addOK == 1)
        {
            ctree->CFarray[cfi].status |= CLUSTER_CF_STATUS_UPDATE;

            if(isleaf == 1)
            {
                // use distance to update leaf cluster radius
                if(distance > ctree->CFarray[cfi].radius)
                {
                    ctree->CFarray[cfi].radius = distance;
                }
            }

            // move upstream to propagate change
            cfi = ctree->CFarray[cfi].parentindex;

        }
        else
        {
            cfi = -1;
        }

        // indicate we are upstream of leaf cluster
        isleaf = 0;
    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
