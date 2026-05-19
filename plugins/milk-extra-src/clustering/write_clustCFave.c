/**
 * @file write_clustCFave.c
 * @brief Write clustcfave module
 */

#include <math.h>

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
#include "COREMOD_iofits/COREMOD_iofits.h"

#include "clustering_defs.h"


errno_t write_clustCFave(
    CLUSTERTREE *ctree,
    const char *__restrict outdname
)
{
    DEBUG_TRACE_FSTART();

    for(long CFindex = 0; CFindex < ctree->NBCF; CFindex++)
    {
        if(ctree->CFarray[CFindex].type != CLUSTER_CF_TYPE_UNUSED)
        {


            // WRITE CF ave file to disk

            IMGID imgCFave = imgid_make_from_name_2D("CFave", ctree->xsize, ctree->ysize);
            createimagefromIMGID(&imgCFave);

            uint64_t xysize = ctree->xsize;
            xysize *= ctree->ysize;

            for(uint64_t ii = 0; ii < xysize; ii++)
            {
                imgCFave.im->array.F[ii] = ctree->CFarray[CFindex].datasumvec[ii] / ctree->CFarray[CFindex].N;
            }

            char name[STRINGMAXLEN_STREAMNAME];
            WRITE_IMAGENAME(name, "%s/CF_%03ld.fits", outdname, CFindex);
            save_fits(imgCFave.name, name);

        }

    }

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}