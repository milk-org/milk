// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file    cubecluster.c
 * @brief   build cluster from image cube
 *
 * Use 3rd dimension as index
 */

#include <math.h>
#include <sys/stat.h>
#include <errno.h>


#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "COREMOD_iofits/COREMOD_iofits.h"

#include "CFmeminit.h"
#include "CFtree_rebuild.h"
#include "compute_imdistance_double.h"
#include "condense.h"
#include "create_new_leaf.h"
#include "ctree_init.h"
#include "ctree_memallocate.h"
#include "ctree_memfree.h"
#include "droptree.h"
#include "get_availableCFindex.h"
#include "leaf_addentry.h"
#include "leafnode_attachleaf.h"
#include "node_attachnode.h"
#include "printCFtree.h"
#include "split_CF_node.h"

#include "write_clustCFdat.h"
#include "write_clustCFave.h"
#include "write_clustleafsummary.h"

// #define DEBUGPRINT


#define pathprobdecay 0.95



static char *farg_inimname;
static char *farg_outdname;

static float *threshold = NULL; // stroke [um] for 100V
static long   fpi_threshold;

static uint32_t *branchB = NULL;
long             fpi_branchB;

static uint32_t *leafposmode = NULL;
long             fpi_leafposmode;

static uint32_t *NBCFmax = NULL;
long             fpi_NBCFmax;

static int64_t *optrebuild;
static long     fpi_optrebuild = -1;

static int64_t *optcondense;
static long     fpi_optcondense = -1;


// List of arguments to function
//
static CLICMDARGDEF farg[] = {
    {
        CLIARG_IMG,
        ".in_name",
        "input image cube",
        "imc1",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &farg_inimname,
        NULL
    },
    {
        CLIARG_STR,
        ".outdname",
        "output directory name",
        "outd",
        CLIARG_VISIBLE_DEFAULT,
        (void **) &farg_outdname,
        NULL
    },
    {
        CLIARG_FLOAT32,
        ".T",
        "threshold",
        "1.0",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &threshold,
        &fpi_threshold
    },
    {
        CLIARG_UINT32,
        ".B",
        "branch number",
        "10",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &branchB,
        &fpi_branchB
    },
    {
        CLIARG_UINT32,
        ".leafposmode",
        "leaf position mode (0:fixed, 1:dyn)",
        "1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &leafposmode,
        &fpi_leafposmode
    },
    {
        CLIARG_UINT32,
        ".NBCFmax",
        "max number of CFs",
        "2048",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &NBCFmax,
        &fpi_NBCFmax
    },
    {
        CLIARG_ONOFF,
        ".opt.rebuild",
        "rebuild tree after scan",
        "1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &optrebuild,
        &fpi_optrebuild
    },
    {
        CLIARG_ONOFF,
        ".opt.condense",
        "condense tree after scan",
        "1",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &optcondense,
        &fpi_optcondense
    }
};

// CLI function initialization data
static CLICMDDATA CLIcmddata =
{
    "cubeclust",            // keyword to call function in CLI
    "compute cube cluster", // description of what the function does
    CLICMD_FIELDS_DEFAULTS
};

// detailed help
static errno_t help_function()
{
    printf("Cluster image cube\n");

    return RETURN_SUCCESS;
}




static errno_t ctree_check(CLUSTERTREE *ctree)
{
    DEBUG_TRACE_FSTART();

#ifdef DEBUGPRINT
    printf("CHECK THREE <<<<<<<<<<<<<<<<<<<<<<<<\n");
#endif

    for(long cfi = 0; cfi < ctree->NBCF; cfi++)
    {
        if(ctree->CFarray[cfi].type != CLUSTER_CF_TYPE_UNUSED)
        {

            if(ctree->CFarray[cfi].N < 1)
            {
                FUNC_RETURN_FAILURE(
                    "node %ld type %d at level %d has N = %ld\n",
                    cfi,
                    ctree->CFarray[cfi].type,
                    ctree->CFarray[cfi].level,
                    ctree->CFarray[cfi].N);
            }

            if(ctree->CFarray[cfi].type == CLUSTER_CF_TYPE_NODE)
            {
                if(ctree->CFarray[cfi].NBchild > ctree->B)
                {
                    FUNC_RETURN_FAILURE(
                        "node %ld at level %d number of childred %d exceeds "
                        "limit %d",
                        cfi,
                        ctree->CFarray[cfi].level,
                        ctree->CFarray[cfi].NBchild,
                        ctree->B);
                }
            }
        }
    }
#ifdef DEBUGPRINT
    printf(">>>>>>>>>>>>>>>>>>>>>>>>>\n");
#endif

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




static errno_t findleafnode(
    CLUSTERTREE *ctree,
    double *datavec,
    long *nodeindex,
    double *distance
)
{
    DEBUG_TRACE_FSTART();

    // increments each time function is executed
    static long findleadnodecnt = 0;

    static long distcnt_eval = 0;
    static long distcnt_skip = 0;

    // find closest node descending the CFT from root
    // start at root
    static int  level   = 0;
    static long CFindex = 0;

    //DEBUG_TRACEPOINT
#ifdef DEBUGPRINT
    printf("[findleafnode]  root CF = #%ld, has %d child\n",
           CFindex,
           ctree->CFarray[CFindex].NBchild);
#endif

    // try last path first ... we may get lucky
    //
    int solution_reuse = 0;
    if(findleadnodecnt != 0)
    {
        if(ctree->leafposmode == CLUSTER_CFPOS_FIXED)
        {
            double distval = 0.0;
            FUNC_CHECK_RETURN(
                compute_imdistance_double(ctree,
                                          ctree->CFarray[CFindex].dataposvec,
                                          1,
                                          datavec,
                                          1,
                                          &distval));

            if(distval < ctree->T)
            {
                // last path leads to solution
                solution_reuse = 1;
                *distance = distval;
            }
        }
    }

    if(solution_reuse == 0)
    {
        level = 0;
        CFindex = ctree->rootindex;
        long distcnt0 = ctree->stat_compdistcnt;
        while(ctree->CFarray[CFindex].NBchild > 0)
        {
            int    scaninit    = 0;
            double distvalmin  = 0;
            long   CFindexbest = 0;

            ctree->path_node[level] = CFindex;

            // minumum and maximum possible distance from point to nodes
            double *mindistarray = (double*) malloc(sizeof(double)*ctree->CFarray[CFindex].NBchild);
            double *maxdistarray = (double*) malloc(sizeof(double)*ctree->CFarray[CFindex].NBchild);

            distvalmin = 2.0 * ctree->T;
            for(long childi = 0; childi < ctree->CFarray[CFindex].NBchild;
                    childi++)
            {
                double distval = 0.0;

                long CFindex1 = ctree->CFarray[CFindex].childindex[childi];

                // Should this point be skipped? (1:skip)
                int skipflag = 0;
                if(scaninit == 0)
                {
                    // don't skip first one
                    skipflag = 0;
                }
                else
                {
                    // Can't skip if dynamic position
                    if(ctree->leafposmode == CLUSTER_CFPOS_FIXED)
                    {
                        // The to-be-computed distance value will be larger than mindistarray[childi]
                        // if mindistarray[childi] is larger than distvalmin, we can skip
                        if ( distvalmin < mindistarray[childi] ) {
                            skipflag = 1;
                        }
                    }
                }

                // already found good-enough solution
                // TBD: less aggressive distance limit can be used here for nodes closer to root
                if(distvalmin < ctree->T) {
                    skipflag = 1;
                }



                if(skipflag == 0)
                {
                    distcnt_eval++;
                    if(ctree->leafposmode == CLUSTER_CFPOS_DYNAMIC)
                    {
                        FUNC_CHECK_RETURN(
                            compute_imdistance_double(ctree,
                                                      ctree->CFarray[CFindex1].datasumvec,
                                                      ctree->CFarray[CFindex1].N,
                                                      datavec,
                                                      1,
                                                      &distval));
                    }
                    else
                    {
                        FUNC_CHECK_RETURN(
                            compute_imdistance_double(ctree,
                                                      ctree->CFarray[CFindex1].dataposvec,
                                                      1,
                                                      datavec,
                                                      1,
                                                      &distval));
                    }

                    if(scaninit == 0)
                    {
                        distvalmin  = distval;
                        CFindexbest = CFindex1;

                        mindistarray[childi] = distval;
                        maxdistarray[childi] = distval;

                        for(long ci=0; ci<ctree->CFarray[CFindex].NBchild; ci++)
                        {
                            if(ci != childi)
                            {
                                long CFindex2 = ctree->CFarray[CFindex].childindex[ci];
                                double dval;
                                compute_CF2CF_posdistance_double(ctree, CFindex1, CFindex2, &dval);
                                double minval = dval - distval;
                                if(minval<0.0)
                                {
                                    minval = 0.0;
                                }
                                double maxval = dval + distval;
                                mindistarray[ci] = minval;
                                maxdistarray[ci] = maxval;
                            }
                        }

                        scaninit    = 1;
                    }
                    else
                    {

                        mindistarray[childi] = distval;
                        maxdistarray[childi] = distval;

                        for(long ci=0; ci<ctree->CFarray[CFindex].NBchild; ci++)
                        {
                            if(ci != childi)
                            {
                                long CFindex2 = ctree->CFarray[CFindex].childindex[ci];
                                double dval;
                                compute_CF2CF_posdistance_double(ctree, CFindex1, CFindex2, &dval);
                                double minval = dval - distval;
                                if(minval<0.0)
                                {
                                    minval = 0.0;
                                }
                                double maxval = dval + distval;
                                if (minval > mindistarray[ci]) {
                                    mindistarray[ci] = minval;
                                }
                                if (maxval < maxdistarray[ci]) {
                                    maxdistarray[ci] = maxval;
                                }
                            }
                        }


                        if(distval < distvalmin)
                        {
                            distvalmin  = distval;
                            CFindexbest = CFindex1;
                        }
                    }
                }
                else
                {
                    distcnt_skip++;
                }
            }
            free(mindistarray);
            free(maxdistarray);

            //DEBUG_TRACEPOINT
#ifdef DEBUGPRINT
            printf("[findleafnode]  level %3d #%4ld  %g\n", level, CFindexbest, distvalmin);
#endif
            CFindex = CFindexbest;
            *distance = distvalmin;
            ctree->path_distcompcnt[level] = ctree->stat_compdistcnt;

            level++;
        }
    }

    //DEBUG_TRACEPOINT
#ifdef DEBUGPRINT
    printf("[findleafnode]  NEAREST NODE : %ld  ( nbchild=%3d)\n",
           CFindex,
           ctree->CFarray[CFindex].NBchild);
#endif
    *nodeindex = CFindex;

    /*long distcnt1 = ctree->stat_compdistcnt;

    for(int l=0; l<level; l++)
    {
        printf("    LEVEL %5d/%5d   NODE = %5ld [NBCH %5d]  CNT = %12ld\n",
               l, level, ctree->path_node[l], ctree->CFarray[ctree->path_node[l]].NBchild,
               ctree->path_distcompcnt[l]);
    }

    printf("   L %05d    %12ld\n", 0, ctree->path_distcompcnt[0]-distcnt0);
    for(int l=1; l<level; l++) {
        printf("   L %05d    %12ld\n", l, ctree->path_distcompcnt[l]-ctree->path_distcompcnt[l-1]);
    }
    printf("  NODE %ld\n\n", CFindex);*/

    //printf("EVAL : %8ld   SKIP : %8ld    frac = %8.6f\n", distcnt_eval, distcnt_skip, 1.0*distcnt_eval/(distcnt_eval+distcnt_skip));

    findleadnodecnt++;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




static errno_t indexpermut(
    long *indexarray,
    long size
)
{
    for(long k0=1; k0<size; k0++)
    {
        long k1 = rand()%size;
        long lval = indexarray[k0];
        indexarray[k0] = indexarray[k1];
        indexarray[k1] = lval;
    }
    return RETURN_SUCCESS;
}




static errno_t imcube_makecluster(
    IMGID img,
    const char *__restrict outdname
)
{
    // entering function, updating trace accordingly
    DEBUG_TRACE_FSTART();
    DEBUG_TRACEPOINT("FARG %s", outdname);

    resolveIMGID(&img, ERRMODE_ABORT);

    uint32_t xsize = img.md->size[0];
    uint32_t ysize = img.md->size[1];
    uint32_t zsize = img.md->size[2];

    uint64_t xysize = xsize;
    xysize *= ysize;

    if(zsize == 0)
    {
        // if 2D image, assume ysize is number of samples
        xysize = xsize;
        zsize  = ysize;
    }

#ifdef DEBUGPRINT
    printf("image size %u %u %u\n", xsize, ysize, zsize);
#endif

    // looking for mask image
    imageID IDmask = image_ID("maskim");
    if(IDmask == -1)
    {
        printf("Creating default mask image %ld pixel\n", xysize);
        create_2Dimage_ID("maskim", xsize, ysize, &IDmask);

        for(uint64_t ii = 0; ii < xysize; ii++)
        {
            data.image[IDmask].array.F[ii] = 1.0;
        }
    }
    else
    {
        printf("Mask image loaded\n");
    }



    // build pixmap to load input images in vectors
    //
    float maskeps = 1.0e-5; // threshold below which pixels are ignored
    long  pixcnt  = 0;
    for(uint64_t ii = 0; ii < xysize; ii++)
    {
        if(data.image[IDmask].array.F[ii] > maskeps)
        {
            pixcnt++;
        }
    }
    long CF_npix = pixcnt;
    DEBUG_TRACEPOINT("CF_npix = %ld", CF_npix);

    long *pixmap = (long *) malloc(sizeof(long) * CF_npix);
    if(pixmap == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }
    double *pixgain = (double *) malloc(sizeof(double) * CF_npix);
    if(pixgain == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }

    long inpixindex = 0;
    for(uint64_t ii = 0; ii < xysize; ii++)
    {
        if(data.image[IDmask].array.F[ii] > maskeps)
        {
            pixmap[inpixindex]  = ii;
            pixgain[inpixindex] = data.image[IDmask].array.F[ii];
            inpixindex++;
        }
    }


    CLUSTERTREE ctree; // cluster tree

    ctree.xsize = xsize;
    ctree.ysize = ysize;

    ctree.NBCF         = *NBCFmax;          // max number of cluster features
    ctree.B            = *branchB;       // max number of branches out of node
    ctree.noise2offset = 0.0;            // if very noisy image, subtract known noise
    ctree.T            = *threshold;     // threshold satisfied by each CF entry of leaf node
    ctree.leafposmode  = *leafposmode;   // leaf position mode, 0:static, 1:dynamic

    ctree.npix = CF_npix;


    ctree.stat_compdistcnt = 0; // counter for distance computations

    // Allocate memory for CFs
    FUNC_CHECK_RETURN(ctree_memallocate(&ctree));


    // pointer to current array
    double *datarray = (double*) malloc(sizeof(double)*CF_npix);


    printf("\n");
    long NBframe = zsize;

    // keeping track of leaf CF index for each frame
    // each frame belongs to a CF
    long *frameleafCFindex = (long *) malloc(sizeof(long) * NBframe);
    if(frameleafCFindex == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }


    // permut ordering
    // index (frameC) in input cube, may be different from loop index (frame)
    // if reading the input cube out of order
    long * frameC = (long*) malloc(sizeof(long)*NBframe);
    for(long frame = 0; frame < NBframe; frame++)
    {
        frameC[frame] = frame;
    }
    //indexpermut(frameC, NBframe);


    // MAIN SCAN THROUGH DATASET
    //

    long framecnt = 0;
    for(long frame = 0; frame < NBframe; frame++)
    {
        DEBUG_TRACEPOINT("PROCESSING FRAME %5ld", frame);
        frameleafCFindex[frame] = -1;



        FUNC_CHECK_RETURN(ctree_check(&ctree));


        // Load image data into vector datarray
        //
        DEBUG_TRACEPOINT("Load image data into datarray xysize=%ld CF_npix=%ld", xysize, CF_npix);
        long double ssqr     = 0.0;

        if ( img.im->md->datatype == _DATATYPE_FLOAT )
        {
            for(long ii = 0; ii < CF_npix; ii++)
            {
                datarray[ii] =
                    pixgain[ii] * img.im->array.F[frameC[frame] * xysize + pixmap[ii]];
                ssqr += datarray[ii]*datarray[ii];
            }
        }
        else if ( img.im->md->datatype == _DATATYPE_DOUBLE )
        {
            for(long ii = 0; ii < CF_npix; ii++)
            {
                datarray[ii] =
                    pixgain[ii] * img.im->array.D[frameC[frame] * xysize + pixmap[ii]];
                ssqr += datarray[ii]*datarray[ii];
            }
        }
        else if ( img.im->md->datatype == _DATATYPE_UINT16 )
        {
            for(long ii = 0; ii < CF_npix; ii++)
            {
                datarray[ii] =
                    pixgain[ii] * img.im->array.UI16[frameC[frame] * xysize + pixmap[ii]];
                ssqr += datarray[ii]*datarray[ii];
            }
        }
        else if ( img.im->md->datatype == _DATATYPE_UINT32 )
        {
            for(long ii = 0; ii < CF_npix; ii++)
            {
                datarray[ii] =
                    pixgain[ii] * img.im->array.UI32[frameC[frame] * xysize + pixmap[ii]];
                ssqr += datarray[ii]*datarray[ii];
            }
        }
        else if ( img.im->md->datatype == _DATATYPE_INT16 )
        {
            for(long ii = 0; ii < CF_npix; ii++)
            {
                datarray[ii] =
                    pixgain[ii] * img.im->array.SI16[frameC[frame] * xysize + pixmap[ii]];
                ssqr += datarray[ii]*datarray[ii];
            }
        }
        else if ( img.im->md->datatype == _DATATYPE_INT32 )
        {
            for(long ii = 0; ii < CF_npix; ii++)
            {
                datarray[ii] =
                    pixgain[ii] * img.im->array.SI32[frameC[frame] * xysize + pixmap[ii]];
                ssqr += datarray[ii]*datarray[ii];
            }
        }
        else
        {
            for(long ii = 0; ii < CF_npix; ii++)
            {
                datarray[ii] = 0.0;
                ssqr += datarray[ii]*datarray[ii];
            }
        }




        DEBUG_TRACEPOINT("Processing ID %ld frame %ld, %ld pix",
                         img.ID,
                         frame,
                         CF_npix);

#ifdef DEBUGPRINT
        FUNC_CHECK_RETURN(printCFtree(&ctree));
#endif

        if(frame == 0)
        {
            // INITIALIZATION
#ifdef DEBUGPRINT
            printf("FIRST FRAME --> INITIALIZE ctree\n");
#endif
            ctree_init(&ctree, datarray, ssqr);
            frameleafCFindex[0] = 1;
        }
        else
        {
            long CFindex;
            double distance = 0.0;
            FUNC_CHECK_RETURN(findleafnode(&ctree, datarray, &CFindex, &distance));

            // we have descended the tree and are now at a leaf node
            long lCFindex = CFindex;

            // only add if radius condition is met
            int addOK = 0;
#ifdef DEBUGPRINT
            printf("[%5d %s] leaf_addentry %ld\n", __LINE__, __func__, lCFindex);
#endif

            FUNC_CHECK_RETURN(leaf_addentry(&ctree,
                                            datarray,
                                            ssqr,
                                            lCFindex,
                                            &addOK,
                                            distance));



            if(addOK == 1)
            {
                // leaf has been added
                frameleafCFindex[frame] = lCFindex;
                DEBUG_TRACEPOINT("Added entry %ld to leaf index %ld",
                                 frame, lCFindex);
            }
            else
            {
                DEBUG_TRACEPOINT(
                    "Radius condition not met for leaf index %ld",
                    lCFindex);
                // indicate that leaf has not been added
                lCFindex = -1;
            }



            if(lCFindex == -1)
            {
                // If leaf has not been added, create new leaf
                //
                long nCFindex;
                FUNC_CHECK_RETURN(
                    create_new_leaf(&ctree, datarray, ssqr, &nCFindex));

                frameleafCFindex[frame] = nCFindex;

                DEBUG_TRACEPOINT("CREATED LEAF at index %ld", nCFindex);

                // attach new leaf to parent
                FUNC_CHECK_RETURN(
                    node_attachleaf(&ctree, nCFindex, ctree.CFarray[CFindex].parentindex));

                DEBUG_TRACEPOINT("ATTACHED LEAF %ld to %ld",
                                 nCFindex,
                                 ctree.CFarray[CFindex].parentindex);

                CFindex = ctree.CFarray[CFindex].parentindex;

                if(ctree.CFarray[CFindex].NBchild == ctree.B + 1)
                {
                    DEBUG_TRACEPOINT(
                        "MAX BRANCH NUMBER REACHED -> SPLIT NODE");

                    long CFi0;
                    long CFi1;
                    FUNC_CHECK_RETURN(
                        split_CF_node(&ctree, CFindex, &CFi0, &CFi1));

                    DEBUG_TRACEPOINT("NODE %ld(%d) -> %ld(%d) %ld(%d)",
                                     CFindex,
                                     ctree.CFarray[CFindex].NBchild,
                                     CFi0,
                                     ctree.CFarray[CFi0].NBchild,
                                     CFi1,
                                     ctree.CFarray[CFi1].NBchild);

                    // check if upstrem # children OK
                    long upCF = ctree.CFarray[CFi0].parentindex;

                    // flag equal to 1 while upstream nodes need to be split
                    int splitupstream = 0;

                    if(ctree.CFarray[upCF].NBchild == ctree.B + 1)
                    {
                        // if more children thn branching parameter, we nned to split
                        splitupstream = 1;
                    }
                    while(splitupstream == 1)
                    {
                        DEBUG_TRACEPOINT("SPLITTING NODE %ld", upCF);

                        if(ctree.CFarray[upCF].level == 0)
                        {
                            FUNC_CHECK_RETURN(droptree(&ctree));
                            // if we're at the root, this is the last split we need to do
                            splitupstream = 0;
                        }

                        long CFi0;
                        long CFi1;

                        FUNC_CHECK_RETURN(
                            split_CF_node(&ctree, upCF, &CFi0, &CFi1));

                        DEBUG_TRACEPOINT("NODE %ld(%d) -> %ld(%d) %ld(%d)",
                                         CFindex,
                                         ctree.CFarray[CFindex].NBchild,
                                         CFi0,
                                         ctree.CFarray[CFi0].NBchild,
                                         CFi1,
                                         ctree.CFarray[CFi1].NBchild);

                        upCF = ctree.CFarray[CFi0].parentindex;
                        if(upCF != -1)
                        {
                            if(ctree.CFarray[upCF].NBchild == ctree.B + 1)
                            {
                                splitupstream = 1;
                            }
                        }
                    }
                }
            }




            // housekeeping, update tracers
            // The point has been added to leaf index frameleafCFindex[frame]
            //
            {
                // scan back to root, add vector to CF along the path
                long cfi = frameleafCFindex[frame];
                //printf(">>>>>>>>>> frame %ld, cfi %ld\n", frame, cfi);
                while(cfi != -1)
                {
                    ctree.CFarray[cfi].pathcnt += 1.0;
                    ctree.CFarray[cfi].pathdistcompcnt += 1.0;

                    // move upstream to propagate change
                    cfi = ctree.CFarray[cfi].parentindex;
                }
            }

            int condensenop = 1;
            while(condensenop > 0)
            {
                // condense = compress levels whenever possible
                //
                FUNC_CHECK_RETURN(ctree_condense(&ctree, &condensenop));

            }

        }

        //printCFtree(&ctree);

        for(long cfi = 0; cfi < ctree.NBCF; cfi++)
        {
            ctree.CFarray[cfi].status = 0;
            ctree.CFarray[cfi].pathcnt *= pathprobdecay;
        }


        {
            char fname[STRINGMAXLEN_FILENAME];
            WRITE_FILENAME(fname, "%s/clust.CF.%05ld.dat", outdname, framecnt);
            DEBUG_TRACEPOINT("writing %s", fname);
            write_clustCFdat(&ctree, fname);
        }


        DEBUG_TRACEPOINT(
            "Frame %ld processed",
            framecnt);
        framecnt++;


        /*printf("[%5ld] ROOT N=%5ld  ", frame, ctree.CFarray[ctree.rootindex].N);
        for(long ii=0; ii<10; ii++)
        {
            printf("  %12f", ctree.CFarray[ctree.rootindex].dataposvec[ii]);
        }
        printf("\n");*/

    }


    printf("\n");
    printf("Processed %ld / %ld frames\n", framecnt, NBframe);
    printf("Distance comp counter:  %ld\n", ctree.stat_compdistcnt);


#ifdef DEBUGPRINT
    FUNC_CHECK_RETURN(printCFtree(&ctree));
#endif




    if(*optrebuild == 1)  // ON state
    {
        printf("Rebuilding CF tree from clusters\n");
        FUNC_CHECK_RETURN(CFtree_rebuild(&ctree, frameleafCFindex, NBframe));
    }


    if(*optcondense == 1)  // ON state
    {
        printf("Condensing CF tree\n");
        int condensenop = 1;
        while(condensenop > 0)
        {
            // condense = compress levels whenever possible
            //
#ifdef DEBUGPRINT
            printf("========================== CONDENSING ===========================\n");
            printCFtree(&ctree);
#endif
            FUNC_CHECK_RETURN(ctree_condense(&ctree, &condensenop));

        }
    }


    DEBUG_TRACEPOINT(" ");

    {
        ctree.nbnode       = 0;
        ctree.nbleaf       = 0;
        ctree.nbleafsingle = 0;
        int maxlevel = 0;
        for(long cfi = 0; cfi < ctree.NBCF; cfi++)
        {
            if(ctree.CFarray[cfi].type != CLUSTER_CF_TYPE_UNUSED)
            {
                if(ctree.CFarray[cfi].level > maxlevel)
                {
                    maxlevel = ctree.CFarray[cfi].level;
                }

                switch(ctree.CFarray[cfi].type)
                {
                case CLUSTER_CF_TYPE_NODE:
                    ctree.nbnode++;
                    if(ctree.CFarray[cfi].level > maxlevel)
                    {
                        maxlevel = ctree.CFarray[cfi].level;
                    }
                    break;

                case CLUSTER_CF_TYPE_LEAF:
                    if(ctree.CFarray[cfi].N == 1)
                    {
                        ctree.nbleafsingle++;
                    }
                    if(ctree.CFarray[cfi].level > maxlevel)
                    {
                        maxlevel = ctree.CFarray[cfi].level;
                    }
                    ctree.nbleaf++;
                    break;
                }
            }
        }
        printf("\n");
        printf("    max level  = %5d\n", maxlevel);
        printf("    nbnode     = %5ld\n", ctree.nbnode);
        printf("    nbleaf     = %5ld (incl %ld singles)\n",
               ctree.nbleaf,
               ctree.nbleafsingle);
        printf("\n");
    }



#ifdef DEBUGPRINT
    // TEST print
    printCFtree(&ctree);
#endif


    DEBUG_TRACEPOINT("Writing output to filesystem");
    {
        errno = 0;
        if(mkdir(outdname, 0777) != 0 && errno != EEXIST )
        {
            FUNC_RETURN_FAILURE("mkdir failure");
        }
    }

    write_clustleafsummary(&ctree, img, pixmap, pixgain, frameleafCFindex, NBframe,
                           outdname);

    {
        char fname[STRINGMAXLEN_FILENAME];
        WRITE_FILENAME(fname, "%s/clust.CF.dat", outdname);
        DEBUG_TRACEPOINT("writing %s", fname);
        write_clustCFdat(&ctree, fname);
    }

    write_clustCFave(&ctree, outdname);



    /*
        {
            // measure distance betweeen nodes and leaves

            char fname[STRINGMAXLEN_FILENAME];
            WRITE_FILENAME(fname, "%s/clust.LFdist.dat", outdname);

            FILE *fp = fopen(fname, "w");

            fprintf(fp,"# Distance between leaf CFs\n");
            fprintf(fp,"#\n");
            fprintf(fp,"# col1   CF index 0\n");
            fprintf(fp,"# col2   CF index 1\n");
            fprintf(fp,"# col3   CF0-CF1 distance\n");
            fprintf(fp,"# col4   CF0-CF1 distance/threshold\n");
            fprintf(fp,"# col5   (N0*N1) / (N0+N1)\n");
            fprintf(fp,"# col6   N0\n");
            fprintf(fp,"# col7   N1\n");
            fprintf(fp,"#\n");

            for(long CFindex0 = 0; CFindex0 < ctree.NBCF; CFindex0++)
            {
                if(ctree.CFarray[CFindex0].type == CLUSTER_CF_TYPE_LEAF)
                {
                    for(long CFindex1 = 0; CFindex1 < CFindex0; CFindex1++)
                    {
                        if(ctree.CFarray[CFindex1].type == CLUSTER_CF_TYPE_LEAF)
                        {
                            if(ctree.CFarray[CFindex0].level ==
                                    ctree.CFarray[CFindex1].level)
                            {
                                double distval;
                                compute_imdistance_double(
                                    &ctree,
                                    ctree.CFarray[CFindex0].datasumvec,
                                    ctree.CFarray[CFindex0].N,
                                    ctree.CFarray[CFindex1].datasumvec,
                                    ctree.CFarray[CFindex1].N,
                                    &distval);

                                fprintf(fp,
                                        "%5ld %5ld      %16g  %6.4f  %6.2f  %3ld %3ld\n",
                                        CFindex0,
                                        CFindex1,
                                        distval,
                                        distval / ctree.T,
                                        1.0 / (1.0 / ctree.CFarray[CFindex0].N +
                                               1.0 / ctree.CFarray[CFindex1].N),
                                        ctree.CFarray[CFindex0].N,
                                        ctree.CFarray[CFindex1].N);
                            }
                        }
                    }
                }
            }
            fclose(fp);
        }
    */

    free(frameleafCFindex);

    printf("Freeing CF memory\n");
    free(pixmap);
    free(pixgain);

    ctree_memfree(&ctree);

    // normal successful return from function :
    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}


// Wrapper function, used by all CLI calls
// Defines how local variables are fed to computation code
// Always local to this translation unit
static errno_t compute_function()
{
    DEBUG_TRACE_FSTART();

    imcube_makecluster(mkIMGID_from_name(farg_inimname), farg_outdname);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

INSERT_STD_FPSCLIfunctions

/** @brief Register CLI command
*
* Adds function to list of CLI commands.
* Called by main module initialization function init_module_CLI().
*/
errno_t
CLIADDCMD_clustering__imcube_mkcluster()
{
    INSERT_STD_CLIREGISTERFUNC

    return RETURN_SUCCESS;
}
