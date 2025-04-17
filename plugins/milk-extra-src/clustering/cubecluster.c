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
#include "addvector_to_CF.h"
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


//#define DEBUGPRINT


static char *farg_inimname;
static char *farg_outdname;

static float *threshold = NULL; // stroke [um] for 100V
static long   fpi_threshold;

static uint32_t *branchB = NULL;
long             fpi_branchB;

static uint32_t *branchL = NULL;
long             fpi_branchL;





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
        ".L",
        "branch number leaf node",
        "10",
        CLIARG_HIDDEN_DEFAULT,
        (void **) &branchL,
        &fpi_branchL
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

            /*if(ctree->CFarray[cfi].type == CLUSTER_CF_TYPE_LEAFNODE)
            {
                if(ctree->CFarray[cfi].NBleaf > ctree->L)
                {
                    FUNC_RETURN_FAILURE(
                        "node %ld at level %d number of leaves %d exceeds "
                        "limit %d",
                        cfi,
                        ctree->CFarray[cfi].level,
                        ctree->CFarray[cfi].NBleaf,
                        ctree->L);
                }
            }*/
        }
    }
#ifdef DEBUGPRINT
    printf(">>>>>>>>>>>>>>>>>>>>>>>>>\n");
#endif

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}




static errno_t
findleafnode(CLUSTERTREE *ctree, double *datavec, long *nodeindex)
{
    DEBUG_TRACE_FSTART();

    // find closest node descending the CFT from root
    // start at root
    int  level   = 0;
    long CFindex = ctree->rootindex;

    //DEBUG_TRACEPOINT
#ifdef DEBUGPRINT
    printf("[findleafnode]  root CF = #%ld, has %d child\n",
           CFindex,
           ctree->CFarray[CFindex].NBchild);
#endif

    while(ctree->CFarray[CFindex].NBchild > 0)
    {
        int    scaninit    = 0;
        double distvalmin  = 0;
        long   CFindexbest = 0;
        for(long childi = 0; childi < ctree->CFarray[CFindex].NBchild;
                childi++)
        {
            double distval = 0.0;

            long CFindex1 = ctree->CFarray[CFindex].childindex[childi];

            //DEBUG_TRACEPOINT(
#ifdef DEBUGPRINT
            printf("[findleafnode]  computing distance to CF node #%ld (N=%ld)\n",
                   CFindex1,
                   ctree->CFarray[CFindex1].N);
#endif

            FUNC_CHECK_RETURN(
                compute_imdistance_double(ctree,
                                          ctree->CFarray[CFindex1].datasumvec,
                                          ctree->CFarray[CFindex1].N,
                                          datavec,
                                          1,
                                          &distval));

#ifdef DEBUGPRINT
            printf("[findleafnode]   distance value = %f\n", distval);
#endif

            if(scaninit == 0)
            {
                distvalmin  = distval;
                CFindexbest = CFindex1;
                scaninit    = 1;
            }
            else
            {
                if(distval < distvalmin)
                {
                    distvalmin  = distval;
                    CFindexbest = CFindex1;
                }
            }
        }

        //DEBUG_TRACEPOINT
#ifdef DEBUGPRINT
        printf("[findleafnode]  level %3d #%4ld  %g\n", level, CFindexbest, distvalmin);
#endif
        CFindex = CFindexbest;
        level++;
    }

    //DEBUG_TRACEPOINT
#ifdef DEBUGPRINT
    printf("[findleafnode]  NEAREST NODE : %ld  ( nbchild=%3d)\n",
           CFindex,
           ctree->CFarray[CFindex].NBchild);
#endif
    *nodeindex = CFindex;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}





static errno_t
findleaf(
    CLUSTERTREE *ctree,
    double *datavec,
    long CFindex,
    long *leafindex
)
{
    DEBUG_TRACE_FSTART();

    int    leafimin     = -1; // leaf index into which entry will be added
    double distvalmin   = 0;
    int    leafloopinit = 0;
    for(long leafi = 0; leafi < ctree->CFarray[CFindex].NBchild; leafi++)
    {
        long CFindex1 = ctree->CFarray[CFindex].childindex[leafi];
        if(ctree->CFarray[CFindex1].type == CLUSTER_CF_TYPE_LEAF)
        {

            //if(sqrt(ctree->CFarray[CFindex1].radius2) < ctree->T)
            //{   // radius below threahold -> we can add
            double distval = 0.0;

            //DEBUG_TRACEPOINT
#ifdef DEBUGPRINT
            printf("[findleaf] computing distance to leaf %ld = node # %ld(%ld)\n",
                   leafi,
                   CFindex1,
                   ctree->CFarray[CFindex1].N);
#endif

            FUNC_CHECK_RETURN(
                compute_imdistance_double(ctree,
                                          ctree->CFarray[CFindex1].datasumvec,
                                          ctree->CFarray[CFindex1].N,
                                          datavec,
                                          1,
                                          &distval));

            //DEBUG_TRACEPOINT
#ifdef DEBUGPRINT
            printf("[findleaf]    dist %4ld(%3ld) - new sample : %g\n",
                   CFindex1,
                   ctree->CFarray[CFindex1].N,
                   (double) distval);
#endif

            if(leafloopinit == 0)
            {
                leafimin     = leafi;
                distvalmin   = distval;
                leafloopinit = 1;
            }
            else
            {
                if(distval < distvalmin)
                {
                    leafimin   = leafi;
                    distvalmin = distval;
                }
            }
            /*}
            else
            {
                printf("        cluster radius %g > %g -> skipping\n",
                       (double) sqrt(ctree->CFarray[CFindex1].radius2), (double) ctree->T);
            }*/
        }
    }


    if(distvalmin > ctree->T)
    {
#ifdef DEBUGPRINT
        printf("[findleaf]        distance too large -> can't add to closest leaf\n");
#endif
        leafimin = -1;
    }

    //DEBUG_TRACEPOINT(
#ifdef DEBUGPRINT
    printf("[findleaf]  leafimin = %d\n", leafimin);
#endif

    *leafindex = leafimin;

    DEBUG_TRACE_FEXIT();
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

    ctree.NBCF         = 20000;       // number of cluster features
    ctree.B            = *branchB;          // max number of branches out of node
    ctree.L            = *branchL;          // max numbers of CF entries in leaf node
    ctree.noise2offset = 0.0;         // if very noisy image, subtract known noise
    ctree.T            = *threshold;     // threshold satisfied by each CF entry of leaf node

    ctree.npix = CF_npix;

    // Allocate memory for CFs
    FUNC_CHECK_RETURN(ctree_memallocate(&ctree));

    // storage for current input vector
    double *datarray0 = (double *) malloc(sizeof(double) * CF_npix);
    if(datarray0 == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }

    // storage for previous input vector
    double *datarray1 = (double *) malloc(sizeof(double) * CF_npix);
    if(datarray1 == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }

    // pointer to current array
    double *datarray;

    // initially, point to datarray0
    datarray = datarray0;

    printf("\n");
    long NBframe = zsize;

    // keeping track of leaf CF index for each frame
    // each frame belongs to a CF
    long *frameleafCFindex = (long *) malloc(sizeof(long) * NBframe);
    if(frameleafCFindex == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }



    long framecnt = 0;
    for(long frame = 0; frame < NBframe; frame++)
    {
#ifdef DEBUGPRINT
        printf("PROCESSING FRAME %5ld\n", frame);
#endif
        frameleafCFindex[frame] = -1;


        FUNC_CHECK_RETURN(ctree_check(&ctree));

        // Load image data into vector datarray
        //
        long double ssqr     = 0.0;
        long double ssqrdiff = 0.0;
        for(long ii = 0; ii < CF_npix; ii++)
        {
            datarray[ii] =
                pixgain[ii] * img.im->array.F[frame * xysize + pixmap[ii]];
            ssqr += datarray[ii] * datarray[ii];

            double vdiff = datarray0[ii] - datarray1[ii];
            ssqrdiff += vdiff * vdiff;
        }

        // check that vector is different from previous one to avoid duplicates
        int frameskip = 0;
        if(ssqrdiff < 1.0e-6 * ssqr)
        {
            // duplicate, skip
#ifdef DEBUGPRINT
            printf("\n skipping ID %5ld frame %5ld  :  %16Lg  %16Lg -> %16Lg   \n",
                   img.ID, frame,
                   ssqrdiff,
                   ssqr,
                   ssqrdiff/ssqr);
#endif
            frameskip = 1;
        }



        if(frame == 0)
        {
            frameskip = 0;
        }

        if(frameskip == 0)
        {
            // NOT SKIPPING

#ifdef DEBUGPRINT
            printf("Processing ID %ld frame %ld, %ld pix    \r",
                   img.ID,
                   frame,
                   CF_npix);
            printf("---------------- %16Lg  %16Lg -> %16Lg\n",
                   ssqrdiff,
                   ssqr,
                   ssqrdiff/ssqr);

            printf("    SSQR = %g\n", (double) ssqr);

            FUNC_CHECK_RETURN(printCFtree(&ctree));
#endif

            if(frame == 0)
            {
                // INITIALIZATION
#ifdef DEBUGPRINT
                printf("FIRST FRAME --> INITIALIZE ctree\n");
#endif
                ctree_init(&ctree, datarray, ssqr);
                frameleafCFindex[0] = 2;
            }
            else
            {
                long CFindex;
                FUNC_CHECK_RETURN(findleafnode(&ctree, datarray, &CFindex));
                //DEBUG_TRACEPOINT(
#ifdef DEBUGPRINT
                printf("[%5d %s] -> CF #%ld type is %d  (2=node, 3=leaf)\n",
                       __LINE__, __func__, CFindex,
                       ctree.CFarray[CFindex].type);
#endif


                // we have descended the tree and are now at a leaf node

                //long leafi;
                //FUNC_CHECK_RETURN(findleaf(&ctree, datarray, CFindex, &leafi));

#ifdef DEBUGPRINT
                // printf("findleaf -> leafi %ld\n", leafi);
#endif
                // we have descended the tree and are now at a leaf node



                //if(leafi != -1)
                //{
                //    long lCFindex = ctree.CFarray[CFindex].childindex[leafi];

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
                                                &addOK));



#ifdef DEBUGPRINT
                printf("[%5d] %s\n", __LINE__, __FILE__);
#endif

                if(addOK == 1)
                {
                    // leaf has been added
                    frameleafCFindex[frame] = lCFindex;
                    DEBUG_TRACEPOINT("Added entry to leaf index %ld",
                                     lCFindex);
                }
                else
                {
                    DEBUG_TRACEPOINT(
                        "Radius condition not met for leaf index %ld",
                        lCFindex);
                    // indicate that leaf has not been added
                    lCFindex = -1;
                }
                // }


#ifdef DEBUGPRINT
                printf("[%5d] %s\n", __LINE__, __FILE__);
#endif

                if(lCFindex == -1)
                {
                    // If leaf has not been added
#ifdef DEBUGPRINT
                    printf("[%5d %s] leaf has not been added\n", __LINE__, __func__);
#endif

                    DEBUG_TRACEPOINT("Creating new leaf # %d",
                                     ctree.CFarray[CFindex].NBleaf);
                    long nCFindex;
#ifdef DEBUGPRINT
                    printf("[%5d %s] creating leaf\n", __LINE__, __func__);
#endif
                    FUNC_CHECK_RETURN(
                        create_new_leaf(&ctree, datarray, ssqr, &nCFindex));

                    frameleafCFindex[frame] = nCFindex;

#ifdef DEBUGPRINT
                    printf("[%5d %s] created leaf %ld\n", __LINE__, __func__, nCFindex);
#endif


                    DEBUG_TRACEPOINT("CREATED LEAF at index %ld", nCFindex);

                    // attach new leaf to parent

                    DEBUG_TRACEPOINT("ATTACHING LEAF %ld to %ld",
                                     nCFindex,
                                     ctree.CFarray[CFindex].parentindex);


#ifdef DEBUGPRINT
                    printf("[%5d] %s\n", __LINE__, __FILE__);
                    printf("[%5d %s] attaching leaf %ld to node %ld (parent of %ld)\n", __LINE__, __func__,
                           nCFindex, ctree.CFarray[CFindex].parentindex, CFindex);
#endif

                    FUNC_CHECK_RETURN(
                        node_attachleaf(&ctree, nCFindex, ctree.CFarray[CFindex].parentindex));

#ifdef DEBUGPRINT
                    printf("[%5d] %s\n", __LINE__, __FILE__);
#endif


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
            }

            //printCFtree(&ctree);

            for(long cfi = 0; cfi < ctree.NBCF; cfi++)
            {
                ctree.CFarray[cfi].status = 0;
            }

            if(datarray == datarray0)
            {
                //printf("0 -> 1\n");
                datarray = datarray1;
            }
            else
            {
                //printf("1 -> 0\n");
                datarray = datarray0;
            }

            int condensenop = 1;
            while(condensenop > 0)
            {
                // condense = compress levels whenever possible
                //
                FUNC_CHECK_RETURN(ctree_condense(&ctree, &condensenop));
            }



#ifdef DEBUGPRINT
            FUNC_CHECK_RETURN(printCFtree(&ctree));
            printf("=================================================\n");
            printf("=================================================\n");
            printf("=================================================\n");
            printf("\n\n\n");
#endif


            framecnt++;

            if(framecnt % 100000 == 0)
            {
                FUNC_CHECK_RETURN(
                    CFtree_rebuild(&ctree, frameleafCFindex, NBframe));
            }
        }
    }


    printf("\n");
    printf("Processed %ld / %ld frames\n", framecnt, NBframe);


#ifdef DEBUGPRINT
    FUNC_CHECK_RETURN(printCFtree(&ctree));
#endif



    printf("Rebuilding CF tree from clusters\n");
    FUNC_CHECK_RETURN(CFtree_rebuild(&ctree, frameleafCFindex, NBframe));




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
        printf("    max level  = %d\n", maxlevel);
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


    printf("Writing output to filesystem\n");
    {
        errno = 0;
        if(mkdir(outdname, 0777) != 0 && errno != EEXIST )
        {
            FUNC_RETURN_FAILURE("mkdir failure");
        }
    }



    {
        char fname[STRINGMAXLEN_FILENAME];
        WRITE_FILENAME(fname, "%s/clust.summary.dat", outdname);

        FILE *fp = fopen(fname, "w");

        fprintf(fp,"# col1   leaf CF index\n");
        fprintf(fp,"# col2   Number of point within leaf CF\n");
        fprintf(fp,"# col3   datasq\n");
        fprintf(fp,"# col4   radius2\n");
        fprintf(fp,"# col5   radius3/threshold\n");


        long NBLFcluster = 0;
        for(long CFindex = 0; CFindex < ctree.NBCF; CFindex++)
        {
            if(ctree.CFarray[CFindex].type == CLUSTER_CF_TYPE_LEAF)
            {
                NBLFcluster++;

                fprintf(fp,
                        "%5ld %5ld %16g %16g    %6.4f\n",
                        CFindex,
                        ctree.CFarray[CFindex].N,
                        (double) ctree.CFarray[CFindex].datassq,
                        (double) sqrt(ctree.CFarray[CFindex].radius2),
                        (double) sqrt(ctree.CFarray[CFindex].radius2) /
                        ctree.T);

                {
                    char fleafname[STRINGMAXLEN_FILENAME];
                    WRITE_FILENAME(fleafname,
                                   "%s/leaf%05ld.dat",
                                   outdname,
                                   CFindex);

                    FILE *fpleaf = fopen(fleafname, "w");

                    fprintf(fpleaf, "# CFindex  %5ld\n", CFindex);
                    fprintf(fpleaf, "# level    %5d\n", ctree.CFarray[CFindex].level);
                    fprintf(fpleaf, "# N        %5ld\n", ctree.CFarray[CFindex].N);
                    fprintf(fpleaf, "# datassq %16g\n", (double) ctree.CFarray[CFindex].datassq);
                    fprintf(fpleaf, "# radius2 %16g\n", (double) ctree.CFarray[CFindex].radius2);


                    for(long frame = 0; frame < NBframe; frame++)
                    {
                        if(frameleafCFindex[frame] == CFindex)
                        {
                            fprintf(fpleaf, "%05ld", frame);
                            fprintf(fpleaf, "\n");
                        }
                    }

                    fclose(fpleaf);
                }
            }
        }

        fclose(fp);

        printf("NB leaf cluster = %ld\n", NBLFcluster);
    }




    {
        char fname[STRINGMAXLEN_FILENAME];
        WRITE_FILENAME(fname, "%s/clust.CF.dat", outdname);

        FILE *fp = fopen(fname, "w");

        fprintf(fp,"# col1   CF index\n");
        fprintf(fp,"# col2   CF type\n");
        fprintf(fp,"# col3   CF level\n");
        fprintf(fp,"# col4   Number of point within CF\n");
        fprintf(fp,"# col5   NBchild\n");
        fprintf(fp,"# col6   datasq\n");
        fprintf(fp,"# col7   radius2\n");
        fprintf(fp,"# col8   radius3/threshold\n");



        for(long CFindex = 0; CFindex < ctree.NBCF; CFindex++)
        {
            if(ctree.CFarray[CFindex].type != CLUSTER_CF_TYPE_UNUSED)
            {

                {
                    // WRITE CF ave file to disk

                    IMGID imgCFave = makeIMGID_2D("CFave", xsize, ysize);
                    createimagefromIMGID(&imgCFave);

                    for(uint64_t ii = 0; ii < xysize; ii++)
                    {
                        imgCFave.im->array.F[ii] = ctree.CFarray[CFindex].datasumvec[ii] / ctree.CFarray[CFindex].N;
                    }

                    char name[STRINGMAXLEN_STREAMNAME];
                    WRITE_IMAGENAME(name, "%s/CF_%03ld.fits", outdname, CFindex);

                    save_fits(imgCFave.name, name);


                }


                fprintf(fp,
                        "%5ld  %2d %2d  %5ld %5d    %16g %16g    %6.4f\n",
                        CFindex,
                        ctree.CFarray[CFindex].type,
                        ctree.CFarray[CFindex].level,
                        ctree.CFarray[CFindex].N,
                        ctree.CFarray[CFindex].NBchild,
                        (double) ctree.CFarray[CFindex].datassq,
                        (double) sqrt(ctree.CFarray[CFindex].radius2),
                        (double) sqrt(ctree.CFarray[CFindex].radius2)/ctree.T
                       );

                /*{
                    char fleafname[STRINGMAXLEN_FILENAME];
                    WRITE_FILENAME(fleafname,
                                   "%s/CF%05ld.dat",
                                   outdname,
                                   CFindex);

                    FILE *fpleaf = fopen(fleafname, "w");

                    fprintf(fpleaf, "# CFindex  %5ld\n", CFindex);
                    fprintf(fpleaf, "# level    %5d\n", ctree.CFarray[CFindex].level);
                    fprintf(fpleaf, "# N        %5ld\n", ctree.CFarray[CFindex].N);
                    fprintf(fpleaf, "# datassq %16g\n", (double) ctree.CFarray[CFindex].datassq);
                    fprintf(fpleaf, "# radius2 %16g\n", (double) ctree.CFarray[CFindex].radius2);


                    for(long frame = 0; frame < NBframe; frame++)
                    {
                        if(frameleafCFindex[frame] == CFindex)
                        {
                            fprintf(fpleaf, "%05ld", frame);
                            fprintf(fpleaf, "\n");
                        }
                    }

                    fclose(fpleaf);
                }
                */
            }

        }

        fclose(fp);
    }


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
    free(datarray0);
    free(datarray1);
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
