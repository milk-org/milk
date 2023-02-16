/**
 * @file    cubecluster.c
 * @brief   build cluster from image cube
 *
 * Use 3rd dimension as index
 */

#include <math.h>
#include <sys/stat.h>

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

static char *farg_inimname;
static char *farg_outdname;

// List of arguments to function
//
static CLICMDARGDEF farg[] = {{
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

            if(ctree->CFarray[cfi].type == CLUSTER_CF_TYPE_LEAFNODE)
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
            }
        }
    }

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

    DEBUG_TRACEPOINT("root CF = %ld, has %d child",
                     CFindex,
                     ctree->CFarray[CFindex].NBchild);

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

            DEBUG_TRACEPOINT("computing distance to CF node # %ld(%ld)",
                             CFindex1,
                             ctree->CFarray[CFindex1].N);

            FUNC_CHECK_RETURN(
                compute_imdistance_double(ctree,
                                          ctree->CFarray[CFindex1].datasumvec,
                                          ctree->CFarray[CFindex1].N,
                                          datavec,
                                          1,
                                          &distval));

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

        DEBUG_TRACEPOINT("level %3d %4ld  %g", level, CFindexbest, distvalmin);

        CFindex = CFindexbest;
        level++;
    }

    DEBUG_TRACEPOINT("NEAREST NODE : %ld  ( nbchild=%3d  nbleaf=%3d)",
                     CFindex,
                     ctree->CFarray[CFindex].NBchild,
                     ctree->CFarray[CFindex].NBleaf);
    *nodeindex = CFindex;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static errno_t
findleaf(CLUSTERTREE *ctree, double *datavec, long CFindex, long *leafindex)
{
    DEBUG_TRACE_FSTART();

    int    leafimin     = -1; // leaf index into which entry will be added
    double distvalmin   = 0;
    int    leafloopinit = 0;
    for(long leafi = 0; leafi < ctree->CFarray[CFindex].NBleaf; leafi++)
    {
        long CFindex1 = ctree->CFarray[CFindex].leafindex[leafi];
        //if(sqrt(ctree->CFarray[CFindex1].radius2) < ctree->T)
        //{   // radius below threahold -> we can add
        double distval = 0.0;

        DEBUG_TRACEPOINT("computing distance to leaf %ld = node # %ld(%ld)",
                         leafi,
                         CFindex1,
                         ctree->CFarray[CFindex1].N);

        FUNC_CHECK_RETURN(
            compute_imdistance_double(ctree,
                                      ctree->CFarray[CFindex1].datasumvec,
                                      ctree->CFarray[CFindex1].N,
                                      datavec,
                                      1,
                                      &distval));

        DEBUG_TRACEPOINT("dist %4ld(%3ld) - new sample : %g",
                         CFindex1,
                         ctree->CFarray[CFindex1].N,
                         (double) distval);

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

    if(distvalmin > ctree->T)
    {
        //printf("        distance too large -> can't add to closest leaf\n");
        leafimin = -1;
    }

    DEBUG_TRACEPOINT("leafimin = %d", leafimin);

    *leafindex = leafimin;

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

static errno_t imcube_makecluster(IMGID img, const char *__restrict outdname)
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

    printf("image size %u %u %u\n", xsize, ysize, zsize);

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

    ctree.NBCF         = 10000;
    ctree.B            = 10; // max number of branches out of node
    ctree.L            = 10; // max numbers of CF entries in leaf node
    ctree.noise2offset = 2.0e10;
    ctree.T = 100000.0; // threshold satisfied by each CF entry of leaf node

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
    long *frameleafCFindex = (long *) malloc(sizeof(long) * NBframe);
    if(frameleafCFindex == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }

    long framecnt = 0;
    for(long frame = 0; frame < NBframe; frame++)
    {
        frameleafCFindex[frame] = -1;

        FUNC_CHECK_RETURN(ctree_check(&ctree));

        // Load image data into vector
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
            /*printf("\n skipping ID %5ld frame %5ld  :  %16Lg  %16Lg -> %16Lg   \n",
                   img.ID, frame,
                   ssqrdiff,
                   ssqr,
                   ssqrdiff/ssqr);
                   */
            frameskip = 1;
        }
        if(frame == 0)
        {
            frameskip = 0;
        }

        if(frameskip == 0)
        {
            printf("Processing ID %ld frame %ld, %ld pix    \r",
                   img.ID,
                   frame,
                   CF_npix);
            /*printf("---------------- %16Lg  %16Lg -> %16Lg\n",
                   ssqrdiff,
                   ssqr,
                   ssqrdiff/ssqr);*/

            //printf("    SSWR = %g\n", (double) ssqr);

            // INITIALIZATION
            if(frame == 0)
            {
                ctree_init(&ctree, datarray, ssqr);
                frameleafCFindex[0] = 2;
            }
            else
            {
                long CFindex;
                FUNC_CHECK_RETURN(findleafnode(&ctree, datarray, &CFindex));
                DEBUG_TRACEPOINT("CF %ld type is %d",
                                 CFindex,
                                 ctree.CFarray[CFindex].type);
                // we have descended the tree and are now at a leaf node

                long leafi;
                FUNC_CHECK_RETURN(findleaf(&ctree, datarray, CFindex, &leafi));

                // we have descended the tree and are now at a leaf node

                if(leafi != -1)
                {
                    long lCFindex = ctree.CFarray[CFindex].leafindex[leafi];

                    // only add if radius condition is met
                    int addOK = 0;
                    FUNC_CHECK_RETURN(leaf_addentry(&ctree,
                                                    datarray,
                                                    ssqr,
                                                    lCFindex,
                                                    &addOK));

                    if(addOK == 1)
                    {
                        // leaf has been added
                        frameleafCFindex[frame] = lCFindex;
                        DEBUG_TRACEPOINT("Added entry to leaf index %ld",
                                         leafi);
                    }
                    else
                    {
                        DEBUG_TRACEPOINT(
                            "Radius condition not met for leaf index %ld",
                            leafi);
                        // indicate that leaf has not been added
                        leafi = -1;
                    }
                }

                if(leafi == -1)
                {

                    DEBUG_TRACEPOINT("Creating new leaf # %d",
                                     ctree.CFarray[CFindex].NBleaf);
                    long nCFindex;
                    FUNC_CHECK_RETURN(
                        create_new_leaf(&ctree, datarray, ssqr, &nCFindex));
                    frameleafCFindex[frame] = nCFindex;
                    DEBUG_TRACEPOINT("CREATED LEAF at index %ld", nCFindex);

                    DEBUG_TRACEPOINT("ATTACHING LEAF %ld to %ld",
                                     nCFindex,
                                     CFindex);

                    FUNC_CHECK_RETURN(
                        leafnode_attachleaf(&ctree, nCFindex, CFindex));

                    DEBUG_TRACEPOINT("ATTACHED LEAF %ld to %ld",
                                     nCFindex,
                                     CFindex);

                    if(ctree.CFarray[CFindex].NBleaf == ctree.L + 1)
                    {
                        DEBUG_TRACEPOINT(
                            "MAX LEAF NUMBER REACHED -> SPLIT LEAFNODE");

                        long CFi0;
                        long CFi1;
                        FUNC_CHECK_RETURN(
                            split_CF_node(&ctree, CFindex, &CFi0, &CFi1));

                        DEBUG_TRACEPOINT("LEAFNODE %ld(%d) -> %ld(%d) %ld(%d)",
                                         CFindex,
                                         ctree.CFarray[CFindex].NBleaf,
                                         CFi0,
                                         ctree.CFarray[CFi0].NBleaf,
                                         CFi1,
                                         ctree.CFarray[CFi1].NBleaf);

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
                FUNC_CHECK_RETURN(ctree_condense(&ctree, &condensenop));
            }

            FUNC_CHECK_RETURN(printCFtree(&ctree));

            framecnt++;

            if(framecnt % 200 == 0)
            {
                FUNC_CHECK_RETURN(
                    CFtree_rebuild(&ctree, frameleafCFindex, NBframe));
            }
        }
    }

    printf("\n");
    printf("Processed %ld / %ld frames\n", framecnt, NBframe);

    FUNC_CHECK_RETURN(printCFtree(&ctree));

    FUNC_CHECK_RETURN(CFtree_rebuild(&ctree, frameleafCFindex, NBframe));

    // TEST print
    printCFtree(&ctree);

    if(mkdir(outdname, 0777) != 0)
    {
        FUNC_RETURN_FAILURE("mkdir failure");
    }

    {
        char fname[STRINGMAXLEN_FILENAME];
        WRITE_FILENAME(fname, "%s/clust.summary.dat", outdname);

        FILE *fp = fopen(fname, "w");

        for(long CFindex = 0; CFindex < ctree.NBCF; CFindex++)
        {
            if(ctree.CFarray[CFindex].type == CLUSTER_CF_TYPE_LEAF)
            {
                //float xave = ctree.CFarray[CFindex].datasumvec[0] / ctree.CFarray[CFindex].N;
                //float yave = ctree.CFarray[CFindex].datasumvec[1] / ctree.CFarray[CFindex].N;
                fprintf(fp,
                        "%5ld %5ld %16g %16g %6.4f\n",
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
                    fprintf(fpleaf,
                            "# %5ld %4d %5ld %16g %16g\n",
                            CFindex,
                            ctree.CFarray[CFindex].level,
                            ctree.CFarray[CFindex].N,
                            (double) ctree.CFarray[CFindex].datassq,
                            (double) ctree.CFarray[CFindex].radius2);
                    fprintf(fpleaf,
                            "#  %16g  %16g\n",
                            ctree.CFarray[CFindex].datasumvec[0] /
                            ctree.CFarray[CFindex].N,
                            ctree.CFarray[CFindex].datasumvec[1] /
                            ctree.CFarray[CFindex].N);

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
    }

    {
        // measure distance betweeen nodes and leaves

        char fname[STRINGMAXLEN_FILENAME];
        WRITE_FILENAME(fname, "%s/clust.dist.dat", outdname);

        FILE *fp = fopen(fname, "w");

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
                                    "%5ld %5ld      %16g  %6.4f  %6.2f\n",
                                    CFindex0,
                                    CFindex1,
                                    distval,
                                    distval / ctree.T,
                                    1.0 / (1.0 / ctree.CFarray[CFindex0].N +
                                           1.0 / ctree.CFarray[CFindex1].N));
                        }
                    }
                }
            }
        }

        fclose(fp);
    }

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
