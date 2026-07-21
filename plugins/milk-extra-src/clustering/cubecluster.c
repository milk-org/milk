// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file cubecluster.c
 * @brief Cubecluster module
 */

#define _GNU_SOURCE
#include <math.h>
#include <sys/stat.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cubecluster.h"

#include "CLIcore.h"
#include "clustering_defs.h"
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "processinfo.h"
#include "ImageStreamIO/ImageStreamIO.h"

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


/* ================================================================
 * 1.  FPS COMPONENT IDENTITY
 * ============================================================= */

static FPS_APP_INFO FPS_app_info = {
    .fps_name         = "cubeclust",
    .cmdkey           = "cubeclust",
    .description      = "compute cube cluster",
    .description_long = "Cluster image cube slices by similarity using k-means or hierarchical "
                        "clustering. Groups similar frames together."
};


/* ================================================================
 * 2.  LOCAL PARAMETER VARIABLES
 * ============================================================= */

static char     *farg_inimname = NULL;
static char     *farg_outdname = NULL;
static float    *threshold     = NULL;
static uint32_t *branchB       = NULL;
static uint32_t *leafposmode   = NULL;
static uint32_t *NBCFmax       = NULL;
static int64_t  *optrebuild    = NULL;
static int64_t  *optcondense   = NULL;


/* ================================================================
 * 3.  UNIFIED PARAMETER TABLE (X-Macro)
 * ============================================================= */

#define FPS_PARAMS(X)                                                                             \
    X(".in_name", &farg_inimname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT, "input image cube") \
    X(".outdname", &farg_outdname, FPTYPE_STREAMNAME, 1, FPFLAG_DEFAULT_INPUT,                    \
      "output directory name")                                                                    \
    X(".T", &threshold, FPTYPE_FLOAT32, 0, FPFLAG_DEFAULT_INPUT, "threshold")                     \
    X(".B", &branchB, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT, "branch number")                    \
    X(".leafposmode", &leafposmode, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT, "leaf position mode") \
    X(".NBCFmax", &NBCFmax, FPTYPE_UINT32, 0, FPFLAG_DEFAULT_INPUT, "max number of CFs")          \
    X(".opt.rebuild", &optrebuild, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,                         \
      "rebuild tree after scan")                                                                  \
    X(".opt.condense", &optcondense, FPTYPE_ONOFF, 0, FPFLAG_DEFAULT_INPUT,                       \
      "condense tree after scan")

#define pathprobdecay 0.95

static errno_t ctree_check(CLUSTERTREE *ctree)
{
    for (long cfi = 0; cfi < ctree->NBCF; cfi++)
    {
        if (ctree->CFarray[cfi].type != CLUSTER_CF_TYPE_UNUSED)
        {
            if (ctree->CFarray[cfi].N < 1)
            {
                return RETURN_FAILURE;
            }
            if (ctree->CFarray[cfi].type == CLUSTER_CF_TYPE_NODE)
            {
                if (ctree->CFarray[cfi].NBchild > ctree->B)
                {
                    return RETURN_FAILURE;
                }
            }
        }
    }
    return RETURN_SUCCESS;
}

static errno_t findleafnode(CLUSTERTREE *ctree, double *datavec, long *nodeindex, double *distance)
{
    static long findleadnodecnt = 0;
    static long CFindex         = 0;
    int         solution_reuse  = 0;
    if (findleadnodecnt != 0 && ctree->leafposmode == CLUSTER_CFPOS_FIXED)
    {
        double distval = 0.0;
        compute_imdistance_double(ctree, ctree->CFarray[CFindex].dataposvec, 1, datavec, 1,
                                  &distval);
        if (distval < ctree->T)
        {
            solution_reuse = 1;
            *distance      = distval;
        }
    }
    if (solution_reuse == 0)
    {
        int level = 0;
        CFindex   = ctree->rootindex;
        while (ctree->CFarray[CFindex].NBchild > 0)
        {
            int    scaninit         = 0;
            double distvalmin       = 0;
            long   CFindexbest      = 0;
            ctree->path_node[level] = CFindex;
            double *mindistarray =
                (double *) malloc(sizeof(double) * ctree->CFarray[CFindex].NBchild);
            double *maxdistarray =
                (double *) malloc(sizeof(double) * ctree->CFarray[CFindex].NBchild);
            distvalmin = 2.0 * ctree->T;
            for (long childi = 0; childi < ctree->CFarray[CFindex].NBchild; childi++)
            {
                double distval  = 0.0;
                long   CFindex1 = ctree->CFarray[CFindex].childindex[childi];
                int    skipflag = 0;
                if (scaninit != 0 && ctree->leafposmode == CLUSTER_CFPOS_FIXED &&
                    distvalmin < mindistarray[childi])
                {
                    skipflag = 1;
                }
                if (distvalmin < ctree->T)
                {
                    skipflag = 1;
                }
                if (skipflag == 0)
                {
                    if (ctree->leafposmode == CLUSTER_CFPOS_DYNAMIC)
                    {
                        compute_imdistance_double(ctree, ctree->CFarray[CFindex1].datasumvec,
                                                  ctree->CFarray[CFindex1].N, datavec, 1, &distval);
                    }
                    else
                    {
                        compute_imdistance_double(ctree, ctree->CFarray[CFindex1].dataposvec, 1,
                                                  datavec, 1, &distval);
                    }
                    if (scaninit == 0)
                    {
                        distvalmin           = distval;
                        CFindexbest          = CFindex1;
                        mindistarray[childi] = distval;
                        maxdistarray[childi] = distval;
                        for (long ci = 0; ci < ctree->CFarray[CFindex].NBchild; ci++)
                        {
                            if (ci != childi)
                            {
                                long   CFindex2 = ctree->CFarray[CFindex].childindex[ci];
                                double dval;
                                compute_CF2CF_posdistance_double(ctree, CFindex1, CFindex2, &dval);
                                double minval = dval - distval;
                                if (minval < 0.0)
                                {
                                    minval = 0.0;
                                }
                                double maxval    = dval + distval;
                                mindistarray[ci] = minval;
                                maxdistarray[ci] = maxval;
                            }
                        }
                        scaninit = 1;
                    }
                    else
                    {
                        mindistarray[childi] = distval;
                        maxdistarray[childi] = distval;
                        for (long ci = 0; ci < ctree->CFarray[CFindex].NBchild; ci++)
                        {
                            if (ci != childi)
                            {
                                long   CFindex2 = ctree->CFarray[CFindex].childindex[ci];
                                double dval;
                                compute_CF2CF_posdistance_double(ctree, CFindex1, CFindex2, &dval);
                                double minval = dval - distval;
                                if (minval < 0.0)
                                {
                                    minval = 0.0;
                                }
                                double maxval = dval + distval;
                                if (minval > mindistarray[ci])
                                {
                                    mindistarray[ci] = minval;
                                }
                                if (maxval < maxdistarray[ci])
                                {
                                    maxdistarray[ci] = maxval;
                                }
                            }
                        }
                        if (distval < distvalmin)
                        {
                            distvalmin  = distval;
                            CFindexbest = CFindex1;
                        }
                    }
                }
            }
            free(mindistarray);
            free(maxdistarray);
            CFindex   = CFindexbest;
            *distance = distvalmin;
            level++;
        }
    }
    *nodeindex = CFindex;
    findleadnodecnt++;
    return RETURN_SUCCESS;
}

static errno_t imcube_makecluster_core(IMAGE *im, const char *__restrict outdname)
{
    uint32_t xsize = im->md[0].size[0], ysize = im->md[0].size[1], zsize = im->md[0].size[2];
    uint64_t xysize = (uint64_t) xsize * ysize;
    if (zsize == 0)
    {
        xysize = xsize;
        zsize  = ysize;
    }
    IMAGE immask;
    if (ImageStreamIO_read_sharedmem_image_toIMAGE("maskim", &immask) != 0)
    {
        ImageStreamIO_createIm(&immask, "maskim", 2, (uint32_t[]) { xsize, ysize }, _DATATYPE_FLOAT,
                               1, 10, 0);
        for (uint64_t ii = 0; ii < xysize; ii++)
        {
            immask.array.F[ii] = 1.0f;
        }
    }
    float maskeps = 1.0e-5;
    long  CF_npix = 0;
    for (uint64_t ii = 0; ii < xysize; ii++)
    {
        if (immask.array.F[ii] > maskeps)
        {
            CF_npix++;
        }
    }
    long   *pixmap     = (long *) malloc(sizeof(long) * CF_npix);
    double *pixgain    = (double *) malloc(sizeof(double) * CF_npix);
    long    inpixindex = 0;
    for (uint64_t ii = 0; ii < xysize; ii++)
    {
        if (immask.array.F[ii] > maskeps)
        {
            pixmap[inpixindex]  = ii;
            pixgain[inpixindex] = immask.array.F[ii];
            inpixindex++;
        }
    }
    CLUSTERTREE ctree;
    ctree.xsize            = xsize;
    ctree.ysize            = ysize;
    ctree.NBCF             = *NBCFmax;
    ctree.B                = *branchB;
    ctree.noise2offset     = 0.0;
    ctree.T                = *threshold;
    ctree.leafposmode      = *leafposmode;
    ctree.npix             = CF_npix;
    ctree.stat_compdistcnt = 0;
    ctree_memallocate(&ctree);
    double *datarray         = (double *) malloc(sizeof(double) * CF_npix);
    long    NBframe          = zsize;
    long   *frameleafCFindex = (long *) malloc(sizeof(long) * NBframe);
    for (long frame = 0; frame < NBframe; frame++)
    {
        frameleafCFindex[frame] = -1;
        ctree_check(&ctree);
        long double ssqr = 0.0;
        if (im->md[0].datatype == _DATATYPE_FLOAT)
        {
            for (long ii = 0; ii < CF_npix; ii++)
            {
                datarray[ii] = pixgain[ii] * im->array.F[frame * xysize + pixmap[ii]];
                ssqr += datarray[ii] * datarray[ii];
            }
        }
        else if (im->md[0].datatype == _DATATYPE_DOUBLE)
        {
            for (long ii = 0; ii < CF_npix; ii++)
            {
                datarray[ii] = pixgain[ii] * im->array.D[frame * xysize + pixmap[ii]];
                ssqr += datarray[ii] * datarray[ii];
            }
        }
        if (frame == 0)
        {
            ctree_init(&ctree, datarray, ssqr);
            frameleafCFindex[0] = 1;
        }
        else
        {
            long   CFindex;
            double distance = 0.0;
            findleafnode(&ctree, datarray, &CFindex, &distance);
            long lCFindex = CFindex;
            int  addOK    = 0;
            leaf_addentry(&ctree, datarray, ssqr, lCFindex, &addOK, distance);
            if (addOK == 1)
            {
                frameleafCFindex[frame] = lCFindex;
            }
            else
            {
                long nCFindex;
                create_new_leaf(&ctree, datarray, ssqr, &nCFindex);
                frameleafCFindex[frame] = nCFindex;
                node_attachleaf(&ctree, nCFindex, ctree.CFarray[CFindex].parentindex);
                CFindex = ctree.CFarray[CFindex].parentindex;
                while (CFindex != -1 && ctree.CFarray[CFindex].NBchild == ctree.B + 1)
                {
                    long CFi0, CFi1;
                    split_CF_node(&ctree, CFindex, &CFi0, &CFi1);
                    if (ctree.CFarray[CFindex].level == 0)
                    {
                        droptree(&ctree);
                        break;
                    }
                    CFindex = ctree.CFarray[CFi0].parentindex;
                }
            }
            long cfi = frameleafCFindex[frame];
            while (cfi != -1)
            {
                ctree.CFarray[cfi].pathcnt += 1.0;
                ctree.CFarray[cfi].pathdistcompcnt += 1.0;
                cfi = ctree.CFarray[cfi].parentindex;
            }
            int condensenop = 1;
            while (condensenop > 0)
            {
                ctree_condense(&ctree, &condensenop);
            }
        }
        for (long cfi = 0; cfi < ctree.NBCF; cfi++)
        {
            ctree.CFarray[cfi].status = 0;
            ctree.CFarray[cfi].pathcnt *= pathprobdecay;
        }
    }
    if (*optrebuild == 1)
    {
        CFtree_rebuild(&ctree, frameleafCFindex, NBframe);
    }
    if (*optcondense == 1)
    {
        int condensenop = 1;
        while (condensenop > 0)
        {
            ctree_condense(&ctree, &condensenop);
        }
    }
    mkdir(outdname, 0777);

    IMGID imgdummy;
    imgdummy.im = im;
    imgdummy.md = &im->md[0];
    imgdummy.ID = 0;
    strncpy(imgdummy.name, im->md[0].name, STRINGMAXLEN_IMAGE_NAME - 1);

    write_clustleafsummary(&ctree, imgdummy, pixmap, pixgain, frameleafCFindex, NBframe, outdname);
    char fname[STRINGMAXLEN_FILENAME];
    WRITE_FILENAME(fname, "%s/clust.CF.dat", outdname);
    write_clustCFdat(&ctree, fname);
    write_clustCFave(&ctree, outdname);
    free(frameleafCFindex);
    free(pixmap);
    free(pixgain);
    ctree_memfree(&ctree);
    free(datarray);
    return RETURN_SUCCESS;
}

/* ================================================================
 * 4.  COMPUTATION LOGIC (above)
 * ============================================================= */


/* ================================================================
 * 5.  BINDINGS, FARG, AND CLI DATA
 * ============================================================= */

FPS_V2_SECTION5(FPS_PARAMS)


/* ================================================================
 * 6.  COMPUTE WRAPPER
 * ============================================================= */

static MILK_HOT errno_t compute_function()
{
    IMGID img = imgid_make_from_name(farg_inimname);
    if (resolveIMGID(&img, ERRMODE_WARN, dcimg, dcnimg) != 0)
    {
        return RETURN_FAILURE;
    }
    imcube_makecluster_core(img.im, farg_outdname);
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

errno_t CLIADDCMD_clustering__imcube_mkcluster()
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
