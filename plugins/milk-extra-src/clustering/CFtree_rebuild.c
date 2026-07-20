// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "CFmeminit.h"
#include "addCF_to_CF.h"
#include "compute_imdistance_double.h"
#include "droptree.h"
#include "get_availableCFindex.h"
#include "leafnode_attachleaf.h"
#include "node_attachnode.h"
#include "printCFtree.h"
#include "split_CF_node.h"


//#define DEBUGPRINT


errno_t CFtree_rebuild(CLUSTERTREE *ctree, long *frameleafCFindex, long NBframe)
{
    DEBUG_TRACE_FSTART();

    // MERGE LEAVES


    printf("REBUILDING\n");


    // collect current ctree stats
    //
    ctree->nbnode       = 0;
    ctree->nbleaf       = 0;
    ctree->nbleafsingle = 0;
    int maxlevel        = 0;
    for (long cfi = 0; cfi < ctree->NBCF; cfi++)
    {
        if (ctree->CFarray[cfi].type != CLUSTER_CF_TYPE_UNUSED)
        {
            if (ctree->CFarray[cfi].level > maxlevel)
            {
                maxlevel = ctree->CFarray[cfi].level;
            }

            switch (ctree->CFarray[cfi].type)
            {
            case CLUSTER_CF_TYPE_NODE:
                ctree->nbnode++;
                break;

            case CLUSTER_CF_TYPE_LEAF:
                if (ctree->CFarray[cfi].N == 1)
                {
                    ctree->nbleafsingle++;
                }
                ctree->nbleaf++;
                break;
            }
        }
    }
    printf("\n");
    printf("    max level  = %5d\n", maxlevel);
    printf("    nbnode     = %5ld\n", ctree->nbnode);
    printf("    nbleaf     = %5ld (incl %ld singles)\n", ctree->nbleaf, ctree->nbleafsingle);

    printf("\n");


    // agglomerative algorithm works by merging tips of subtrees
    // initially, tips = leafs
    //
    long *tipCFi = (long *) malloc(sizeof(long) * ctree->nbleaf);
    if (tipCFi == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }

    // pairwise distance between tips
    //
    double *tipdist = (double *) malloc(sizeof(double) * ctree->nbleaf * ctree->nbleaf);
    if (tipdist == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }

    long nodeCFi_cnt     = 0;
    long nodeleafCFi_cnt = 0;
    long leafCFi_cnt     = 0;
    for (long cfi = 0; cfi < ctree->NBCF; cfi++)
    {
        switch (ctree->CFarray[cfi].type)
        {
        // modes are erased
        case CLUSTER_CF_TYPE_NODE:
            FUNC_CHECK_RETURN(CFmeminit(ctree, cfi, 0));
            nodeCFi_cnt++;
            break;

        // build index of leafs
        case CLUSTER_CF_TYPE_LEAF:
            tipCFi[leafCFi_cnt]             = cfi;
            ctree->CFarray[cfi].parentindex = -1;
            ctree->CFarray[cfi].level       = 0;
            leafCFi_cnt++;
            break;
        }
    }

    // interleaf distances
    // imageID IDleafdist;
    // create_2Dimage_ID("leafdist", ctree->nbleaf, ctree->nbleaf, &IDleafdist);
#ifdef DEBUGPRINT
    printf("%s      Computing pairwise leaf distance matrix\n", __func__);
#endif
    double maxldist = 0.0;
    for (long lf0 = 0; lf0 < ctree->nbleaf; lf0++)
    {
        tipdist[lf0 * ctree->nbleaf + lf0] = 0.0;
        long cfi0                          = tipCFi[lf0];
        for (long lf1 = lf0 + 1; lf1 < ctree->nbleaf; lf1++)
        {
            long   cfi1    = tipCFi[lf1];
            double distval = 0.0;
            FUNC_CHECK_RETURN(compute_imdistance_double(
                ctree, ctree->CFarray[cfi0].datasumvec, ctree->CFarray[cfi0].N,
                ctree->CFarray[cfi1].datasumvec, ctree->CFarray[cfi1].N, &distval));
            if (distval > maxldist)
            {
                maxldist = distval;
            }
            tipdist[lf1 * ctree->nbleaf + lf0] = distval;
            tipdist[lf0 * ctree->nbleaf + lf1] = distval;
        }
    }

#ifdef DEBUGPRINT
    printf("%s      Done\n", __func__);
#endif


    double minleafdist = 0.0;
    int    nbmergeop   = 0;
    long   lf0cnt      = 10;
    int    opOK        = 1;

    long toptip = 0;
    while (opOK == 1)
    {
#ifdef DEBUGPRINT
        printf("[%5d]\n", __LINE__);
#endif

        opOK = 0;

        // identify closest pair of leafs
        // these are candidates for being merged
        //
        long minleafdist_lf0  = -1;
        long minleafdist_lf1  = -1;
        long minleafdist_cfi0 = -1;
        long minleafdist_cfi1 = -1;
        minleafdist           = maxldist + 1.0e200;
        lf0cnt                = 0;
        for (long lf0 = 0; lf0 < ctree->nbleaf; lf0++)
        {
            long cfi0 = tipCFi[lf0];

            // select tips only
            if ((cfi0 != -1) && (ctree->CFarray[cfi0].parentindex == -1) &&
                (ctree->CFarray[cfi0].type != CLUSTER_CF_TYPE_UNUSED))
            {
                lf0cnt++;
                for (long lf1 = lf0 + 1; lf1 < ctree->nbleaf; lf1++)
                {
                    long cfi1 = tipCFi[lf1];
                    if ((cfi1 != -1) && (ctree->CFarray[cfi1].parentindex == -1) &&
                        (ctree->CFarray[cfi1].type != CLUSTER_CF_TYPE_UNUSED))
                    {
                        double ldist = tipdist[lf1 * ctree->nbleaf + lf0];

                        // add # pt regularization
                        //ldist += ctree->CFarray[cfi0].N * ctree->T;
                        //ldist += ctree->CFarray[cfi1].N * ctree->T;

                        if (ldist < minleafdist)
                        {
                            minleafdist_lf0  = lf0;
                            minleafdist_cfi0 = cfi0;
                            minleafdist_lf1  = lf1;
                            minleafdist_cfi1 = cfi1;
                            minleafdist      = ldist;
                        }
                    }
                }
            }
        }
        if (minleafdist_lf0 == -1)
        {
            break;
        }

        long lf0       = minleafdist_lf0;
        int  lf0update = 0; // 1 if lf0 changes
        long lf1       = minleafdist_lf1;
        int  lf1update = 0; // 1 if lf1 changes
        long cfi0      = minleafdist_cfi0;
        long cfi1      = minleafdist_cfi1;


#ifdef DEBUGPRINT
        /*        printf(
                    "[%3ld] Minimum intertip distance  "
                    "CF %ld (N=%ld R=%6.4f)- CF %ld (N=%ld R=%6.4f) = %lf  ( dist = "
                    "%6.4f )\n",
                    lf0cnt,
                    minleafdist_cfi0,
                    ctree->CFarray[minleafdist_cfi0].N,
                    sqrt(ctree->CFarray[cfi0].radius2) / ctree->T,
                    minleafdist_cfi1,
                    ctree->CFarray[minleafdist_cfi1].N,
                    sqrt(ctree->CFarray[cfi1].radius2) / ctree->T,
                    minleafdist,
                    minleafdist / ctree->T);
        */
#endif

        // If both tips are leaf type, try to merge them into a single leaf
        //
        //
        if ((ctree->CFarray[cfi0].type == CLUSTER_CF_TYPE_LEAF) &&
            (ctree->CFarray[cfi1].type == CLUSTER_CF_TYPE_LEAF))
        {
#ifdef DEBUGPRINT
            printf("    LEAF - LEAF\n");
#endif

            // Trying to merge cfi1 into cfi0
            int addOK = 0;

            FUNC_CHECK_RETURN(addCF_to_CF(ctree, ctree->CFarray[cfi1], cfi0, &addOK));
            if (addOK == 1)
            {
#ifdef DEBUGPRINT
                printf("        LEAF MERGE CF %ld into CF %ld\n", cfi1, cfi0);
#endif
                opOK = 1;
                nbmergeop++;

                // keep track of frames origin
                for (long fr = 0; fr < NBframe; fr++)
                {
                    if (frameleafCFindex[fr] == cfi1)
                    {
                        frameleafCFindex[fr] = cfi0;
                    }
                }
                // removing cfi1
                FUNC_CHECK_RETURN(CFmeminit(ctree, cfi1, 0));
#ifdef DEBUGPRINT
//                printf("        -> CF %ld   N =  %ld  R = %6.4f\n",
//                       cfi0,
//                       ctree->CFarray[cfi0].N,
//                       sqrt(ctree->CFarray[cfi0].radius2) / ctree->T);
#endif

                lf0update = 1;
                // discard lf1
                tipCFi[lf1] = -1;

                toptip = cfi0;
            }
            else
            {
#ifdef DEBUGPRINT
                printf("        INSERT IN NODE\n");
#endif

                opOK = 1;

                // Create empty node ncfi
                long ncfi;
                FUNC_CHECK_RETURN(get_availableCFindex(ctree, &ncfi));
                ctree->CFarray[ncfi].type        = CLUSTER_CF_TYPE_NODE;
                ctree->CFarray[ncfi].level       = 1;
                ctree->CFarray[ncfi].parentindex = -1;
                ctree->CFarray[ncfi].NBchild     = 0;
                ctree->CFarray[ncfi].N           = 0;

                // attach cfi0 and cfi1 to new node
                FUNC_CHECK_RETURN(node_attachleaf(ctree, cfi0, ncfi));
                FUNC_CHECK_RETURN(node_attachleaf(ctree, cfi1, ncfi));

                // point lf0 new CF indices
                tipCFi[lf0] = ncfi;
                lf0update   = 1;
                // and discard lf1
                tipCFi[lf1] = -1;

                toptip = ncfi;
            }
#ifdef DEBUGPRINT
            printf("    LEAF - LEAF -> DONE\n\n");
#endif
        }


        if ((ctree->CFarray[cfi0].type == CLUSTER_CF_TYPE_LEAF) &&
            (ctree->CFarray[cfi1].type == CLUSTER_CF_TYPE_NODE))
        {
#ifdef DEBUGPRINT
            printf("    LEAF - NODE\n");
            printf("        attach %ld to %ld\n", cfi0, cfi1);
#endif
            opOK = 1;
            FUNC_CHECK_RETURN(node_attachleaf(ctree, cfi0, cfi1));

            if (ctree->CFarray[cfi1].NBchild > ctree->B)
            {
#ifdef DEBUGPRINT
                printf("%s          MAX LEAF NUMBER REACHED -> SPLIT LEAFNODE\n", __func__);
#endif
                long splitcfi0;
                long splitcfi1;
                FUNC_CHECK_RETURN(split_CF_node(ctree, cfi1, &splitcfi0, &splitcfi1));

                // point to new CF indices
                tipCFi[lf0] = splitcfi0;
                tipCFi[lf1] = splitcfi1;
            }
        }


        if ((ctree->CFarray[cfi0].type == CLUSTER_CF_TYPE_NODE) &&
            (ctree->CFarray[cfi1].type == CLUSTER_CF_TYPE_LEAF))
        {
#ifdef DEBUGPRINT
            printf("    NODE - LEAF\n");
            printf("        attach %ld to %ld\n", cfi1, cfi0);
#endif
            opOK = 1;
            FUNC_CHECK_RETURN(node_attachleaf(ctree, cfi1, cfi0));
            toptip = cfi0;

            if (ctree->CFarray[cfi0].NBchild > ctree->B)
            {
#ifdef DEBUGPRINT
                printf("%s         MAX BRANCH NUMBER REACHED -> SPLIT NODE %ld\n", __func__, cfi0);
#endif
                long splitcfi0;
                long splitcfi1;
                FUNC_CHECK_RETURN(split_CF_node(ctree, cfi0, &splitcfi0, &splitcfi1));

                // point to new CF indices
                tipCFi[lf0] = splitcfi0;
                tipCFi[lf1] = splitcfi1;
            }
        }


        int nodemerge = 0;
        if (opOK == 0)
        {
            if ((ctree->CFarray[cfi0].type == CLUSTER_CF_TYPE_NODE) &&
                (ctree->CFarray[cfi1].type == CLUSTER_CF_TYPE_NODE))
            {
#ifdef DEBUGPRINT
                printf("    NODE - NODE\n");
#endif
                nodemerge = 1;
            }
        }


        if (nodemerge == 1)
        {
            opOK = 1;
            // Create empty node
            long ncfi;
            FUNC_CHECK_RETURN(get_availableCFindex(ctree, &ncfi));
            ctree->CFarray[ncfi].type        = CLUSTER_CF_TYPE_NODE;
            ctree->CFarray[ncfi].level       = 1;
            ctree->CFarray[ncfi].parentindex = -1;
            ctree->CFarray[ncfi].NBchild     = 0;
            ctree->CFarray[ncfi].N           = 0;

            // attach cfi0 and cfi1 to new node
            FUNC_CHECK_RETURN(node_attachnode(ctree, cfi0, ncfi));
            FUNC_CHECK_RETURN(node_attachnode(ctree, cfi1, ncfi));

            // point lf0 new CF indices
            tipCFi[lf0] = ncfi;
            lf0update   = 1;
            // and discard lf1
            tipCFi[lf1] = -1;

            toptip = ncfi;
        }


        // update distance matrix

        if (lf0update == 1)
        {
            // update distances involving cfi0
            //
            for (long lf = 0; lf < ctree->nbleaf; lf++)
            {
                long cfi = tipCFi[lf];
                if ((cfi != -1) && (ctree->CFarray[cfi].type != CLUSTER_CF_TYPE_UNUSED) &&
                    (cfi != cfi0))
                {
                    double distval = 0.0;
                    FUNC_CHECK_RETURN(compute_imdistance_double(
                        ctree, ctree->CFarray[cfi0].datasumvec, ctree->CFarray[cfi0].N,
                        ctree->CFarray[cfi].datasumvec, ctree->CFarray[cfi].N, &distval));
                    tipdist[lf * ctree->nbleaf + lf0] = distval;
                    tipdist[lf0 * ctree->nbleaf + lf] = distval;
                }
            }
        }

        if (lf1update == 1)
        {
            // update distances involving cfi1
            //
            for (long lf = 0; lf < ctree->nbleaf; lf++)
            {
                long cfi = tipCFi[lf];
                if ((cfi != -1) && (ctree->CFarray[cfi].type != CLUSTER_CF_TYPE_UNUSED) &&
                    (cfi != cfi1))
                {
                    double distval = 0.0;
                    FUNC_CHECK_RETURN(compute_imdistance_double(
                        ctree, ctree->CFarray[cfi1].datasumvec, ctree->CFarray[cfi1].N,
                        ctree->CFarray[cfi].datasumvec, ctree->CFarray[cfi].N, &distval));
                    tipdist[lf * ctree->nbleaf + lf1] = distval;
                    tipdist[lf1 * ctree->nbleaf + lf] = distval;
                }
            }
        }


        if (tipCFi[lf0] == -1)
        {
            for (long lf = 0; lf < ctree->nbleaf; lf++)
            {
                tipdist[lf * ctree->nbleaf + lf0] = 0.0;
                tipdist[lf0 * ctree->nbleaf + lf] = 0.0;
            }
        }
        if (tipCFi[lf1] == -1)
        {
            for (long lf = 0; lf < ctree->nbleaf; lf++)
            {
                tipdist[lf * ctree->nbleaf + lf1] = 0.0;
                tipdist[lf1 * ctree->nbleaf + lf] = 0.0;
            }
        }

        /*for(long lf0=0; lf0 < ctree->nbleaf; lf0 ++)
        {
            for(long lf1=0; lf1 < ctree->nbleaf; lf1 ++)
            {
                data.image[IDleafdist].array.F[lf1*ctree->nbleaf + lf0] =
                    leafdist[lf1*ctree->nbleaf + lf0] / ctree->T;
            }
        }
        save_fl_fits("leafdist", "leafdist1.fits");*/
    }


    {
        int leveloffset = ctree->CFarray[toptip].level;
        for (long cfi = 0; cfi < ctree->NBCF; cfi++)
        {
            if (ctree->CFarray[cfi].type != CLUSTER_CF_TYPE_UNUSED)
            {
                ctree->CFarray[cfi].level -= leveloffset;
            }
        }
    }
    ctree->rootindex = toptip;

    if (ctree->CFarray[toptip].type == CLUSTER_CF_TYPE_LEAF)
    {
        droptree(ctree);
    }


    //#ifdef DEBUGPRINT
    printf("NUMBER OF MERGE OPERATION = %d\n", nbmergeop);
    printf("TOP CF INDEX = %ld\n", toptip);

    {
        ctree->nbnode       = 0;
        ctree->nbleaf       = 0;
        ctree->nbleafsingle = 0;
        int maxlevel        = 0;
        for (long cfi = 0; cfi < ctree->NBCF; cfi++)
        {
            if (ctree->CFarray[cfi].type != CLUSTER_CF_TYPE_UNUSED)
            {
                if (ctree->CFarray[cfi].level > maxlevel)
                {
                    maxlevel = ctree->CFarray[cfi].level;
                }

                switch (ctree->CFarray[cfi].type)
                {
                case CLUSTER_CF_TYPE_NODE:
                    ctree->nbnode++;
                    if (ctree->CFarray[cfi].level > maxlevel)
                    {
                        maxlevel = ctree->CFarray[cfi].level;
                    }
                    break;

                case CLUSTER_CF_TYPE_LEAF:
                    if (ctree->CFarray[cfi].N == 1)
                    {
                        ctree->nbleafsingle++;
                    }
                    if (ctree->CFarray[cfi].level > maxlevel)
                    {
                        maxlevel = ctree->CFarray[cfi].level;
                    }
                    ctree->nbleaf++;
                    break;
                }
            }
        }
        printf("\n");
        printf("    max level  = %d\n", maxlevel);
        printf("    nbnode     = %5ld\n", ctree->nbnode);
        printf("    nbleaf     = %5ld (incl %ld singles)\n", ctree->nbleaf, ctree->nbleafsingle);
        printf("\n");
    }
    //#endif


#ifdef DEBUGPRINT
    /*    {
            // list leaves
            printf("\n\n ---------- leaf CFs -------------------------------\n");
            for(long cfi = 0; cfi < ctree->NBCF; cfi++)
            {
                if(ctree->CFarray[cfi].type == CLUSTER_CF_TYPE_LEAF)
                {
                    printf("[CF %4ld]  N= %4ld   R= %6.4f | ",
                           cfi,
                           ctree->CFarray[cfi].N,
                           sqrt(ctree->CFarray[cfi].radius2) / ctree->T);
                    for(long fr = 0; fr < NBframe; fr++)
                    {
                        if(frameleafCFindex[fr] == cfi)
                        {
                            printf(" %5ld", fr);
                        }
                    }
                    printf("\n");
                }
            }
        }
    */
#endif

#ifdef DEBUGPRINT
//    FUNC_CHECK_RETURN(printCFtree(ctree));
#endif

    free(tipdist);
    free(tipCFi);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
