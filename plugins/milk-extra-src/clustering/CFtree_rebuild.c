#include <math.h>

#include "CommandLineInterface/CLIcore.h"
#include "clustering_defs.h"

#include "CFmeminit.h"
#include "addvector_to_CF.h"
#include "compute_imdistance_double.h"
#include "droptree.h"
#include "get_availableCFindex.h"
#include "leafnode_attachleaf.h"
#include "node_attachnode.h"
#include "printCFtree.h"
#include "split_CF_node.h"

errno_t CFtree_rebuild(CLUSTERTREE *ctree, long *frameleafCFindex, long NBframe)
{
    DEBUG_TRACE_FSTART();

    // MERGE LEAVES

    ctree->nbnode       = 0;
    ctree->nbleafnode   = 0;
    ctree->nbleaf       = 0;
    ctree->nbleafsingle = 0;
    for(long cfi = 0; cfi < ctree->NBCF; cfi++)
    {
        switch(ctree->CFarray[cfi].type)
        {
            case CLUSTER_CF_TYPE_NODE:
                ctree->nbnode++;
                break;

            case CLUSTER_CF_TYPE_LEAFNODE:
                ctree->nbleafnode++;
                break;

            case CLUSTER_CF_TYPE_LEAF:
                if(ctree->CFarray[cfi].N == 1)
                {
                    ctree->nbleafsingle++;
                }
                ctree->nbleaf++;
                break;
        }
    }
    printf("    nbnode     = %5ld\n", ctree->nbnode);
    printf("    nbleafnode = %5ld\n", ctree->nbleafnode);
    printf("    nbleaf     = %5ld (incl %ld singles)\n",
           ctree->nbleaf,
           ctree->nbleafsingle);

    // agglomerative algorithm works by merging tips of subtrees
    // initially, tips = leafs
    long *tipCFi = (long *) malloc(sizeof(long) * ctree->nbleaf);
    if(tipCFi == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }

    // pairwise distance between tips
    double *tipdist =
        (double *) malloc(sizeof(double) * ctree->nbleaf * ctree->nbleaf);
    if(tipdist == NULL)
    {
        FUNC_RETURN_FAILURE("malloc error");
    }

    long nodeCFi_cnt     = 0;
    long nodeleafCFi_cnt = 0;
    long leafCFi_cnt     = 0;
    for(long cfi = 0; cfi < ctree->NBCF; cfi++)
    {
        switch(ctree->CFarray[cfi].type)
        {
            case CLUSTER_CF_TYPE_NODE:
                //nodeCFi[nodeCFi_cnt] = cfi;
                FUNC_CHECK_RETURN(CFmeminit(ctree, cfi, 0));
                nodeCFi_cnt++;
                break;

            case CLUSTER_CF_TYPE_LEAFNODE:
                //nodeleafCFi[nodeleafCFi_cnt] = cfi;
                FUNC_CHECK_RETURN(CFmeminit(ctree, cfi, 0));
                nodeleafCFi_cnt++;
                break;

            case CLUSTER_CF_TYPE_LEAF:
                tipCFi[leafCFi_cnt]             = cfi;
                ctree->CFarray[cfi].parentindex = -1;
                ctree->CFarray[cfi].level       = 0;
                leafCFi_cnt++;
                break;
        }
    }

    // interleaf distances
    //imageID IDleafdist;
    //create_2Dimage_ID("leafdist", ctree->nbleaf, ctree->nbleaf, &IDleafdist);
    printf("%s      Computing pairwise leaf distance matrix\n", __func__);

    double maxldist = 0.0;
    for(long lf0 = 0; lf0 < ctree->nbleaf; lf0++)
    {
        tipdist[lf0 * ctree->nbleaf + lf0] = 0.0;
        long cfi0                          = tipCFi[lf0];
        for(long lf1 = lf0 + 1; lf1 < ctree->nbleaf; lf1++)
        {
            long   cfi1    = tipCFi[lf1];
            double distval = 0.0;
            FUNC_CHECK_RETURN(
                compute_imdistance_double(ctree,
                                          ctree->CFarray[cfi0].datasumvec,
                                          ctree->CFarray[cfi0].N,
                                          ctree->CFarray[cfi1].datasumvec,
                                          ctree->CFarray[cfi1].N,
                                          &distval));
            if(distval > maxldist)
            {
                maxldist = distval;
            }
            tipdist[lf1 * ctree->nbleaf + lf0] = distval;
            tipdist[lf0 * ctree->nbleaf + lf1] = distval;
        }
    }

    printf("%s      Done\n", __func__);
    /*for(long lf0=0; lf0 < ctree->nbleaf; lf0 ++)
    {
        for(long lf1=0; lf1 < ctree->nbleaf; lf1 ++)
        {
            data.image[IDleafdist].array.F[lf1*ctree->nbleaf + lf0] =
                tipdist[lf1*ctree->nbleaf + lf0] / ctree->T;
        }
    }*/
    //save_fl_fits("leafdist", "leafdist0.fits");

    double minleafdist = 0.0;
    int    nbmergeop   = 0;
    long   lf0cnt      = 10;
    int    opOK        = 1;

    long toptip = 0;
    while(opOK == 1)
    {
        opOK = 0;

        long minleafdist_lf0  = -1;
        long minleafdist_lf1  = -1;
        long minleafdist_cfi0 = -1;
        long minleafdist_cfi1 = -1;
        minleafdist           = maxldist + 1.0e200;
        lf0cnt                = 0;
        for(long lf0 = 0; lf0 < ctree->nbleaf; lf0++)
        {
            long cfi0 = tipCFi[lf0];

            // select tips only
            if((cfi0 != -1) && (ctree->CFarray[cfi0].parentindex == -1) &&
                    (ctree->CFarray[cfi0].type != CLUSTER_CF_TYPE_UNUSED))
            {
                lf0cnt++;
                for(long lf1 = lf0 + 1; lf1 < ctree->nbleaf; lf1++)
                {
                    long cfi1 = tipCFi[lf1];
                    if((cfi1 != -1) &&
                            (ctree->CFarray[cfi1].parentindex == -1) &&
                            (ctree->CFarray[cfi1].type != CLUSTER_CF_TYPE_UNUSED))
                    {
                        double ldist = tipdist[lf1 * ctree->nbleaf + lf0];

                        // add # pt regularization
                        ldist += ctree->CFarray[cfi0].N * ctree->T;
                        ldist += ctree->CFarray[cfi1].N * ctree->T;

                        if(ldist < minleafdist)
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
        if(minleafdist_lf0 == -1)
        {
            break;
        }

        long lf0       = minleafdist_lf0;
        int  lf0update = 0; // 1 if lf0 changes
        long lf1       = minleafdist_lf1;
        int  lf1update = 0; // 1 if lf1 changes
        long cfi0      = minleafdist_cfi0;
        long cfi1      = minleafdist_cfi1;

        printf(
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

        // If both tips are leat type, try to merge them into a single leaf
        if((ctree->CFarray[cfi0].type == CLUSTER_CF_TYPE_LEAF) &&
                (ctree->CFarray[cfi1].type == CLUSTER_CF_TYPE_LEAF))
        {
            printf("    LEAF - LEAF\n");

            // Trying to merge cfi1 into cfi0
            int addOK = 0;

            FUNC_CHECK_RETURN(addvector_to_CF(ctree,
                                              ctree->CFarray[cfi1].datasumvec,
                                              ctree->CFarray[cfi1].datassq,
                                              ctree->CFarray[cfi1].N,
                                              cfi0,
                                              &addOK));
            if(addOK == 1)
            {
                printf("    LEAF MERGE CF %ld into CF %ld\n", cfi1, cfi0);
                opOK = 1;
                nbmergeop++;

                // keep track of frames origin
                for(long fr = 0; fr < NBframe; fr++)
                {
                    if(frameleafCFindex[fr] == cfi1)
                    {
                        frameleafCFindex[fr] = cfi0;
                    }
                }
                // removing cfi1
                FUNC_CHECK_RETURN(CFmeminit(ctree, cfi1, 0));
                printf("        -> CF %ld   N =  %ld  R = %6.4f\n",
                       cfi0,
                       ctree->CFarray[cfi0].N,
                       sqrt(ctree->CFarray[cfi0].radius2) / ctree->T);

                lf0update = 1;
                // discard lf1
                tipCFi[lf1] = -1;

                toptip = cfi0;
            }
            else
            {
                printf("    INSERT IN LEAFNODE\n");
                opOK = 1;

                // Create empty leaf node
                long ncfi;
                FUNC_CHECK_RETURN(get_availableCFindex(ctree, &ncfi));
                ctree->CFarray[ncfi].type        = CLUSTER_CF_TYPE_LEAFNODE;
                ctree->CFarray[ncfi].level       = 1;
                ctree->CFarray[ncfi].parentindex = -1;
                ctree->CFarray[ncfi].NBchild     = 0;
                ctree->CFarray[ncfi].NBleaf      = 0;
                ctree->CFarray[ncfi].N           = 0;

                // attach cfi0 and cfi1 to new node
                FUNC_CHECK_RETURN(leafnode_attachleaf(ctree, cfi0, ncfi));
                FUNC_CHECK_RETURN(leafnode_attachleaf(ctree, cfi1, ncfi));

                // point lf0 new CF indices
                tipCFi[lf0] = ncfi;
                lf0update   = 1;
                // and discard lf1
                tipCFi[lf1] = -1;

                toptip = ncfi;
            }
        }

        if((ctree->CFarray[cfi0].type == CLUSTER_CF_TYPE_LEAF) &&
                (ctree->CFarray[cfi1].type == CLUSTER_CF_TYPE_LEAFNODE))
        {
            printf("    LEAF - LEAFNODE\n");
            printf("    attach %ld to %ld\n", cfi0, cfi1);
            opOK = 1;
            FUNC_CHECK_RETURN(leafnode_attachleaf(ctree, cfi0, cfi1));

            if(ctree->CFarray[cfi1].NBleaf > ctree->L)
            {
#ifdef DEBUGPRINT
                printf("%s      MAX LEAF NUMBER REACHED -> SPLIT LEAFNODE\n",
                       __func__);
#endif
                long splitcfi0;
                long splitcfi1;
                FUNC_CHECK_RETURN(
                    split_CF_node(ctree, cfi1, &splitcfi0, &splitcfi1));

                // point to new CF indices
                tipCFi[lf0] = splitcfi0;
                tipCFi[lf1] = splitcfi1;
            }
        }

        if((ctree->CFarray[cfi0].type == CLUSTER_CF_TYPE_LEAFNODE) &&
                (ctree->CFarray[cfi1].type == CLUSTER_CF_TYPE_LEAF))
        {
            printf("    LEAFNODE - LEAF\n");
            printf("    attach %ld to %ld\n", cfi1, cfi0);
            opOK = 1;
            FUNC_CHECK_RETURN(leafnode_attachleaf(ctree, cfi1, cfi0));
            toptip = cfi0;

            if(ctree->CFarray[cfi0].NBleaf > ctree->L)
            {
#ifdef DEBUGPRINT
                printf(
                    "%s      MAX LEAF NUMBER REACHED -> SPLIT LEAFNODE %ld\n",
                    __func__,
                    cfi0);
#endif
                long splitcfi0;
                long splitcfi1;
                FUNC_CHECK_RETURN(
                    split_CF_node(ctree, cfi0, &splitcfi0, &splitcfi1));

                // point to new CF indices
                tipCFi[lf0] = splitcfi0;
                tipCFi[lf1] = splitcfi1;
            }
        }

        if((ctree->CFarray[cfi0].type == CLUSTER_CF_TYPE_LEAF) &&
                (ctree->CFarray[cfi1].type == CLUSTER_CF_TYPE_NODE))
        {
            printf("    LEAF - NODE\n");

            // Create empty leaf node
            long ncfi;
            FUNC_CHECK_RETURN(get_availableCFindex(ctree, &ncfi));
            ctree->CFarray[ncfi].type        = CLUSTER_CF_TYPE_LEAFNODE;
            ctree->CFarray[ncfi].level       = 1;
            ctree->CFarray[ncfi].parentindex = -1;
            ctree->CFarray[ncfi].NBchild     = 0;
            ctree->CFarray[ncfi].NBleaf      = 0;
            ctree->CFarray[ncfi].N           = 0;

            // attach cfi0 to new node
            FUNC_CHECK_RETURN(leafnode_attachleaf(ctree, cfi0, ncfi));

            // update pointer
            tipCFi[lf0] = ncfi;
            cfi0        = tipCFi[lf0];

            toptip = ncfi;
        }

        if((ctree->CFarray[cfi0].type == CLUSTER_CF_TYPE_NODE) &&
                (ctree->CFarray[cfi1].type == CLUSTER_CF_TYPE_LEAF))
        {
            printf("    NODE - LEAF\n");

            // Create empty leaf node
            long ncfi;
            FUNC_CHECK_RETURN(get_availableCFindex(ctree, &ncfi));
            ctree->CFarray[ncfi].type        = CLUSTER_CF_TYPE_LEAFNODE;
            ctree->CFarray[ncfi].level       = 1;
            ctree->CFarray[ncfi].parentindex = -1;
            ctree->CFarray[ncfi].NBchild     = 0;
            ctree->CFarray[ncfi].NBleaf      = 0;
            ctree->CFarray[ncfi].N           = 0;

            // attach cfi1 to new node
            FUNC_CHECK_RETURN(leafnode_attachleaf(ctree, cfi1, ncfi));

            // update pointer
            tipCFi[lf1] = ncfi;
            cfi1        = tipCFi[lf1];

            toptip = ncfi;
        }

        int nodemerge = 0;
        if(opOK == 0)
        {
            if((ctree->CFarray[cfi0].type == CLUSTER_CF_TYPE_LEAFNODE) &&
                    (ctree->CFarray[cfi1].type == CLUSTER_CF_TYPE_LEAFNODE))
            {
                printf("    LEAFNODE - LEAFNODE\n");
                nodemerge = 1;
            }
            if((ctree->CFarray[cfi0].type == CLUSTER_CF_TYPE_LEAFNODE) &&
                    (ctree->CFarray[cfi1].type == CLUSTER_CF_TYPE_NODE))
            {
                printf("    LEAFNODE - NODE\n");
                nodemerge = 1;
            }

            if((ctree->CFarray[cfi0].type == CLUSTER_CF_TYPE_NODE) &&
                    (ctree->CFarray[cfi1].type == CLUSTER_CF_TYPE_LEAFNODE))
            {
                printf("    NODE - LEAFNODE\n");
                nodemerge = 1;
            }
            if((ctree->CFarray[cfi0].type == CLUSTER_CF_TYPE_NODE) &&
                    (ctree->CFarray[cfi1].type == CLUSTER_CF_TYPE_NODE))
            {
                printf("    NODE - NODE\n");
                nodemerge = 1;
            }
        }

        if(nodemerge == 1)
        {
            opOK = 1;
            // Create empty node
            long ncfi;
            FUNC_CHECK_RETURN(get_availableCFindex(ctree, &ncfi));
            ctree->CFarray[ncfi].type        = CLUSTER_CF_TYPE_NODE;
            ctree->CFarray[ncfi].level       = 1;
            ctree->CFarray[ncfi].parentindex = -1;
            ctree->CFarray[ncfi].NBchild     = 0;
            ctree->CFarray[ncfi].NBleaf      = 0;
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

        if(lf0update == 1)
        {
            // update distances involving cfi0
            for(long lf = 0; lf < ctree->nbleaf; lf++)
            {
                long cfi = tipCFi[lf];
                if((cfi != -1) &&
                        (ctree->CFarray[cfi].type != CLUSTER_CF_TYPE_UNUSED) &&
                        (cfi != cfi0))
                {
                    double distval = 0.0;
                    FUNC_CHECK_RETURN(compute_imdistance_double(
                                          ctree,
                                          ctree->CFarray[cfi0].datasumvec,
                                          ctree->CFarray[cfi0].N,
                                          ctree->CFarray[cfi].datasumvec,
                                          ctree->CFarray[cfi].N,
                                          &distval));
                    tipdist[lf * ctree->nbleaf + lf0] = distval;
                    tipdist[lf0 * ctree->nbleaf + lf] = distval;
                }
            }
        }

        if(lf1update == 1)
        {
            // update distances involving cfi1
            for(long lf = 0; lf < ctree->nbleaf; lf++)
            {
                long cfi = tipCFi[lf];
                if((cfi != -1) &&
                        (ctree->CFarray[cfi].type != CLUSTER_CF_TYPE_UNUSED) &&
                        (cfi != cfi1))
                {
                    double distval = 0.0;
                    FUNC_CHECK_RETURN(compute_imdistance_double(
                                          ctree,
                                          ctree->CFarray[cfi1].datasumvec,
                                          ctree->CFarray[cfi1].N,
                                          ctree->CFarray[cfi].datasumvec,
                                          ctree->CFarray[cfi].N,
                                          &distval));
                    tipdist[lf * ctree->nbleaf + lf1] = distval;
                    tipdist[lf1 * ctree->nbleaf + lf] = distval;
                }
            }
        }

        if(tipCFi[lf0] == -1)
        {
            for(long lf = 0; lf < ctree->nbleaf; lf++)
            {
                tipdist[lf * ctree->nbleaf + lf0] = 0.0;
                tipdist[lf0 * ctree->nbleaf + lf] = 0.0;
            }
        }
        if(tipCFi[lf1] == -1)
        {
            for(long lf = 0; lf < ctree->nbleaf; lf++)
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
    printf("NUMBER OF MERGE OPERATION = %d\n", nbmergeop);

    printf("TOP CF INDEX = %ld\n", toptip);

    {
        int leveloffset = ctree->CFarray[toptip].level;
        printf("level offset = %d\n", leveloffset);

        for(long cfi = 0; cfi < ctree->NBCF; cfi++)
        {
            if(ctree->CFarray[cfi].type != CLUSTER_CF_TYPE_UNUSED)
            {
                ctree->CFarray[cfi].level -= leveloffset;
            }
        }
    }
    ctree->rootindex = toptip;

    if(ctree->CFarray[toptip].type == CLUSTER_CF_TYPE_LEAFNODE)
    {
        droptree(ctree);
    }

    if(ctree->CFarray[toptip].type == CLUSTER_CF_TYPE_LEAF)
    {
        droptree(ctree);
        droptree(ctree);
    }

    {
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

    FUNC_CHECK_RETURN(printCFtree(ctree));

    free(tipdist);
    free(tipCFi);

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}
