/**
 * @file write_clustleafsummary.h
 * @brief Write clustleafsummary module
 */

#ifndef CLUSTERING_WRITECLUSTLEAFSUMMARY
#define CLUSTERING_WRITECLUSTLEAFSUMMARY

errno_t write_clustleafsummary(
    CLUSTERTREE *ctree,
    IMGID img,
    long *pixmap,
    double *pixgain,
    long *frameleafCFindex,
    long NBframe,
    const char *__restrict outdname
);

#endif
