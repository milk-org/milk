#ifndef CLUSTERING_DEFS_H
#define CLUSTERING_DEFS_H

#define CLUSTER_CF_TYPE_UNUSED 0
#define CLUSTER_CF_TYPE_ROOT   1
#define CLUSTER_CF_TYPE_NODE   2

// a LEAF node has leaves
#define CLUSTER_CF_TYPE_LEAF     3
#define CLUSTER_CF_TYPE_LEAFNODE 4

// CF needs to be recomputed
#define CLUSTER_CF_STATUS_UPDATE  0x0001
#define CLUSTER_CF_STATUS_COMPUTE 0x0002
#define CLUSTER_CF_STATUS_CREATE  0x0004

// cluster feature
typedef struct
{
    // see CLUSTER_CF_TYPE defines
    int type;
    int level; // 0 for root

    int NBchild;
    // child index, -1 if no child
    long *childindex;

    int   NBleaf;
    long *leafindex;

    // index of parent. -1 if no parent
    long parentindex;

    long        N;          // number of points aggregated in node
    double     *datasumvec; // sum
    long double datassq;    // sum squared
    long double sum2;       // square norm of sumvec
    double      radius2;    // square cluster radius

    uint32_t status; // check status flag

} CLUSTERING_CF;

typedef struct
{
    long           npix;
    int            B;       // branching parameter
    int            L;       // max number of leafs in leaf node
    double         T;       // threshold
    long           NBCF;    // number of cluster features in memory
    CLUSTERING_CF *CFarray; // pointer to cluster features
    long           rootindex;

    // correction for uncorrelated noise
    double noise2offset;

    // characteristic distance
    // updated as distances are computed
    // used to define meaningful threshold value
    double cdist;

    double minnoise2;

    long long cdistcnt;    // number of distance computation
    long long cdistnegcnt; // number of neg distance

    long nbnode;
    long nbleafnode;
    long nbleaf;
    long nbleafsingle;

} CLUSTERTREE;

#endif
