#ifndef CLUSTERING_DEFS_H
#define CLUSTERING_DEFS_H

#define CLUSTER_CF_TYPE_UNUSED 0
#define CLUSTER_CF_TYPE_ROOT   1
#define CLUSTER_CF_TYPE_NODE   2

// a LEAF node has leaves
#define CLUSTER_CF_TYPE_LEAF     3
//#define CLUSTER_CF_TYPE_LEAFNODE 4

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

    // index of parent. -1 if no parent
    long parentindex;

    // This is the reference point defining the CF position.
    // If the CF is not a leaf, then this is the same as datasumvec.
    // If the CF is a lead, then it may be different from datasumvec.
    //
    // The criteria for belonging to a leaf is being within distance T of this point.
    // This is the coordinate of the first point assigned to the leaf.
    // Note: This is different from BIRCH which has this point be the average of the points in the leaf
    // The problem with the average is that it can drift away as points are added, so clusters 
    // could become stretched as points are added. Here we ensure that all point in a leaf cluster
    // are within T of this unmovable point.
    double     *dataposvec;

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
    long nbleaf;
    long nbleafsingle;

} CLUSTERTREE;

#endif
