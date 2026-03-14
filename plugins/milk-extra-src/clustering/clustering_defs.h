/**
 * @file clustering_defs.h
 * @brief are CF positions fixed or dynamic
 */

#ifndef CLUSTERING_DEFS_H
#define CLUSTERING_DEFS_H


// are CF positions fixed or dynamic
// if fixed, then adding points will not change the CF position
// In Fixed mode, the CF position is ONLY allocated at creation when the first point is added
// Fixed mode is faster
#define CLUSTER_CFPOS_FIXED 0
#define CLUSTER_CFPOS_DYNAMIC 1

#define CLUSTER_CF_TYPE_UNUSED 0
#define CLUSTER_CF_TYPE_ROOT   1
#define CLUSTER_CF_TYPE_NODE   2

// a LEAF node has leaves
#define CLUSTER_CF_TYPE_LEAF     3
//#define CLUSTER_CF_TYPE_LEAFNODE 4

#define CLUSTER_CF_MAXLEVEL   4096


// CF needs to be recomputed
#define CLUSTER_CF_STATUS_UPDATE  0x0001
#define CLUSTER_CF_STATUS_COMPUTE 0x0002
#define CLUSTER_CF_STATUS_CREATE  0x0004
#define CLUSTER_CF_STATUS_MEMALLOC  0x0008 // has memory been allocated ?

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

    // If the position vector is inherited from a node or cluster, this is the index of the
    // node/cluster from which it is derived.
    // If not, this is set to -1.
    long        posvecsourceID;

    long        N;          // number of points aggregated in node
    double     *datasumvec; // sum vector
    long double datassq;    // sum squared
    long double sum2;       // square norm of sumvec
    double      radius2;    // square cluster radius

    // stats
    // Probability that this node is on the path to solution,
    // computed from recent searches.
    // Not normalized.
    // This is simlar to N, but with more weight on recent points.
    double pathcnt;
    // Average number of distance computations needed to find solution from this point
    double pathdistcompcnt;

    // max distance from pos to point(s) within and downstream of this node
    double radius;

    uint32_t status; // check status flag

} CLUSTERING_CF;

typedef struct
{
    long           npix;

    // for 2D image representation of CFs
    uint32_t       xsize;
    uint32_t       ysize;

    int            B;           // branching parameter
    double         T;           // threshold
    int            leafposmode; // leaf position mode. 0=static, 1=dynamic
    long           NBCF;        // number of cluster features in memory
    CLUSTERING_CF *CFarray;     // pointer to cluster features
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

    // computation stats
    long stat_compdistcnt; // number of distances computed

    // current path and stats along
    long path_node[CLUSTER_CF_MAXLEVEL];
    long path_distcompcnt[CLUSTER_CF_MAXLEVEL];

    // Increments at each new CF allocation (node or leaf)
    // Used as a unique identifier to a CF, ensuring no re-use
    long CFIDcnt;

    // Stores distances between CFs
    // For fixed positions only
    // If unknown, val set to negative
    //
    double *CFCFdist;

} CLUSTERTREE;

#endif
