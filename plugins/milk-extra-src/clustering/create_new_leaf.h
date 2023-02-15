#ifndef CLUSTERING__CREATE_NEW_LEAF_H
#define CLUSTERING__CREATE_NEW_LEAF_H

errno_t create_new_leaf(CLUSTERTREE *ctree,
                        double      *datarray,
                        long double  ssqr,
                        long        *CFindex);

#endif
