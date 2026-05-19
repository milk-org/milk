/**
 * @file leaf_addentry.h
 * @brief Leaf addentry module
 */

#ifndef CLUSTERING__LEAF_ADDENTRY_H
#define CLUSTERING__LEAF_ADDENTRY_H

errno_t leaf_addentry(CLUSTERTREE *ctree,
                      double      *datavec,
                      long double  ssqr,
                      long         lCFindex,
                      int         *addOK,
                      double       distance);

#endif
