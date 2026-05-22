/**
 * @file compute_imdistance_double.h
 * @brief Compute imdistance double module
 */

#ifndef CLUSTERING__COMPUTE_IMDISTANCE_DOUBLE_H
#define CLUSTERING__COMPUTE_IMDISTANCE_DOUBLE_H

errno_t compute_imdistance_double(CLUSTERTREE *ctree,
                                  double      *vec1,
                                  long         N1,
                                  double      *vec2,
                                  long         N2,
                                  double      *distval);

errno_t compute_CF2CF_posdistance_double(CLUSTERTREE *ctree, long CFI0, long CFI1, double *distval);

#endif
