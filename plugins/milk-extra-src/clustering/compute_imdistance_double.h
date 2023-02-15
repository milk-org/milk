#ifndef CLUSTERING__COMPUTE_IMDISTANCE_DOUBLE_H
#define CLUSTERING__COMPUTE_IMDISTANCE_DOUBLE_H

errno_t compute_imdistance_double(CLUSTERTREE *ctree,
                                  double      *vec1,
                                  long         N1,
                                  double      *vec2,
                                  long         N2,
                                  double      *distval);

#endif
