#ifndef CLUSTERING__ADDVECTOR_TO_CF_H
#define CLUSTERING__ADDVECTOR_TO_CF_H

errno_t addvector_to_CF(CLUSTERTREE *ctree,
                        double      *datavec,
                        long double  ssqr,
                        long         N,
                        long         CFindex,
                        int         *addOK);

errno_t subvector_to_CF(CLUSTERTREE *ctree,
                        double      *datavec,
                        long double  ssqr,
                        long         N,
                        long         CFindex);

#endif
