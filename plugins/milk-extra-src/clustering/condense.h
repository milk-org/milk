/**
 * @file condense.h
 * @brief Condense module
 */

#ifndef CLUSTERING__CONDENSE_H
#define CLUSTERING__CONDENSE_H


errno_t ctree_condense_node(
    CLUSTERTREE *ctree,
    long CFindex,
    int *nbop
);

errno_t ctree_condense(CLUSTERTREE *ctree, int *nbop);

#endif
