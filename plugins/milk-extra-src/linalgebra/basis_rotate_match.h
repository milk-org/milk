#ifndef LINALGEBRA_BASIS_ROTATE_MATCH_H
#define LINALGEBRA_BASIS_ROTATE_MATCH_H


errno_t compute_basis_rotate_match(
    IMGID imginAB,
    IMGID *imgArot,
    int optmode
);

errno_t CLIADDCMD_linalgebra__basis_rotate_match();

#endif
