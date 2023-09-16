#ifndef COREMOD_MODULE_ARITH_IMSET_COL_H
#define COREMOD_MODULE_ARITH_IMSET_COL_H

errno_t image_set_col(
    IMGID    inimg,
    double   value,
    uint32_t colindex
);

errno_t CLIADDCMD_COREMOD_arith__imset_col();

#endif
