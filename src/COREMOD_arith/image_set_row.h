#ifndef COREMOD_MODULE_ARITH_IMSET_ROW_H
#define COREMOD_MODULE_ARITH_IMSET_ROW_H

errno_t image_set_row(
    IMGID    inimg,
    double   value,
    uint32_t rowindex
);

errno_t CLIADDCMD_COREMOD_arith__imset_row();

#endif
