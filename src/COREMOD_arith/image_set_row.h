#ifndef COREMOD_MODULE_ARITH_IMSET_ROW_H
#define COREMOD_MODULE_ARITH_IMSET_ROW_H

errno_t image_set_row(
    IMGID    inimg,
    uint32_t rowindex,
    double   value
);

errno_t CLIADDCMD_COREMOD_arith__imset_row();

#endif
