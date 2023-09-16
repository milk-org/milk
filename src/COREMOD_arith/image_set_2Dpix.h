#ifndef COREMOD_MODULE_ARITH_IMSET_2DPIX_H
#define COREMOD_MODULE_ARITH_IMSET_2DPIX_H

errno_t image_set_2Dpix(
    IMGID    inimg,
    uint32_t colindex,
    uint32_t rowindex,
    double   value
);

errno_t CLIADDCMD_COREMOD_arith__imset_2Dpix();

#endif
