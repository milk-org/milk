#ifndef COREMOD_MODULE_ARITH_IMSET_3DAXES_H
#define COREMOD_MODULE_ARITH_IMSET_3DAXES_H

errno_t image_set_3Daxes(
    IMGID    inimg,
    uint32_t size0,
    uint32_t size1,
    uint32_t size2
);

errno_t CLIADDCMD_COREMOD_arith__imset_3Daxes();

#endif
