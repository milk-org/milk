#ifndef COREMOD_MODULE_ARITH_IMSET_AXES_H
#define COREMOD_MODULE_ARITH_IMSET_AXES_H

errno_t image_set_axes(
    IMGID    inimg,
    uint32_t size0,
    uint32_t size1,
    uint32_t size2
);

errno_t CLIADDCMD_COREMOD_arith__imset_axes();

#endif
