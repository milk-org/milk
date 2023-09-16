#ifndef COREMOD_MODULE_ARITH_IMSET_1DPIXRANGE_H
#define COREMOD_MODULE_ARITH_IMSET_1DPIXRANGE_H

errno_t image_set_1Dpixrange(
    IMGID    inimg,
    double   value,
    uint32_t minindex,
    uint32_t maxindex
);

errno_t CLIADDCMD_COREMOD_arith__imset_1Dpixrange();

#endif
