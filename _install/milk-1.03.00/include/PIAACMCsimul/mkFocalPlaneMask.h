#ifndef PIAACMCSIMUL_MKFOCALPLANEMASK_H
#define PIAACMCSIMUL_MKFOCALPLANEMASK_H

errno_t mkFocalPlaneMask(const char *IDzonemap_name,
                         const char *ID_name,
                         int         mode,
                         int         saveMask,
                         imageID    *outID);

#endif
