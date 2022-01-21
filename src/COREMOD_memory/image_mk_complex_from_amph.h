#ifndef COREMOD_MEMORY_IMAGE_MK_COMPLEX_FROM_AMPH_H
#define COREMOD_MEMORY_IMAGE_MK_COMPLEX_FROM_AMPH_H

errno_t mk_complex_from_amph(const char *amp_name,
                             const char *pha_name,
                             const char *out_name,
                             int         sharedmem);

errno_t CLIADDCMD_COREMOD__mk_complex_from_amph();

#endif
