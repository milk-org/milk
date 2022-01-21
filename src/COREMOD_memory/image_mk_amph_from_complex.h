#ifndef COREMOD_MEMORY_IMAGE_MK_AMPH_FROM_COMPLEX_H
#define COREMOD_MEMORY_IMAGE_MK_AMPH_FROM_COMPLEX_H

errno_t mk_amph_from_complex(const char *in_name,
                             const char *am_name,
                             const char *ph_name,
                             int         sharedmem);

errno_t CLIADDCMD_COREMOD__mk_amph_from_complex();

#endif
