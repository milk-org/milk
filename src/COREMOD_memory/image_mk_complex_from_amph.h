#ifndef COREMOD_MEMORY_IMAGE_MK_COMPLEX_FROM_AMPH_H
#define COREMOD_MEMORY_IMAGE_MK_COMPLEX_FROM_AMPH_H

errno_t mk_complexIMG_from_amphIMG(
    IMGID imginamp,
    IMGID imginpha,
    IMGID *imgoutC
);

errno_t mk_complex_from_amph(
    const char *am_name,
    const char *ph_name,
    const char *out_name,
    int         sharedmem
);

errno_t CLIADDCMD_COREMOD__mk_complex_from_amph();

#endif
