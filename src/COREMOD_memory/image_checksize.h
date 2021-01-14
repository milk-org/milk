/**
 * @file    image_checksize.h
 */




int check_2Dsize(
    const char *ID_name,
    uint32_t    xsize,
    uint32_t    ysize
);

int check_3Dsize(
    const char *ID_name,
    uint32_t    xsize,
    uint32_t    ysize,
    uint32_t    zsize
);

int COREMOD_MEMORY_check_2Dsize(
    const char *IDname,
    uint32_t    xsize,
    uint32_t    ysize
);

int COREMOD_MEMORY_check_3Dsize(
    const char *IDname,
    uint32_t    xsize,
    uint32_t    ysize,
    uint32_t    zsize
);
