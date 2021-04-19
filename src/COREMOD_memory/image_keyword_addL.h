#ifndef COREMOD_MEMORY_IMAGE_KEYWORD_ADDL_H
#define COREMOD_MEMORY_IMAGE_KEYWORD_ADDL_H

errno_t image_keyword_addL(
    IMGID img,
    char *kwname,
    long kwval,
    char *comment
);

errno_t CLIADDCMD_COREMOD_memory__image_keyword_addL();

#endif
