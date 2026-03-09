/**
 * @file image_keyword_addD.h
 * @brief Image keyword addd module
 */

#ifndef COREMOD_MEMORY_IMAGE_KEYWORD_ADDD_H
#define COREMOD_MEMORY_IMAGE_KEYWORD_ADDD_H

errno_t
image_keyword_addD(IMGID img, char *kwname, double kwval, char *comment);

errno_t CLIADDCMD_COREMOD_memory__image_keyword_addD();

#endif
