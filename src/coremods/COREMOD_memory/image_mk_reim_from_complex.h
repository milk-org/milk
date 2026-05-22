/**
 * @file image_mk_reim_from_complex.h
 * @brief Image mk reim from complex module
 */

#ifndef COREMOD_MEMORY_IMAGE_MK_REIM_FROM_COMPLEX_H
#define COREMOD_MEMORY_IMAGE_MK_REIM_FROM_COMPLEX_H

errno_t mk_reim_from_complex(const char *in_name,
                             const char *re_name,
                             const char *im_name,
                             int         sharedmem);

errno_t mk_reim_from_complex_IMGID(IMGID *imgin, IMGID *imgre, IMGID *imgim);

errno_t CLIADDCMD_COREMOD__mk_reim_from_complex();

#endif
