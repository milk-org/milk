#ifndef _IMGREDUCE_H
#define _IMGREDUCE_H

void __attribute__((constructor)) libinit_img_reduce();

imageID IMG_REDUCE_cubesimplestat(const char *IDin_name);

imageID IMG_REDUCE_cleanbadpix_fast(const char *IDname,
                                    const char *IDbadpix_name,
                                    const char *IDoutname,
                                    int         streamMode);

imageID IMG_REDUCE_centernormim(const char *IDin_name,
                                const char *IDref_name,
                                const char *IDout_name,
                                long        xcent0,
                                long        ycent0,
                                long        xcentsize,
                                long        ycentsize,
                                int         mode,
                                int         semtrig);

errno_t IMG_REDUCE_cubeprocess(const char *IDin_name);

#endif
