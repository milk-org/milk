#ifndef IMG_REDUCE_INTERNAL_H
#define IMG_REDUCE_INTERNAL_H

#include <err.h>
#include <fcntl.h>
#include <malloc.h>
#include <math.h>
#include <ncurses.h>
#include <sched.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include <fitsio.h>

#ifndef MILK_NO_CLI
#    include <stdio.h>
#    include <stdlib.h>
#    include <string.h>
#    include <math.h>
#    include <stdint.h>
#    include <stdbool.h>
#    include <unistd.h>

#    ifdef MILK_NO_CLI
#        include "CLIcore_standalone.h"
#    else
#        include "libmilkdata/milkdata.h"
#        include "milkDebugTools.h"
#        include "fps.h"
#        include "ImageStreamIO/ImageStreamIO.h"
#    endif
#else
#    include "CLIcore_standalone.h"
#endif

#include "COREMOD_arith/COREMOD_arith.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "fft/fft.h"
#include "image_filter/image_filter.h"

#include "img_reduce/img_reduce.h"

#ifdef _OPENMP
#    include <omp.h>
#    define OMP_NELEMENT_LIMIT 1000000
#endif

extern int    badpixclean_init;
extern long   badpixclean_NBop;
extern long  *badpixclean_array_indexin;
extern long  *badpixclean_array_indexout;
extern float *badpixclean_array_coeff;

extern long  badpixclean_NBbadpix;
extern long *badpixclean_indexlist;

#endif // IMG_REDUCE_INTERNAL_H
