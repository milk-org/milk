#ifndef _TOOLS_H
#define _TOOLS_H


#include "COREMOD_tools/fileutils.h"
#include "COREMOD_tools/imdisplay3d.h"
#include "COREMOD_tools/logfunc.h"
#include "COREMOD_tools/mvprocCPUset.h"
#include "COREMOD_tools/quicksort.h"
#include "COREMOD_tools/statusstat.h"
#include "COREMOD_tools/stringutils.h"
#include "COREMOD_tools/timeutils.h"











errno_t lin_regress(
    double *a,
    double *b,
    double *Xi2,
    double *x,
    double *y,
    double *sig,
    unsigned int nb_points
);



errno_t tp(
    const char *word
);



#endif








































