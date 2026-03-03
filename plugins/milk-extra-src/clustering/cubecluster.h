#ifndef CLUSTERING_CUBECLUSTER_H
#define CLUSTERING_CUBECLUSTER_H

#include "CLIcore.h"
#include "fps.h"
#include "fps_cli_binding.h"
#include "processinfo.h"

errno_t CLIADDCMD_clustering__imcube_mkcluster();

#define CUBECLUSTER_PARAMS(X) \
    X( \
        ".in_name", \
        &farg_inimname, \
        FPTYPE_STREAMNAME, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "input image cube" \
    ) \
    X( \
        ".outdname", \
        &farg_outdname, \
        FPTYPE_STREAMNAME, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "output directory name" \
    ) \
    X( \
        ".T", \
        &threshold, \
        FPTYPE_FLOAT32, \
        0, \
        FPFLAG_DEFAULT_INPUT, \
        "threshold" \
    )  \
    X( \
        ".B", \
        &branchB, \
        FPTYPE_UINT32, \
        0, \
        FPFLAG_DEFAULT_INPUT, \
        "branch number" \
    )  \
    X( \
        ".leafposmode", \
        &leafposmode, \
        FPTYPE_UINT32, \
        0, \
        FPFLAG_DEFAULT_INPUT, \
        "leaf position mode" \
    )  \
    X( \
        ".NBCFmax", \
        &NBCFmax, \
        FPTYPE_UINT32, \
        0, \
        FPFLAG_DEFAULT_INPUT, \
        "max number of CFs" \
    )  \
    X( \
        ".opt.rebuild", \
        &optrebuild, \
        FPTYPE_ONOFF, \
        0, \
        FPFLAG_DEFAULT_INPUT, \
        "rebuild tree after scan" \
    )  \
    X( \
        ".opt.condense", \
        &optcondense, \
        FPTYPE_ONOFF, \
        0, \
        FPFLAG_DEFAULT_INPUT, \
        "condense tree after scan" \
    )

extern char     *farg_inimname;
extern char     *farg_outdname;
extern float    *threshold;
extern uint32_t *branchB;
extern uint32_t *leafposmode;
extern uint32_t *NBCFmax;
extern int64_t  *optrebuild;
extern int64_t  *optcondense;

#define CUBECLUSTER_HELPTEXT \
    "cubeclust: compute cube cluster\n" \
    "==============================\n"

#endif
