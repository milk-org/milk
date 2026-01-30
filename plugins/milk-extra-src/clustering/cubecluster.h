#ifndef CLUSTERING_CUBECLUSTER_H
#define CLUSTERING_CUBECLUSTER_H

#include "CLIcore.h"
#include "fps.h"
#include "processinfo.h"

errno_t CLIADDCMD_clustering__imcube_mkcluster();

#define CUBECLUSTER_PARAMS(X) \
    X(CLIARG_IMG,     FPTYPE_STREAMNAME, char*, ".in_name",      "input image cube",      "imc1", "imc1", &farg_inimname, (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_STR,     FPTYPE_STREAMNAME, char*, ".outdname",     "output directory name", "outd", "outd", &farg_outdname, (void*)val,  CLIARG_VISIBLE_DEFAULT) \
    X(CLIARG_FLOAT32, FPTYPE_FLOAT32,    float, ".T",             "threshold",             "1.0",  1.0,   &threshold,       (void*)&val, CLIARG_HIDDEN_DEFAULT)  \
    X(CLIARG_UINT32,  FPTYPE_UINT32,     uint32_t, ".B",          "branch number",         "10",   10,    &branchB,         (void*)&val, CLIARG_HIDDEN_DEFAULT)  \
    X(CLIARG_UINT32,  FPTYPE_UINT32,     uint32_t, ".leafposmode","leaf position mode",    "1",    1,     &leafposmode,     (void*)&val, CLIARG_HIDDEN_DEFAULT)  \
    X(CLIARG_UINT32,  FPTYPE_UINT32,     uint32_t, ".NBCFmax",    "max number of CFs",     "2048", 2048,  &NBCFmax,         (void*)&val, CLIARG_HIDDEN_DEFAULT)  \
    X(CLIARG_ONOFF,   FPTYPE_ONOFF,      int64_t, ".opt.rebuild", "rebuild tree after scan","1",    1,     &optrebuild,      (void*)&val, CLIARG_HIDDEN_DEFAULT)  \
    X(CLIARG_ONOFF,   FPTYPE_ONOFF,      int64_t, ".opt.condense","condense tree after scan","1",   1,     &optcondense,     (void*)&val, CLIARG_HIDDEN_DEFAULT)

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
