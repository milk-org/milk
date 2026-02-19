#include "fps.h"
#include "processinfo.h"

errno_t __attribute__((cold)) cubecollapse_addCLIcmd();

imageID cube_collapse(const char *__restrict ID_in_name,
                      const char *__restrict ID_out_name);

/* ================================================================== */
/* PARAMETER DEFINITION (X-MACRO)                                     */
/* ================================================================== */

#define CUBECOLLAPSE_PARAMS(X) \
    X(FPTYPE_STREAMNAME, char*, ".in_name",  "input cube image",  "im1", "im1", &cubecollapse_inimname, (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT)) \
    X(FPTYPE_STREAMNAME, char*, ".out_name", "output 2D image",   "out1", "out1", &cubecollapse_outimname, (FPFLAG_DEFAULT_INPUT | FPFLAG_CLI_INPUT))

extern char *cubecollapse_inimname;
extern char *cubecollapse_outimname;

#define CUBECOLLAPSE_HELPTEXT \
    "cubecollapse: collapse a cube along z\n" \
    "====================================\n" \
    "Sums all slices of a 3D cube into a single 2D image.\n"

