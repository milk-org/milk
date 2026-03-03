#include "fps.h"
#include "fps_cli_binding.h"
#include "processinfo.h"

errno_t __attribute__((cold)) cubecollapse_addCLIcmd();

imageID cube_collapse(const char *__restrict ID_in_name,
                      const char *__restrict ID_out_name);

/* ================================================================== */
/* PARAMETER DEFINITION (X-MACRO)                                     */
/* ================================================================== */

#define CUBECOLLAPSE_PARAMS(X) \
    X( \
        ".in_name", \
        &cubecollapse_inimname, \
        FPTYPE_STREAMNAME, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "input cube image" \
    ) \
    X( \
        ".out_name", \
        &cubecollapse_outimname, \
        FPTYPE_STREAMNAME, \
        1, \
        FPFLAG_DEFAULT_INPUT, \
        "output 2D image" \
    )

extern char *cubecollapse_inimname;
extern char *cubecollapse_outimname;

#define CUBECOLLAPSE_HELPTEXT \
    "cubecollapse: collapse a cube along z\n" \
    "====================================\n" \
    "Sums all slices of a 3D cube into a single 2D image.\n"

