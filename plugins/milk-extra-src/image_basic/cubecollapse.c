#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "CLIcore.h"
#include "cubecollapse.h"
#include "fps.h"
#include "processinfo.h"
#include "ImageStreamIO.h"

#include "COREMOD_memory/COREMOD_memory.h"

char *cubecollapse_inimname = NULL;
char *cubecollapse_outimname = NULL;

static void cube_collapse_step(IMAGE *imgin, IMAGE *imgout)
{
    uint32_t xsize = imgin->md[0].size[0];
    uint32_t ysize = imgin->md[0].size[1];
    uint32_t ksize = imgin->md[0].size[2];
    for(uint32_t i = 0; i < xsize * ysize; i++) {
        float v = 0.0;
        for(uint32_t k = 0; k < ksize; k++) v += imgin->array.F[k * xsize * ysize + i];
        imgout->array.F[i] = v;
    }
}

#ifndef FPS_STANDALONE
static CLICMDARGDEF farg[] = {
#define X_CLI_DEF(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { cli_type, key, descr, def_str, cli_flags, (void **) ptr_addr, NULL },
    CUBECOLLAPSE_PARAMS(X_CLI_DEF)
#undef X_CLI_DEF
};
static CLICMDDATA CLIcmddata = { "cubecollapse", "collapse a cube along z", CLICMD_FIELDS_DEFAULTS };
static errno_t help_function() { if (data.fpsptr && data.fpsptr->md) printf("%s\n", data.fpsptr->md->helptext); return RETURN_SUCCESS; }

imageID cube_collapse(const char *ID_in_name, const char *ID_out_name)
{
    IMGID in = mkIMGID_from_name(ID_in_name); resolveIMGID(&in, ERRMODE_ABORT);
    IMGID out = stream_connect_create_2Df32(ID_out_name, in.md->size[0], in.md->size[1]);
    cube_collapse_step(in.im, out.im);
    ImageStreamIO_UpdateIm(out.im); return out.ID;
}

static errno_t compute_function() { cube_collapse(cubecollapse_inimname, cubecollapse_outimname); return RETURN_SUCCESS; }
INSERT_STD_FPSCLIfunctions
errno_t __attribute__((cold)) cubecollapse_addCLIcmd() { INSERT_STD_CLIREGISTERFUNC return RETURN_SUCCESS; }
#endif

#ifdef FPS_STANDALONE
int FPSINIT_cubecollapse(const char *fps_name, const char *keywords, const char *description) {
    FUNCTION_PARAMETER_STRUCT fps; FPS_INIT_STD_PREAMBLE(fps, fps_name, keywords, description, CUBECOLLAPSE_HELPTEXT); FPS_INIT_PROCINFO_DEFAULTS(fps, "im1", 1);
#define X_FPS_INIT(cli_type, fps_type, c_type, key, descr, def_str, def_val, ptr_addr, val_expr, cli_flags) { c_type val = def_val; function_parameter_add_entry(&fps, key, descr, fps_type, FPFLAG_DEFAULT_INPUT, val_expr, NULL); }
    CUBECOLLAPSE_PARAMS(X_FPS_INIT)
#undef X_FPS_INIT
    fps_add_processinfo_entries(&fps); function_parameter_FPCONFexit(&fps); return 0;
}
int FPSCONF_cubecollapse(const char *fps_name, int loop) { FPS_CONF_STD_BODY(fps_name, loop, { cubecollapse_inimname = functionparameter_GetParamPtr_STRING(&fps, ".in_name"); cubecollapse_outimname = functionparameter_GetParamPtr_STRING(&fps, ".out_name"); }, { }); return 0; }
FPS_MAKE_STANDALONE_CONFSTOP(cubecollapse) FPS_MAKE_STANDALONE_RUNSTOP(cubecollapse)
int FPSRUN_cubecollapse(const char *fps_name) {
    FUNCTION_PARAMETER_STRUCT fps; FPS_RUN_STD_PREAMBLE(fps_name, fps, { cubecollapse_inimname = functionparameter_GetParamPtr_STRING(&fps, ".in_name"); cubecollapse_outimname = functionparameter_GetParamPtr_STRING(&fps, ".out_name"); });
    IMAGE iin; if (ImageStreamIO_read_sharedmem_image_toIMAGE(cubecollapse_inimname, &iin) != 0) return 1;
    IMAGE iout; uint32_t size[2] = { iin.md[0].size[0], iin.md[0].size[1] };
    if (ImageStreamIO_read_sharedmem_image_toIMAGE(cubecollapse_outimname, &iout) != 0) { ImageStreamIO_createIm(&iout, cubecollapse_outimname, 2, size, _DATATYPE_FLOAT, 1, 10, 0); }
    PROCESSINFO *pinfo = processinfo_setup((char*)fps_name, "Run", "Looping", __FUNCTION__, __FILE__, __LINE__);
    processinfo_waitoninputstream_init(pinfo, &iin, PROCESSINFO_TRIGGERMODE_SEMAPHORE, -1);
    fps_to_processinfo(&fps, pinfo); processinfo_loopstart(pinfo);
    while(processinfo_loopstep(pinfo)) { processinfo_waitoninputstream(pinfo); if (pinfo->triggerstatus == PROCESSINFO_TRIGGERSTATUS_TIMEDOUT) continue;
        processinfo_exec_start(pinfo); cube_collapse_step(&iin, &iout); processinfo_exec_end(pinfo); ImageStreamIO_UpdateIm(&iout);
    }
    processinfo_cleanExit(pinfo); function_parameter_struct_disconnect(&fps); return 0;
}
FPS_MAIN_STANDALONE("cubecollapse", cubecollapse, CUBECOLLAPSE_HELPTEXT, CUBECOLLAPSE_PARAMS)
#endif
