/**
 * @file fps_globals.c
 * @brief Fps globals module
 */

#include "fps_globals.h"

long FPS_TIMESTAMP = 0;
char FPS_PROCESS_TYPE[STRINGMAXLEN_FPSPROCESSTYPE] = "UNDEF";

uint32_t FPS_CMDCODE = 0;
char FPS_name[STRINGMAXLEN_FPS_NAME] = "";
char FPS_callprogname[FPS_CALLPROGNAME_STRMAXLEN] = "milk-cli";
char FPS_callfuncname[FPS_CALLFUNCNAME_STRMAXLEN] = "unknown";
errno_t (*FPS_CONFfunc)() = NULL;
errno_t (*FPS_RUNfunc)() = NULL;

FUNCTION_PARAMETER_STRUCT *fpsarray = NULL;
