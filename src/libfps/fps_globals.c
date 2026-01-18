#include "fps_globals.h"

long FPS_TIMESTAMP = 0;
char FPS_PROCESS_TYPE[STRINGMAXLEN_FPSPROCESSTYPE] = "UNDEF";

uint32_t FPS_CMDCODE = 0;
char FPS_name[STRINGMAXLEN_FPS_NAME] = "";
errno_t (*FPS_CONFfunc)() = NULL;
errno_t (*FPS_RUNfunc)() = NULL;

FUNCTION_PARAMETER_STRUCT *fpsarray = NULL;
