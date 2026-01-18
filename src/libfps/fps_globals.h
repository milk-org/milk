#ifndef FPS_GLOBALS_H
#define FPS_GLOBALS_H

#include <time.h>
#include "milkDebugTools.h"
#include "fps.h"

extern long FPS_TIMESTAMP;
extern char FPS_PROCESS_TYPE[STRINGMAXLEN_FPSPROCESSTYPE];

// Globals formerly in DATA struct
extern uint32_t FPS_CMDCODE;
extern char FPS_name[STRINGMAXLEN_FPS_NAME];
extern errno_t (*FPS_CONFfunc)();
extern errno_t (*FPS_RUNfunc)();

extern FUNCTION_PARAMETER_STRUCT *fpsarray;

#endif
