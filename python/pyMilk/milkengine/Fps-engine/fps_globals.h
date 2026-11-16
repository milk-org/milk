#ifndef FPS_GLOBALS_H
#define FPS_GLOBALS_H

#include <time.h>
#include "ImageStreamIO/milkDebugTools.h"
#include "fps.h"

/** @brief Current system timestamp, used for FPS synchronization. */
extern long FPS_TIMESTAMP;

/** @brief String representing the current process type (e.g., "conf-name", "run-name"). */
extern char FPS_PROCESS_TYPE[STRINGMAXLEN_FPSPROCESSTYPE];

// Globals formerly in DATA struct

/** @brief Global command code for the current FPS operation (e.g., FPSCMDCODE_CONFSTART). */
extern uint32_t FPS_CMDCODE;

/** @brief Global name of the current FPS being managed by this process. */
extern char FPS_name[STRINGMAXLEN_FPS_NAME];

/** @brief Name of the program that initialized the current FPS session. */
extern char FPS_callprogname[FPS_CALLPROGNAME_STRMAXLEN];

/** @brief Name of the function that initialized the current FPS session. */
extern char FPS_callfuncname[FPS_CALLFUNCNAME_STRMAXLEN];

/** @brief Optional pointer to the user-provided configuration function. */
extern errno_t (*FPS_CONFfunc)();

/** @brief Optional pointer to the user-provided run function. */
extern errno_t (*FPS_RUNfunc)();

/** @brief Global array of connected FPS structures (internal use). */
extern FUNCTION_PARAMETER_STRUCT *fpsarray;

#endif
