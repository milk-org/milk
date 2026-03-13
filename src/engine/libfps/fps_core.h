/**
 * @file fps_core.h
 * @brief Core FPS API: connect, disconnect, param access.
 *
 * Include this header when you only need the essential
 * FPS operations (connect/disconnect, parameter read/write,
 * entry management) without pulling in the full fps.h
 * umbrella (scan, tmux, lifecycle, logging, V2 framework).
 *
 * This is a subset of fps.h — anything that includes
 * fps.h already has everything in this file.
 */

#ifndef FPS_CORE_H
#define FPS_CORE_H

#include "fps_types.h"

/* Entry management */
#include "fps_add_entry.h"
#include "fps_SetParamCLIindex.h"

/* Connection management */
#include "fps_connect.h"
#include "fps_connectExternalFPS.h"
#include "fps_disconnect.h"

/* Parameter access */
#include "fps_paramvalue.h"
#include "fps_GetParamIndex.h"
#include "fps_GetTypeString.h"
#include "fps_checkparameter.h"

/* Info / diagnostics */
#include "fps_print_info.h"
#include "fps_PrintParameterInfo.h"
#include "fps_printparameter_valuestring.h"

#endif /* FPS_CORE_H */
