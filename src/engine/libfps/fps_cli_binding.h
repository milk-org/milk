/**
 * @file    fps_cli_binding.h
 * @brief   FPS-CLI binding type and X-macro expansion helpers
 *
 * Provides the FPS_CLI_BINDING structure and associated macros
 * for the unified FPS-CLI parameter architecture.
 */

#ifndef FPS_CLI_BINDING_H
#define FPS_CLI_BINDING_H

#include <stdint.h>

#include "fps.h"
#include "milkdata_clicmd.h"


/**
 * @brief Application identity for FPS-based commands.
 *
 * Every FPS module defines one of these describing the command
 * name, FPS base name, and human-readable description.
 */
typedef struct FPS_APP_INFO_
{
    const char *fps_name;
    const char *cmdkey;
    const char *description;
} FPS_APP_INFO;


/**
 * @brief Binding between an FPS keyword and a local C variable.
 *
 * Replaces CLICMDARGDEF by directly linking FPS keywords to
 * the memory locations of module-local variables.
 *
 * @param fpskeyword  FPS keyword (e.g. ".gain")
 * @param ptr         Pointer to local variable
 * @param type        FPS type (FPTYPE_FLOAT64, etc.)
 * @param is_primary  1 if primary CLI argument, 0 otherwise
 * @param fpflag      Standard FPS flags
 * @param descr       Human-readable description
 */
typedef struct FPS_CLI_BINDING_
{
    const char *fpskeyword;
    void       *ptr;
    uint64_t    type;
    int         is_primary;
    uint64_t    fpflag;
    const char *descr;
} FPS_CLI_BINDING;


/* ---- X-macro expansion helpers ---- */

/**
 * @brief Expand a PARAMS(X) macro into an FPS_CLI_BINDING array.
 *
 * Usage:
 *   static FPS_CLI_BINDING bindings[] = {
 *       MY_PARAMS(FPS_X_BINDING)
 *   };
 */
#define FPS_X_BINDING(kw, ptr, type, is_primary, flag, desc) \
    { kw, ptr, type, is_primary, flag, desc },

/**
 * @brief Expand a PARAMS(X) macro into a CLICMDARGDEF array.
 *
 * Usage:
 *   static CLICMDARGDEF farg[] = {
 *       MY_PARAMS(FPS_X_FARG)
 *   };
 */
#define FPS_X_FARG(kw, ptr, fctype, is_primary, flag, desc) \
    { fctype, kw, desc, "", \
      flag | (is_primary ? FPFLAG_PRIMARY_CLI_INPUT : 0), \
      NULL, NULL },


#endif /* FPS_CLI_BINDING_H */
