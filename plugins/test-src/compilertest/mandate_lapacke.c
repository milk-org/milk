// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "milk_runnable.h"
#include "COREMOD_memory/COREMOD_memory.h"

// MILK_CMAKE_MANDATE_LAPACKE
#include "milk_blas_lapacke.h"

static char some_string[FUNCTION_PARAMETER_STRMAXLEN] = "string";

#define MANDLAPACKE_PARAMS(X)                                                         \
    X(".some_string", some_string, FPTYPE_STRING_NOT_STREAM, 1, FPFLAG_DEFAULT_INPUT, \
      "Unused string")

MILK_PARAMS_DEFINE(mandlapacke, MANDLAPACKE_PARAMS);

static errno_t mandlapacke_compute(const MILK_RUNNABLE *self)
{
    DEBUG_TRACE_FSTART();

    MILK_PROCINFO_INIT(self)

    MILK_PROCINFO_LOOPSTART(self)
    {
        printf("Lapacke function body...\n");
    }
    MILK_PROCINFO_END(self)

    DEBUG_TRACE_FEXIT();
    return RETURN_SUCCESS;
}

// Not declared in compilertest.h: the module registry declares it weak (file may be dropped).
const MILK_RUNNABLE runnable_ct_mandatelapacke = {
    .kind    = MILK_RUNNABLE_FPS,
    .app     = { .fps_name         = "mandlapacke",
                 .cmdkey           = "mandlapacke",
                 .description      = "Dumb test for MILK_CMAKE_MANDATE_LAPACKE test",
                 .description_long = "Dumb test for MILK_CMAKE_MANDATE_LAPACKE test" },
    .srcfile = __FILE__,
    MILK_PARAMS_REF(mandlapacke),
    .compute = mandlapacke_compute,
};
EXPORT_FPSEXEC(milk, "compilertest-mandatelapacke", runnable_ct_mandatelapacke);
