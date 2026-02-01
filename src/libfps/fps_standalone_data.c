#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * fitsio.h often expects LONGLONG to be defined.
 * It is typically defined by system headers if _GNU_SOURCE is present,
 * but we provide a fallback here just in case.
 */
#ifndef LONGLONG
#define LONGLONG long long
#endif

#include "CLIcore.h"

/**
 * @brief Global data structure for standalone FPS executables.
 *
 * Defined here to satisfy runtime symbol lookups from shared libraries
 * that reference 'data' but are linked without libCLIcore.
 */
DATA __attribute__((used)) data;

/*
 * Minimal implementations of CLI registration functions to satisfy
 * runtime symbol lookups in modules that call them during initialization.
 */

errno_t RegisterModule(
    const char *restrict FileName,
    const char *restrict PackageName,
    const char *restrict InfoString,
    int versionmajor,
    int versionminor,
    int versionpatch
) {
    return RETURN_SUCCESS;
}

uint32_t RegisterCLIcommand(
    const char *restrict CLIkey,
    const char *restrict CLImodulesrc,
    errno_t (*CLIfptr)(),
    const char *restrict CLIinfo,
    const char *restrict CLIsyntax,
    const char *restrict CLIexample,
    const char *restrict CLICcall
) {
    return 0;
}

uint32_t RegisterCLIcmd(
    CLICMDDATA CLIcmddata,
    errno_t (*CLIfptr)()
) {
    return 0;
}

imageID image_ID(const char *name, IMAGE *imagearray, long NB_images)
{
    return -1;
}