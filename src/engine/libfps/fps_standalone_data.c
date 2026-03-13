/**
 * @file fps_standalone_data.c
 * @brief Fps standalone data module
 */

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

/*
 * Global data structure for standalone FPS executables.
 * Some executables (milk-fps-info, etc.) do not link
 * against libCLIcore.so and need this definition.
 * V2 standalone executables that link libCLIcore.so
 * will have this symbol overridden by the library's
 * copy via ELF symbol interposition.
 */
DATA __attribute__((used)) data;

/* CLIPID is normally provided by CLIcore.c;
 * standalone builds need their own storage.
 *
 * Skip when FPS_STANDALONE_SKIP_STUBS is set
 * (static LTO builds get CLIPID from the
 * processinfo static archive).
 */
#ifndef FPS_STANDALONE_SKIP_STUBS
pid_t CLIPID;
#endif


/*
 * Minimal implementations of CLI registration functions to satisfy
 * runtime symbol lookups in modules that call them during initialization.
 *
 * When MILK_NO_CLI is defined, CLIcore_standalone.h already
 * provides these as static inline stubs, so we skip them here.
 */

#ifndef FPS_STANDALONE_SKIP_STUBS
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

imageID image_ID(
    const char *name,
    IMAGE      *imagearray,
    long        NB_images
)
{
    for (long i = 0; i < NB_images; i++)
    {
        if (imagearray[i].used == 1 &&
            strncmp(imagearray[i].name,
                    name,
                    STRINGMAXLEN_IMAGE_NAME)
                == 0)
        {
            return i;
        }
    }
    return -1;
}
#endif /* !FPS_STANDALONE_SKIP_STUBS */