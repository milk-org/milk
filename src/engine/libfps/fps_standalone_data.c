/**
 * @file fps_standalone_data.c
 * @brief Fps standalone data module
 */

#define _GNU_SOURCE

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
 *
 * Also skip when MILK_NO_CLI is defined,
 * because CLIcore_standalone.h already
 * provides a static CLIPID.
 */
#if !defined(FPS_STANDALONE_SKIP_STUBS) \
    && !defined(MILK_NO_CLI)
pid_t CLIPID;
#endif


/*
 * Minimal implementations of CLI registration functions to satisfy
 * runtime symbol lookups in modules that call them during initialization.
 *
 * When MILK_NO_CLI is defined, CLIcore_standalone.h already
 * provides these as static inline stubs, so we skip them here.
 */

#if !defined(FPS_STANDALONE_SKIP_STUBS) \
    && !defined(MILK_NO_CLI)
/**
 * @brief Stub: no-op module registration for standalone.
 */
errno_t RegisterModule(
    const char *restrict FileName __attribute__((unused)),
    const char *restrict PackageName __attribute__((unused)),
    const char *restrict InfoString __attribute__((unused)),
    int versionmajor __attribute__((unused)),
    int versionminor __attribute__((unused)),
    int versionpatch __attribute__((unused)))
{
    return RETURN_SUCCESS;
}

/**
 * @brief Stub: no-op CLI command registration.
 */
uint32_t RegisterCLIcommand(
    const char *restrict CLIkey __attribute__((unused)),
    const char *restrict CLImodulesrc __attribute__((unused)),
    errno_t (*CLIfptr)() __attribute__((unused)),
    const char *restrict CLIinfo __attribute__((unused)),
    const char *restrict CLIsyntax __attribute__((unused)),
    const char *restrict CLIexample __attribute__((unused)),
    const char *restrict CLICcall __attribute__((unused)))
{
    return 0;
}

/**
 * @brief Stub: no-op CLI command registration (v2).
 */
uint32_t RegisterCLIcmd(
    CLICMDDATA CLIcmddata __attribute__((unused)),
    errno_t (*CLIfptr)() __attribute__((unused)))
{
    return 0;
}

/**
 * @brief Stub: image lookup by name for standalone builds.
 *
 * Linear search through the provided image array.
 *
 * @param name        Stream name to find
 * @param imagearray  Image array to search
 * @param NB_images   Number of entries in imagearray
 * @return imageID index or -1 if not found
 */
imageID image_ID(
    const char *name,
    IMAGE      *imagearray __attribute__((unused)),
    long NB_images __attribute__((unused)))
{
    for(long ii = 0; ii < NB_images; ii++)
    {
        if(imagearray[ii].used == 1 &&
                strncmp(imagearray[ii].name,
                        name,
                        STRINGMAXLEN_IMAGE_NAME)
                == 0)
        {
            return ii;
        }
    }
    return -1;
}
#endif /* !FPS_STANDALONE_SKIP_STUBS && !MILK_NO_CLI */

/* =====================================
 * ncurses stubs (always stub in no-CLI)
 * ===================================== */

errno_t
functionparameter_CTRLscreen(
    uint32_t mode __attribute__((unused)),
    char *fpsnamemask __attribute__((unused)),
    char *fpsCTRLfifoname __attribute__((unused)),
    double timeout_sec __attribute__((unused)))
{
    return 0;
}

errno_t
processinfo_CTRLscreen(void)
{
    return 0;
}

void
TUI_printfw(
    const char *fmt __attribute__((unused)),
    ...)
{
}

/** @brief No-op stub: cursor newline (standalone). */
void TUI_newline(void) {}
/** @brief No-op stub: enable reverse video (standalone). */
void screenprint_setreverse(void) {}
/** @brief No-op stub: disable reverse video (standalone). */
void screenprint_unsetreverse(void) {}

void
screenprint_setcolor(int p __attribute__((unused)))
{
}

void
screenprint_unsetcolor(int p __attribute__((unused)))
{
}

void
TUI_set_screenprintmode(
    int m __attribute__((unused)))
{
}

errno_t
TUI_init_terminal(
    short unsigned int *wrow __attribute__((unused)),
    short unsigned int *wcol __attribute__((unused)))
{
    return 0;
}

int
get_singlechar_nonblock(void)
{
    return -1;
}

/** @brief No-op stub: restore terminal (standalone). */
errno_t TUI_exit(void)
{
    return 0;
}
