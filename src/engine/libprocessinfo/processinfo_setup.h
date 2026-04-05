/**
 * @file processinfo_setup.h
 * @brief Processinfo setup module
 */

#ifndef _PROCESSINFO_SETUP_H
#define _PROCESSINFO_SETUP_H

#include "processinfo_shm_create.h"
#include <string.h>

PROCESSINFO *processinfo_setup(char       *pinfoname,
                               const char *descriptionstring,
                               const char *msgstring,
                               const char *functionname,
                               const char *filename,
                               int         linenumber);

errno_t processinfo_error(PROCESSINFO *processinfo, char *errmsgstring);

errno_t processinfo_loopstart(PROCESSINFO *processinfo);

/**
 * @brief Initialize an auxiliary PROCESSINFO struct with source coordinates.
 *
 * Use this for networking or sub-thread loops that need separate
 * SHM tracking without overriding the executable's main static processinfo.
 *
 * @param pinfo_ptr  (PROCESSINFO *) pointer variable to assign.
 * @param pinfoname  (const char *) name of the processinfo instance.
 * @param descr      (const char *) description (can be NULL).
 * @param msg        (const char *) status message (can be NULL).
 */
#define PROCESSINFO_AUX_SETUP(pinfo_ptr, pinfoname, descr, msg) \
    do { \
        (pinfo_ptr) = processinfo_shm_create((pinfoname), 0); \
        if ((pinfo_ptr) != NULL) { \
            (pinfo_ptr)->loopstat = 0; \
            strncpy((pinfo_ptr)->source_FUNCTION, __FUNCTION__, \
                STRINGMAXLEN_PROCESSINFO_SRCFUNC - 1); \
            strncpy((pinfo_ptr)->source_FILE, __FILE__, \
                STRINGMAXLEN_PROCESSINFO_SRCFILE - 1); \
            (pinfo_ptr)->source_LINE = __LINE__; \
            if (descr) { \
                strncpy((pinfo_ptr)->description, (descr), \
                    STRINGMAXLEN_PROCESSINFO_DESCRIPTION - 1); \
            } \
            if (msg) { \
                processinfo_WriteMessage((pinfo_ptr), (msg)); \
            } \
        } \
    } while (0)

#endif
