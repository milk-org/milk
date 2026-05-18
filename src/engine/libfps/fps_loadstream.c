/**
 * @file    fps_loadstream.c
 * @brief   Load image stream with @X: prefix support
 *
 * Parses the optional @X: modifier prefix on stream
 * name strings and applies the corresponding flags
 * before delegating to COREMOD_IOFITS_LoadMemStream.
 *
 * Modifier letters:
 *   L  Local memory only
 *   S  Force shared memory
 *   F  Load from FITS conf
 *   E  Must exist (error if missing)
 *   N  Must not exist (error if found)
 */

#include <stdio.h>

#include "fps.h"
#include "fps_streamname_parse.h"

imageID COREMOD_IOFITS_LoadMemStream(
    const char *sname,
    uint64_t   *streamflag,
    uint32_t   *imLOC
);


/**
 * @brief Load a stream, honoring @X: prefix modifiers.
 *
 * Parses the parameter's string value for a prefix,
 * maps it to FPFLAG_STREAM_LOAD_FORCE_* flags, then
 * calls COREMOD_IOFITS_LoadMemStream with the bare
 * name.  Also enforces must-exist / must-new guards.
 *
 * @param fps            FPS structure pointer
 * @param pindex         Parameter index
 * @param fpsconnectmode FPSCONNECT_CONF or _RUN
 * @return imageID or -1 on failure
 */
imageID functionparameter_LoadStream(
    FPS *fps,
    int                        pindex,
    int                        fpsconnectmode
)
{
    imageID  ID = -1;
    uint32_t imLOC;

    const char *rawname =
        fps->parray[pindex].val.string[0];

    /* Parse @X: prefix */
    FPS_STREAMNAME_PARSED sp =
        fps_streamname_parse(rawname);

    if (sp.error)
    {
        PRINT_ERROR(
               "invalid stream modifier "
               "prefix in \"%s\"", rawname);
        return -1;
    }
    /* Apply location modifier to flags */
    uint64_t saved_flags =
        fps->parray[pindex].fpflag;

    switch (sp.loc)
    {
    case 'L':
        fps->parray[pindex].fpflag |=
            FPFLAG_STREAM_LOAD_FORCE_LOCALMEM;
        break;

    case 'S':
        fps->parray[pindex].fpflag |=
            FPFLAG_STREAM_LOAD_FORCE_SHAREMEM;
        break;

    default:
        break;
    }

    /* must-new check */
    if (sp.must_new)
    {
        uint32_t probeLOC;
        uint64_t probeflags = 0;
        imageID  probeID =
            COREMOD_IOFITS_LoadMemStream(
                sp.name, &probeflags, &probeLOC);

        if (probeID != -1)
        {
            PRINT_ERROR(
                   "@N modifier — "
                   "stream \"%s\" already "
                   "exists (ID %ld)",
                   sp.name, (long) probeID);
            fps->parray[pindex].fpflag =
                saved_flags;
            return -1;
        }
    }

    /* Load using bare name */
    ID = COREMOD_IOFITS_LoadMemStream(
        sp.name,
        &(fps->parray[pindex].fpflag),
        &imLOC);

    /* Concise one-line status — include the FPS key so empty stream
     * names can be traced back to the parameter that caused them.
     * Tags appended:
     *   [empty]         — parameter value was never set
     *   [RUN-REQUIRED]  — runstart will abort if not found
     *   [CONF-REQUIRED] — confstart will abort if not found
     */
    int name_empty = (sp.name[0] == '\0');
    int run_req  = (fpsconnectmode == FPSCONNECT_RUN) &&
                   (fps->parray[pindex].fpflag & FPFLAG_STREAM_RUN_REQUIRED);
    int conf_req = (fpsconnectmode == FPSCONNECT_CONF) &&
                   (fps->parray[pindex].fpflag & FPFLAG_STREAM_CONF_REQUIRED);

    printf("  stream [%s] \"%s\"",
           fps->parray[pindex].keywordfull, sp.name);
    if (name_empty)
    {
        printf(" \033[33m[empty]\033[0m");
    }
    if (run_req)
    {
        printf(" \033[1;31m[RUN-REQUIRED]\033[0m");
    }
    if (conf_req)
    {
        printf(" \033[1;31m[CONF-REQUIRED]\033[0m");
    }
    if (sp.loc != '\0' || sp.must_exist || sp.must_new)
    {
        char label[8];
        fps_streamname_modifier_label(&sp, label, sizeof(label));
        printf(" %s", label);
    }
    if (ID >= 0)
    {
        printf(" -> \033[32mFOUND\033[0m (ID %ld)\n", (long) ID);
    }
    else if (name_empty)
    {
        printf(" -> \033[90m[skipped: parameter not configured]\033[0m\n");
    }
    else
    {
        printf(" -> \033[33mNOT FOUND\033[0m\n");
    }

    /* Restore original flags */
    fps->parray[pindex].fpflag = saved_flags;

    /* Location modifier enforcement */
    if (sp.loc == 'L' && ID >= 0 &&
        imLOC == STREAM_LOAD_SOURCE_SHAREMEM)
    {
        PRINT_ERROR(
               "@L modifier — "
               "stream \"%s\" is in shared"
               " memory, not local",
               sp.name);
        return -1;
    }
    if (sp.loc == 'S' && ID >= 0 &&
        imLOC == STREAM_LOAD_SOURCE_LOCALMEM)
    {
        PRINT_ERROR(
               "@S modifier — "
               "stream \"%s\" is in local"
               " memory, not shared",
               sp.name);
        return -1;
    }

    /* must-exist check */
    if (sp.must_exist && ID == -1)
    {
        PRINT_ERROR(
               "@E modifier — "
               "stream \"%s\""
               " not found",
               sp.name);
        return -1;
    }

    /* Required-stream enforcement: abort with a clear message
     * identifying the parameter and stream name that blocked startup. */
    if (conf_req && ID == -1)
    {
        if (name_empty)
        {
            fprintf(stderr,
                    "\n\033[1;31mABORT\033[0m confstart: "
                    "required stream parameter [%s] has no name set.\n"
                    "  Fix: milk-fps-set %s %s <stream_name>\n",
                    fps->parray[pindex].keywordfull,
                    fps->md->name,
                    fps->parray[pindex].keywordfull);
        }
        else
        {
            fprintf(stderr,
                    "\n\033[1;31mABORT\033[0m confstart: "
                    "required stream \"%s\" (parameter [%s]) "
                    "could not be loaded.\n"
                    "  Fix: create the stream or "
                    "update the parameter value.\n",
                    sp.name,
                    fps->parray[pindex].keywordfull);
        }
        fflush(stderr);
        return -1;
    }

    if (run_req && ID == -1)
    {
        if (name_empty)
        {
            fprintf(stderr,
                    "\n\033[1;31mABORT\033[0m runstart: "
                    "required stream parameter [%s] has no name set.\n"
                    "  Fix: milk-fps-set %s %s <stream_name>\n",
                    fps->parray[pindex].keywordfull,
                    fps->md->name,
                    fps->parray[pindex].keywordfull);
        }
        else
        {
            fprintf(stderr,
                    "\n\033[1;31mABORT\033[0m runstart: "
                    "required stream \"%s\" (parameter [%s]) "
                    "could not be loaded.\n"
                    "  Fix: create the stream or "
                    "update the parameter value.\n",
                    sp.name,
                    fps->parray[pindex].keywordfull);
        }
        fflush(stderr);
        return -1;
    }

    return ID;
}
