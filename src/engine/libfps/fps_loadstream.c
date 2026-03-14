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
    FUNCTION_PARAMETER_STRUCT *fps,
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
        printf("\033[1;31mFAILURE\033[0m:"
               " invalid stream modifier "
               "prefix in \"%s\"\n", rawname);
        exit(EXIT_FAILURE);
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
            printf("\033[1;31mFAILURE\033[0m:"
                   " @N modifier — "
                   "stream \"%s\" already "
                   "exists (ID %ld)\n",
                   sp.name, (long) probeID);
            fps->parray[pindex].fpflag =
                saved_flags;
            exit(EXIT_FAILURE);
        }
    }

    /* Load using bare name */
    ID = COREMOD_IOFITS_LoadMemStream(
        sp.name,
        &(fps->parray[pindex].fpflag),
        &imLOC);

    /* Concise one-line status */
    printf("  stream \"%s\"", sp.name);
    if (sp.loc != '\0' ||
        sp.must_exist || sp.must_new)
    {
        char label[8];
        fps_streamname_modifier_label(
            &sp, label, sizeof(label));
        printf(" %s", label);
    }
    if (ID >= 0)
    {
        printf(" -> \033[32mFOUND\033[0m"
               " (ID %ld)\n",
               (long) ID);
    }
    else
    {
        printf(" -> \033[33mNOT FOUND\033[0m"
               "\n");
    }

    /* Restore original flags */
    fps->parray[pindex].fpflag = saved_flags;

    /* Location modifier enforcement */
    if (sp.loc == 'L' && ID >= 0 &&
        imLOC == STREAM_LOAD_SOURCE_SHAREMEM)
    {
        printf("\033[1;31mFAILURE\033[0m:"
               " @L modifier — "
               "stream \"%s\" is in shared"
               " memory, not local\n",
               sp.name);
        exit(EXIT_FAILURE);
    }
    if (sp.loc == 'S' && ID >= 0 &&
        imLOC == STREAM_LOAD_SOURCE_LOCALMEM)
    {
        printf("\033[1;31mFAILURE\033[0m:"
               " @S modifier — "
               "stream \"%s\" is in local"
               " memory, not shared\n",
               sp.name);
        exit(EXIT_FAILURE);
    }

    /* must-exist check */
    if (sp.must_exist && ID == -1)
    {
        printf("\033[1;31mFAILURE\033[0m:"
               " @E modifier — "
               "stream \"%s\""
               " not found\n",
               sp.name);
        exit(EXIT_FAILURE);
    }

    /* Standard required-stream checks */
    if (fpsconnectmode == FPSCONNECT_CONF)
    {
        if (fps->parray[pindex].fpflag
            & FPFLAG_STREAM_CONF_REQUIRED)
        {
            printf("    FPFLAG_STREAM_CONF_"
                   "REQUIRED\n");
            if (ID == -1)
            {
                printf(
                    "FAILURE: FPSCONNECT_CONF "
                    "Required stream %s could "
                    "not be loaded\n",
                    sp.name);
                exit(EXIT_FAILURE);
            }
        }
    }

    if (fpsconnectmode == FPSCONNECT_RUN)
    {
        if (fps->parray[pindex].fpflag
            & FPFLAG_STREAM_RUN_REQUIRED)
        {
            printf("    FPFLAG_STREAM_RUN_"
                   "REQUIRED\n");
            if (ID == -1)
            {
                printf(
                    "FAILURE: FPSCONNECT_RUN "
                    "Required stream %s could "
                    "not be loaded\n",
                    sp.name);
                exit(EXIT_FAILURE);
            }
        }
    }

    return ID;
}
