/**
 * @file    logshmim_save.c
 * @brief   FITS save thread for stream logging
 *
 * Contains the save_telemetry_fits_function()
 * worker thread used by logshmim.c.
 *
 * @see logshmim.c for the main compute loop.
 */

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#    include "CLIcore_standalone.h"
#else
#    include "libmilkdata/milkdata.h"
#    include "milkDebugTools.h"
#    include "ImageStreamIO/ImageStreamIO.h"
#endif
#include "fps.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "milk_rt.h"
#include "timeutils.h"

#ifdef USE_CFITSIO
#    include "COREMOD_iofits/COREMOD_iofits.h"
#endif

#include "image_ID.h"
#include "shmimlog_types.h"


/**
 * ## Purpose
 *
 * Save telemetry stream data
 * Writes FITS file and timing file
 */
void *save_telemetry_fits_function(void *ptr)
{
    long *tret_ptr = (long *) calloc(1, sizeof(long));
    if (tret_ptr == NULL)
    {
        return NULL;
    }
    *tret_ptr = 0;
#define tret (*tret_ptr)
    STREAMSAVE_THREAD_MESSAGE *tmsg;
    tmsg = (STREAMSAVE_THREAD_MESSAGE *) ptr;

    struct timespec tstart;
    clock_gettime(CLOCK_MILK, &tstart);

    milkrt_RTPrio((int) tmsg->writerRTprio);

    int            NBcustomKW = 9;
    IMAGE_KEYWORD *imkwarray  = (IMAGE_KEYWORD *) calloc(NBcustomKW, sizeof(IMAGE_KEYWORD));

    snprintf(imkwarray->name, KEYWORD_MAX_STRING, "UT");
    imkwarray->type = 'S';

    timedouble_to_UTC_timeofdaystring(0.5 * tmsg->arraytime[0] +
                                          0.5 * tmsg->arraytime[tmsg->cubesize - 1],
                                      imkwarray->value.valstr, KEYWORD_MAX_STRING);
    snprintf(imkwarray->comment, KEYWORD_MAX_COMMENT,
             "HH:MM:SS.SS typical UTC"
             " at exposure");

    snprintf(imkwarray[1].name, KEYWORD_MAX_STRING, "UT-STR");
    imkwarray[1].type = 'S';
    timedouble_to_UTC_timeofdaystring(tmsg->arraytime[0], imkwarray[1].value.valstr,
                                      KEYWORD_MAX_STRING);
    snprintf(imkwarray[1].comment, KEYWORD_MAX_COMMENT,
             "HH:MM:SS.SS UTC at"
             " exposure start");

    snprintf(imkwarray[2].name, KEYWORD_MAX_STRING, "UT-END");
    imkwarray[2].type = 'S';
    timedouble_to_UTC_timeofdaystring(tmsg->arraytime[tmsg->cubesize - 1],
                                      imkwarray[2].value.valstr, KEYWORD_MAX_STRING);
    snprintf(imkwarray[2].comment, KEYWORD_MAX_COMMENT,
             "HH:MM:SS.SS UTC at"
             " exposure end");

    snprintf(imkwarray[3].name, KEYWORD_MAX_STRING, "MJD");
    imkwarray[3].type = 'D';
    imkwarray[3].value.numf =
        (0.5 * tmsg->arraytime[0] + 0.5 * tmsg->arraytime[tmsg->cubesize - 1]) / 86400.0 + 40587.0;
    snprintf(imkwarray[3].comment, KEYWORD_MAX_COMMENT,
             "Modified Julian Day"
             " at exposure");

    snprintf(imkwarray[4].name, KEYWORD_MAX_STRING, "MJD-STR");
    imkwarray[4].type       = 'D';
    imkwarray[4].value.numf = tmsg->arraytime[0] / 86400.0 + 40587.0;
    snprintf(imkwarray[4].comment, KEYWORD_MAX_COMMENT,
             "Modified Julian Day at"
             " exposure start");

    snprintf(imkwarray[5].name, KEYWORD_MAX_STRING, "MJD-END");
    imkwarray[5].type       = 'D';
    imkwarray[5].value.numf = (tmsg->arraytime[tmsg->cubesize - 1] / 86400.0) + 40587.0;
    snprintf(imkwarray[5].comment, KEYWORD_MAX_COMMENT,
             "Modified Julian Day at"
             " exposure end");

    snprintf(imkwarray[6].name, KEYWORD_MAX_STRING, "%s", TZ_MILK_STR);
    imkwarray[6].type = 'S';
    timedouble_to_UTC_timeofdaystring(
        (0.5 * tmsg->arraytime[0] + 0.5 * tmsg->arraytime[tmsg->cubesize - 1]) + TZ_MILK_UTC_OFF,
        imkwarray[6].value.valstr, KEYWORD_MAX_STRING);
    snprintf(imkwarray[6].comment, KEYWORD_MAX_COMMENT,
             "HH:MM:SS.SS typical %s"
             " at exposure",
             TZ_MILK_STR);

    snprintf(imkwarray[7].name, KEYWORD_MAX_STRING, "%s-STR", TZ_MILK_STR);
    imkwarray[7].type = 'S';
    timedouble_to_UTC_timeofdaystring(tmsg->arraytime[0] + TZ_MILK_UTC_OFF,
                                      imkwarray[7].value.valstr, KEYWORD_MAX_STRING);
    snprintf(imkwarray[7].comment, KEYWORD_MAX_COMMENT,
             "HH:MM:SS.SS typical %s"
             " at exposure start",
             TZ_MILK_STR);

    snprintf(imkwarray[8].name, KEYWORD_MAX_STRING, "%s-END", TZ_MILK_STR);
    imkwarray[8].type = 'S';
    timedouble_to_UTC_timeofdaystring(tmsg->arraytime[tmsg->cubesize - 1] + TZ_MILK_UTC_OFF,
                                      imkwarray[8].value.valstr, KEYWORD_MAX_STRING);
    snprintf(imkwarray[8].comment, KEYWORD_MAX_COMMENT,
             "HH:MM:SS.SS typical %s"
             " at exposure end",
             TZ_MILK_STR);

#ifdef USE_CFITSIO
    saveFITS_opt_trunc(tmsg->iname, tmsg->partial ? tmsg->cubesize : -1, tmsg->fname, 0,
                       tmsg->fname_auxFITSheader, imkwarray, NBcustomKW, tmsg->compress_string);
#else
    (void) tmsg->fname;
    (void) tmsg->fname_auxFITSheader;
    (void) tmsg->compress_string;
    printf("WARNING: FITS save disabled"
           " (built without cfitsio)\n");
#endif

    free(imkwarray);

    if (tmsg->saveascii == 1)
    {
        FILE *fp;

        if ((fp = fopen(tmsg->fnameascii, "w")) == NULL)
        {
            PRINT_ERROR("cannot create file \"%s\"", tmsg->fnameascii);
        }
        else
        {
            fprintf(fp, "# Telemetry stream"
                        " timing data \n");
            fprintf(fp,
                    "# File written by"
                    " function %s in file %s\n",
                    __FUNCTION__, __FILE__);
            fprintf(fp, "# \n");
            fprintf(fp, "# col1 : datacube"
                        " frame index\n");
            fprintf(fp, "# col2 : Main index\n");
            fprintf(fp, "# col3 : Time since cube"
                        " origin (logging)\n");
            fprintf(fp, "# col4 : Absolute time"
                        " (logging)\n");
            fprintf(fp, "# col5 : Absolute time"
                        " (acquisition)\n");
            fprintf(fp, "# col6 : stream cnt0"
                        " index\n");
            fprintf(fp, "# col7 : stream cnt1"
                        " index\n");
            fprintf(fp, "# \n");

            double t0;
            t0 = tmsg->arraytime[0];
            for (long k = 0; k < tmsg->cubesize; k++)
            {
                fprintf(fp,
                        "%10ld  %10lu  %15.9lf"
                        "   %20.9lf  %17.6lf"
                        "   %10ld   %10ld\n",
                        k, tmsg->arrayindex[k], tmsg->arraytime[k] - t0, tmsg->arraytime[k],
                        tmsg->arrayaqtime[k], tmsg->arraycnt0[k], tmsg->arraycnt1[k]);
            }
            fclose(fp);
        }
    }

    {
        IMGID img = imgid_make_from_name(tmsg->iname);
        resolveIMGID(&img, ERRMODE_NULL, dcimg, dcnimg);
        tret = img.ID;
    }

    struct timespec tend;
    clock_gettime(CLOCK_MILK, &tend);

    double timediff =
        1.0 * (tend.tv_sec - tstart.tv_sec) + 1.0e-9 * (tend.tv_nsec - tstart.tv_nsec);
    tmsg->timespan = timediff;

    pthread_exit(&tret);
}
