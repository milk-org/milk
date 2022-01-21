/**
 * @file fps_WriteParameterToDisk.c
 *
 */

#include <sys/syscall.h> // needed for tid = syscall(SYS_gettid);
#include <unistd.h>

#include "CommandLineInterface/CLIcore.h"

#include "fps_GetFileName.h"

/** @brief Write parameter to disk
 *
 * ## TAG names
 *
 * One of the following:
 * - "setval"  Set value
 * - "fpsname" Name of FPS to which parameter belongs
 * - "fpsdir"  FPS directory
 * - "minval"  Minimum value (if applicable)
 * - "maxval"  Maximum value (if applicable)
 * - "currval" Current value (if applicable)
 *
 *
 */
int functionparameter_WriteParameterToDisk(FUNCTION_PARAMETER_STRUCT *fpsentry,
                                           int                        pindex,
                                           char                      *tagname,
                                           char *commentstr)
{
    char  fname[200];
    FILE *fp;

    // create time change tag
    pid_t tid;
    tid = syscall(SYS_gettid);

    // Get GMT time
    char            timestring[200];
    struct timespec tnow;
    time_t          now;

    clock_gettime(CLOCK_REALTIME, &tnow);
    now = tnow.tv_sec;
    struct tm *uttime;
    uttime = gmtime(&now);

    sprintf(timestring,
            "%04d%02d%02d%02d%02d%02d.%09ld %8ld [%6d %6d] %s",
            1900 + uttime->tm_year,
            1 + uttime->tm_mon,
            uttime->tm_mday,
            uttime->tm_hour,
            uttime->tm_min,
            uttime->tm_sec,
            tnow.tv_nsec,
            fpsentry->parray[pindex].cnt0,
            getpid(),
            (int) tid,
            commentstr);

    if (strcmp(tagname, "setval") == 0) // VALUE
        {
            functionparameter_GetFileName(fpsentry,
                                          &(fpsentry->parray[pindex]),
                                          fname,
                                          tagname);
            fp = fopen(fname, "w");
            switch (fpsentry->parray[pindex].type)
                {

                case FPTYPE_INT64:
                    fprintf(fp,
                            "%10ld  # %s\n",
                            fpsentry->parray[pindex].val.i64[0],
                            timestring);
                    break;

                case FPTYPE_FLOAT64:
                    fprintf(fp,
                            "%18f  # %s\n",
                            fpsentry->parray[pindex].val.f64[0],
                            timestring);
                    break;

                case FPTYPE_FLOAT32:
                    fprintf(fp,
                            "%18f  # %s\n",
                            fpsentry->parray[pindex].val.f32[0],
                            timestring);
                    break;

                case FPTYPE_PID:
                    fprintf(fp,
                            "%18ld  # %s\n",
                            (long) fpsentry->parray[pindex].val.pid[0],
                            timestring);
                    break;

                case FPTYPE_TIMESPEC:
                    fprintf(fp,
                            "%15ld %09ld  # %s\n",
                            (long) fpsentry->parray[pindex].val.ts[0].tv_sec,
                            (long) fpsentry->parray[pindex].val.ts[0].tv_nsec,
                            timestring);
                    break;

                case FPTYPE_FILENAME:
                    fprintf(fp,
                            "%s  # %s\n",
                            fpsentry->parray[pindex].val.string[0],
                            timestring);
                    break;

                case FPTYPE_FITSFILENAME:
                    fprintf(fp,
                            "%s  # %s\n",
                            fpsentry->parray[pindex].val.string[0],
                            timestring);
                    break;

                case FPTYPE_EXECFILENAME:
                    fprintf(fp,
                            "%s  # %s\n",
                            fpsentry->parray[pindex].val.string[0],
                            timestring);
                    break;

                case FPTYPE_DIRNAME:
                    fprintf(fp,
                            "%s  # %s\n",
                            fpsentry->parray[pindex].val.string[0],
                            timestring);
                    break;

                case FPTYPE_STREAMNAME:
                    fprintf(fp,
                            "%s  # %s\n",
                            fpsentry->parray[pindex].val.string[0],
                            timestring);
                    break;

                case FPTYPE_STRING:
                    fprintf(fp,
                            "%s  # %s\n",
                            fpsentry->parray[pindex].val.string[0],
                            timestring);
                    break;

                case FPTYPE_ONOFF:
                    if (fpsentry->parray[pindex].fpflag & FPFLAG_ONOFF)
                        {
                            fprintf(fp,
                                    "1  %10s # %s\n",
                                    fpsentry->parray[pindex].val.string[1],
                                    timestring);
                        }
                    else
                        {
                            fprintf(fp,
                                    "0  %10s # %s\n",
                                    fpsentry->parray[pindex].val.string[0],
                                    timestring);
                        }
                    break;

                case FPTYPE_FPSNAME:
                    fprintf(fp,
                            "%s  # %s\n",
                            fpsentry->parray[pindex].val.string[0],
                            timestring);
                    break;
                }
            fclose(fp);
        }

    if (strcmp(tagname, "minval") == 0) // MIN VALUE
        {
            functionparameter_GetFileName(fpsentry,
                                          &(fpsentry->parray[pindex]),
                                          fname,
                                          tagname);

            switch (fpsentry->parray[pindex].type)
                {

                case FPTYPE_INT64:
                    fp = fopen(fname, "w");
                    fprintf(fp,
                            "%10ld  # %s\n",
                            fpsentry->parray[pindex].val.i64[1],
                            timestring);
                    fclose(fp);
                    break;

                case FPTYPE_FLOAT64:
                    fp = fopen(fname, "w");
                    fprintf(fp,
                            "%18f  # %s\n",
                            fpsentry->parray[pindex].val.f64[1],
                            timestring);
                    fclose(fp);
                    break;

                case FPTYPE_FLOAT32:
                    fp = fopen(fname, "w");
                    fprintf(fp,
                            "%18f  # %s\n",
                            fpsentry->parray[pindex].val.f32[1],
                            timestring);
                    fclose(fp);
                    break;
                }
        }

    if (strcmp(tagname, "maxval") == 0) // MAX VALUE
        {
            functionparameter_GetFileName(fpsentry,
                                          &(fpsentry->parray[pindex]),
                                          fname,
                                          tagname);

            switch (fpsentry->parray[pindex].type)
                {

                case FPTYPE_INT64:
                    fp = fopen(fname, "w");
                    fprintf(fp,
                            "%10ld  # %s\n",
                            fpsentry->parray[pindex].val.i64[2],
                            timestring);
                    fclose(fp);
                    break;

                case FPTYPE_FLOAT64:
                    fp = fopen(fname, "w");
                    fprintf(fp,
                            "%18f  # %s\n",
                            fpsentry->parray[pindex].val.f64[2],
                            timestring);
                    fclose(fp);
                    break;

                case FPTYPE_FLOAT32:
                    fp = fopen(fname, "w");
                    fprintf(fp,
                            "%18f  # %s\n",
                            fpsentry->parray[pindex].val.f32[2],
                            timestring);
                    fclose(fp);
                    break;
                }
        }

    if (strcmp(tagname, "currval") == 0) // CURRENT VALUE
        {
            functionparameter_GetFileName(fpsentry,
                                          &(fpsentry->parray[pindex]),
                                          fname,
                                          tagname);

            switch (fpsentry->parray[pindex].type)
                {

                case FPTYPE_INT64:
                    fp = fopen(fname, "w");
                    fprintf(fp,
                            "%10ld  # %s\n",
                            fpsentry->parray[pindex].val.i64[3],
                            timestring);
                    fclose(fp);
                    break;

                case FPTYPE_FLOAT64:
                    fp = fopen(fname, "w");
                    fprintf(fp,
                            "%18f  # %s\n",
                            fpsentry->parray[pindex].val.f64[3],
                            timestring);
                    fclose(fp);
                    break;

                case FPTYPE_FLOAT32:
                    fp = fopen(fname, "w");
                    fprintf(fp,
                            "%18f  # %s\n",
                            fpsentry->parray[pindex].val.f32[3],
                            timestring);
                    fclose(fp);
                    break;
                }
        }

    if (strcmp(tagname, "fpsname") == 0) // FPS name
        {
            functionparameter_GetFileName(fpsentry,
                                          &(fpsentry->parray[pindex]),
                                          fname,
                                          tagname);
            fp = fopen(fname, "w");
            fprintf(fp, "%10s    # %s\n", fpsentry->md->name, timestring);
            fclose(fp);
        }

    if (strcmp(tagname, "fpsdir") == 0) // FPS name
        {
            functionparameter_GetFileName(fpsentry,
                                          &(fpsentry->parray[pindex]),
                                          fname,
                                          tagname);
            fp = fopen(fname, "w");
            fprintf(fp, "%10s    # %s\n", fpsentry->md->workdir, timestring);
            fclose(fp);
        }

    if (strcmp(tagname, "status") == 0) // FPS name
        {
            functionparameter_GetFileName(fpsentry,
                                          &(fpsentry->parray[pindex]),
                                          fname,
                                          tagname);
            fp = fopen(fname, "w");
            fprintf(fp,
                    "%10ld    # %s\n",
                    fpsentry->parray[pindex].fpflag,
                    timestring);
            fclose(fp);
        }

    return RETURN_SUCCESS;
}
