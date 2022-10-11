/**
 * @file    fps_save2disk.c
 * @brief   Save FPS content to disk
 */

#include <dirent.h>
#include <sys/stat.h>    // fstat
#include <sys/syscall.h> // needed for tid = syscall(SYS_gettid);

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_iofits/COREMOD_iofits.h"
#include "CommandLineInterface/timeutils.h"

#include "fps_GetParamIndex.h"
#include "fps_WriteParameterToDisk.h"
#include "fps_printparameter_valuestring.h"




int functionparameter_SaveParam2disk(FUNCTION_PARAMETER_STRUCT *fpsentry,
                                     const char                *paramname)
{
    int pindex;

    pindex = functionparameter_GetParamIndex(fpsentry, paramname);


    functionparameter_WriteParameterToDisk(fpsentry,
                                           pindex,
                                           "setval",
                                           "SaveParam2disk");

    return RETURN_SUCCESS;
}




int functionparameter_SaveFPS2disk_dir(FUNCTION_PARAMETER_STRUCT *fpsentry,
                                       char                      *dirname)
{
    char  fname[STRINGMAXLEN_FULLFILENAME];
    FILE *fpoutval;
    int   stringmaxlen = 500;
    char  outfpstring[stringmaxlen];

    struct stat st = {0};
    if(stat(dirname, &st) == -1)
    {
        mkdir(dirname, 0700);
    }

    sprintf(fname, "%s/%s.fps", dirname, fpsentry->md->name);
    fpoutval = fopen(fname, "w");

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
            "%04d-%02d-%02dT%02d:%02d:%02d.%09ld",
            1900 + uttime->tm_year,
            1 + uttime->tm_mon,
            uttime->tm_mday,
            uttime->tm_hour,
            uttime->tm_min,
            uttime->tm_sec,
            tnow.tv_nsec);

    fprintf(fpoutval, "# TIMESTRING %s\n", timestring);
    fprintf(fpoutval, "# PID        %d\n", getpid());
    fprintf(fpoutval, "# TID        %d\n", (int) tid);
    fprintf(fpoutval, "# root dir   %s\n", fpsentry->md->workdir);
    fprintf(fpoutval, "#\n");

    for(int pindex = 0; pindex < fpsentry->md->NBparamMAX; pindex++)
    {
        errno_t ret = functionparameter_PrintParameter_ValueString(
                          &fpsentry->parray[pindex],
                          outfpstring,
                          stringmaxlen);
        if(ret == RETURN_SUCCESS)
        {
            fprintf(fpoutval, "%s\n", outfpstring);
        }
    }
    fclose(fpoutval);

    return RETURN_SUCCESS;
}




/** @brief save entire FPS to disk
 *
 * Writes in subdirectory fps datatir
 */
int functionparameter_SaveFPS2disk(FUNCTION_PARAMETER_STRUCT *fps)
{
    char outdir[STRINGMAXLEN_FULLFILENAME];
    WRITE_FULLFILENAME(outdir, "%s/%s", fps->md->workdir, fps->md->datadir);
    functionparameter_SaveFPS2disk_dir(fps, outdir);

    char timestring[100];
    char timestringnow[100];

    // assemble timestrings
    mkUTtimestring_microsec(timestring, fps->md->runpidstarttime);
    mkUTtimestring_microsec_now(timestringnow);

    char  ffname[STRINGMAXLEN_FULLFILENAME];
    FILE *fpout;
    WRITE_FULLFILENAME(ffname,
                       "%s/%s/%s.fps.outlog",
                       fps->md->workdir,
                       fps->md->datadir,
                       fps->md->name);
    fpout = fopen(ffname, "w");
    fprintf(fpout,
            "%s %s %s fps %s %s fps\n",
            timestring,
            timestringnow,
            fps->md->name,
            fps->md->name,
            fps->md->name);
    fclose(fpout);

    return RETURN_SUCCESS;
}




/** @brief Write archive script to .log2fps entry
 *
 * Funciton to be executed inside fps->md->datadir.
 *
 * Writes a script to be executed to archive most recent data
 * in ../datadir/ (usually a sym link)
 *
 * takes fps as input
 *
 * Optional input:
 *
 * File loglist.dat in directory .conf.dirname
 *
 */
errno_t functionparameter_write_archivescript(FUNCTION_PARAMETER_STRUCT *fps)
{
    // Write archive script
    // to be executed to archive most recent calibration data
    // takes fpsname as input
    //
    FILE *fplogscript;
    char  ffname[STRINGMAXLEN_FULLFILENAME];

    char datadirname[STRINGMAXLEN_DIRNAME];

    char timestring[FUNCTION_PARAMETER_STRMAXLEN];
    strncpy(timestring,
            functionparameter_GetParamPtr_STRING(fps, ".conf.timestring"),
            FUNCTION_PARAMETER_STRMAXLEN - 1);

    WRITE_FULLFILENAME(ffname, "archlogscript.bash");

    fplogscript = fopen(ffname, "w");
    fprintf(fplogscript, "#!/bin/bash\n");
    fprintf(fplogscript, "\n");
    fprintf(fplogscript, "# %s fps.%s.dat\n", timestring, fps->md->name);

    char datestring[9];
    strncpy(datestring, timestring, 8);
    datestring[8] = '\0';

    // save FPS
    WRITE_DIRNAME(datadirname,
                  "../datadir/%s/%s/fps.%s",
                  datestring,
                  fps->md->name,
                  fps->md->name);
    fprintf(fplogscript, "mkdir -p %s\n", datadirname);
    fprintf(fplogscript,
            "cp fps.%s.dat %s/fps.%s.%s.dat\n",
            fps->md->name,
            datadirname,
            fps->md->name,
            timestring);

    // save files listed in loglist.dat
    FILE *fploglist;
    char  loglistfname[STRINGMAXLEN_FULLFILENAME];
    WRITE_FULLFILENAME(loglistfname, "loglist.dat");
    fploglist = fopen(loglistfname, "r");
    if(fploglist != NULL)
    {
        char  *line = NULL;
        size_t llen = 0;
        char   logfname[STRINGMAXLEN_FILENAME];

        while(getline(&line, &llen, fploglist) != -1)
        {
            sscanf(line, "%s", logfname);
            WRITE_DIRNAME(datadirname,
                          "../datadir/%s/%s/%s",
                          datestring,
                          fps->md->name,
                          logfname);
            fprintf(fplogscript, "mkdir -p %s\n", datadirname);
            fprintf(fplogscript,
                    "cp -r %s %s/%s.%s\n",
                    logfname,
                    datadirname,
                    logfname,
                    timestring);
        }
        fclose(fploglist);
    }

    fclose(fplogscript);
    chmod(ffname, S_IRWXU | S_IRWXG | S_IROTH);

    //    functionparameter_SetParamValue_STRING(fps, ".conf.archivescript", ffname);

    return RETURN_SUCCESS;
}




/** @brief Save image as FITS
 *
 * Standard function to save output of FPS RUN function.
 *
 * imagename is the in-memory image to be saved to disk, written as
 * outname.fits.
 *
 */
errno_t fps_write_RUNoutput_image(FUNCTION_PARAMETER_STRUCT *fps,
                                  const char                *imagename,
                                  const char                *outname)
{
    char ffname[STRINGMAXLEN_FULLFILENAME];
    char timestring[100];
    char timestringnow[100];

    // assemble timestrings
    mkUTtimestring_microsec(timestring, fps->md->runpidstarttime);
    mkUTtimestring_microsec_now(timestringnow);

    WRITE_FULLFILENAME(ffname, "%s/%s.fits", fps->md->datadir, outname);
    save_fits(imagename, ffname);

    FILE *fpout;
    WRITE_FULLFILENAME(ffname, "%s/%s.fits.outlog", fps->md->datadir, outname);
    fpout = fopen(ffname, "w");
    fprintf(fpout,
            "%s %s %s fits %s %s fits\n",
            timestring,
            timestringnow,
            outname,
            fps->md->name,
            outname);
    fclose(fpout);

    return RETURN_SUCCESS;
}




/** @brief Save text file
 *
 * Standard function to save output of FPS RUN function.
 *
 *
 */
FILE *fps_write_RUNoutput_file(FUNCTION_PARAMETER_STRUCT *fps,
                               const char                *filename,
                               const char                *extension)
{
    FILE *fp;

    char ffname[STRINGMAXLEN_FULLFILENAME];
    char timestring[100];
    char timestringnow[100];

    // assemble timestrings
    mkUTtimestring_microsec(timestring, fps->md->runpidstarttime);
    mkUTtimestring_microsec_now(timestringnow);

    WRITE_FULLFILENAME(ffname,
                       "%s/%s.%s",
                       fps->md->datadir,
                       filename,
                       extension);
    fp = fopen(ffname, "w");

    FILE *fpout;
    WRITE_FULLFILENAME(ffname,
                       "%s/%s.%s.outlog",
                       fps->md->datadir,
                       filename,
                       extension);
    fpout = fopen(ffname, "w");
    fprintf(fpout,
            "%s %s %s %s %s %s %s\n",
            timestring,
            timestringnow,
            filename,
            extension,
            fps->md->name,
            filename,
            extension);
    fclose(fpout);

    return fp;
}




/** @brief Get file extension
 */
static char *get_filename_ext(const char *filename)
{
    char *dot = strrchr(filename, '.');
    if(!dot || dot == filename)
    {
        return "";
    }
    return dot + 1;
}




static char *remove_filename_ext(const char *filename)
{
    char *tmpstring;

    if((tmpstring = malloc(strlen(filename) + 1)) == NULL)
    {
        return NULL;
    }
    strcpy(tmpstring, filename);
    char *lastdot = strrchr(tmpstring, '.');
    if(lastdot != NULL)
    {
        *lastdot = '\0';
    }
    return tmpstring;
}




/** @brief Copy file
 */
static errno_t filecopy(char *sourcefilename, char *destfilename)
{
    FILE *fp1, *fp2;
    char  ch;
    int   pos;

    if((fp1 = fopen(sourcefilename, "r")) == NULL)
    {
        printf("Cannot open file \"%s\" \n", sourcefilename);
        return RETURN_FAILURE;
    }

    fp2 = fopen(destfilename, "w");

    fseek(fp1, 0L, SEEK_END); // file pointer at end of file
    pos = ftell(fp1);
    fseek(fp1, 0L, SEEK_SET); // file pointer set at start
    while(pos--)
    {
        ch = fgetc(fp1); // copying file character by character
        fputc(ch, fp2);
    }
    fclose(fp1);
    fclose(fp2);

    return RETURN_SUCCESS;
}




/** @brief Save FPS from datadir to confdir
 *
 *	Scan datadir, looking for .outlog file(s).
 *
 * For each such file, copy <file>.outlog and <file> from datadir to confdir.
 *
 */
errno_t fps_datadir_to_confdir(FUNCTION_PARAMETER_STRUCT *fps)
{
    struct dirent *indirentry; // Pointer for directory entry
    char          *file_ext;   // extension

    // opendir() returns a pointer of DIR type.
    DIR *indir = opendir(fps->md->datadir);
    if(indir == NULL)  // opendir returns NULL if couldn't open directory
    {
        printf("Cannot open directory \"%s\"\n", fps->md->datadir);
        return RETURN_FAILURE;
    }

    DIR *outdir = opendir(fps->md->confdir);
    if(outdir == NULL)  // opendir returns NULL if couldn't open directory
    {
        printf("Cannot open directory\"%s\"", fps->md->confdir);
        return RETURN_FAILURE;
    }

    while((indirentry = readdir(indir)) != NULL)
    {
        //printf("%s\n", indirentry->d_name);
        file_ext = get_filename_ext(indirentry->d_name);

        if(strcmp(file_ext, "outlog") == 0)
        {
            char ffnamein[STRINGMAXLEN_FULLFILENAME];
            char ffnameout[STRINGMAXLEN_FULLFILENAME];

            WRITE_FULLFILENAME(ffnamein,
                               "%s/%s",
                               fps->md->datadir,
                               indirentry->d_name);
            WRITE_FULLFILENAME(ffnameout,
                               "%s/%s",
                               fps->md->confdir,
                               indirentry->d_name);
            filecopy(ffnamein, ffnameout);

            char *fnamenoext;
            fnamenoext = remove_filename_ext(indirentry->d_name);

            WRITE_FULLFILENAME(ffnamein, "%s/%s", fps->md->datadir, fnamenoext);
            WRITE_FULLFILENAME(ffnameout,
                               "%s/%s",
                               fps->md->confdir,
                               fnamenoext);
            filecopy(ffnamein, ffnameout);
        }
    }

    closedir(indir);
    closedir(outdir);

    //sleep(10);

    return RETURN_SUCCESS;
}
