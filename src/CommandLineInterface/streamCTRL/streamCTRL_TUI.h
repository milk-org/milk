
/**
 * @file streamCTRL.h
 * @brief Data streams control panel
 *
 * Manages data streams
 *
 *
 */

#ifndef _STREAMCTRL_H
#define _STREAMCTRL_H

/* =============================================================================================== */
/* =============================================================================================== */
/*                                        HEADER FILES                                             */
/* =============================================================================================== */
/* =============================================================================================== */

#include <stdint.h>
#include <unistd.h> // getpid()

/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */

#define STRINGLENMAX 32

#define streamNBID_MAX      10000
#define streamOpenNBpid_MAX 100

#define STRINGMAXLEN_STREAMINFO_NAME 100

#define PIDnameStringLen 12

/* =============================================================================================== */
/* =============================================================================================== */
/*                                  GLOBAL DATA DECLARATION                                        */
/* =============================================================================================== */
/* =============================================================================================== */

typedef struct
{
    char sname[STRINGMAXLEN_STREAMINFO_NAME]; // stream name
    int  SymLink;
    char linkname
    [STRINGMAXLEN_STREAMINFO_NAME]; // if stream is sym link, resolve link name

    imageID ID;

    pid_t streamOpenPID[streamOpenNBpid_MAX];
    int   streamOpenPID_cnt;
    int   streamOpenPID_cnt1; // number of processes accessing stream
    int   streamOpenPID_status;

    int datatype;

    double updatevalue; // higher value = more actively recent updates [Hz]
    double updatevalue_frozen;

    long long cnt0; // used to check if cnt0 has changed
    long      deltacnt0;

} STREAMINFO;

typedef struct
{
    int    twaitus; // sleep time between scans
    double dtscan;  // measured time interval between scans [s]

    int  loop; // 1 : loop     0 : exit
    long loopcnt;

    int  filter; // 1 if applying filter to name
    char namefilter[STRINGLENMAX];

    int WriteFlistToFile; // 1 if output to file

    STREAMINFO *sinfo;
    long        NBstream;
    int         fuserUpdate;
    int         fuserUpdate0;
    int         sindexscan;
    char      **PIDtable; // stores names of PIDs

} STREAMINFOPROC;

/* =============================================================================================== */
/* =============================================================================================== */
/*                                    FUNCTIONS SOURCE CODE                                        */
/* =============================================================================================== */
/* =============================================================================================== */

#ifdef __cplusplus
extern "C"
{
#endif

/**
* INITIALIZE ncurses
*
*/

int get_process_name_by_pid(const int pid, char *pname);

int streamCTRL_CatchSignals();

int
find_streams(STREAMINFO *streaminfo, int filter, const char *namefilter);

void *streamCTRL_scan(void *thptr);

/**
* ## Purpose
*
* Control screen for stream structures
*
* ## Description
*
* Relies on ncurses for display\n
*
*
*/

errno_t streamCTRL_CTRLscreen();

long image_ID_from_images(IMAGE *images, const char *name);

long image_get_first_ID_available_from_images(IMAGE *images);

#ifdef __cplusplus
}
#endif

#endif // _STREAMCTRL_H
