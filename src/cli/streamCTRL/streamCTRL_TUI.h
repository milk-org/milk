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



#include <stdint.h>
#include <unistd.h> // getpid()



#define STRINGLENMAX 32

#define streamNBID_MAX      10000
#define streamOpenNBpid_MAX 100

#define STRINGMAXLEN_STREAMINFO_NAME 100

#define PIDnameStringLen 12









// shared memory access permission
#define FILEMODE 0666

#define STRINGLENMAX 32

#define streamOpenNBpid_MAX 100
#define nameNBchar          100
#define PIDnameStringLen    12

/* Sort column identifiers for streamCTRL */
#define STREAM_SORT_NONE  0
#define STREAM_SORT_NAME  1
#define STREAM_SORT_TYPE  2
#define STREAM_SORT_SIZE  3
#define STREAM_SORT_CNT0  4
#define STREAM_SORT_CPID  5
#define STREAM_SORT_OPID  6
#define STREAM_SORT_FREQ  7
#define STREAM_NB_SORT_COLS 7

#define DISPLAY_MODE_HELP     1
#define DISPLAY_MODE_SUMMARY  2
#define DISPLAY_MODE_WRITE    3
#define DISPLAY_MODE_READ     4
#define DISPLAY_MODE_SPTRACE  5
#define DISPLAY_MODE_FUSER    6

#define PRINT_PID_DEFAULT          0
#define PRINT_PID_FORCE_NOUPSTREAM 1

#define NO_DOWNSTREAM_INDEX 100




typedef struct
{
    char sname[STRINGMAXLEN_STREAMINFO_NAME]; // stream name
    int  SymLink;

    // if stream is sym link, resolve link name
    char linkname[STRINGMAXLEN_STREAMINFO_NAME];

    // ISIO return value from command
    // ImageStreamIO_read_sharedmem_image_toIMAGE
    int ISIOretval;

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
    int       erased;

    double last_wave_t; /* CLOCK_MONOTONIC seconds of last deltacnt0 != 0 */

    /* 1-second block average of update frequency */
    uint64_t cnt0_avg_start; /* cnt0 at start of current 1-s window */
    double   t_avg_start;    /* CLOCK_MONOTONIC time of window start */
    double   frequ_disp;     /* averaged Hz displayed in TUI */

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

    //STREAMINFO *sinfo;
    long        NBstream;
    int         fuserUpdate;
    int         fuserUpdate0;
    int         sindexscan;
    char      **PIDtable; // stores names of PIDs

} STREAMINFOPROC;




// strructure holding data required for streamCTRL
typedef struct
{
    STREAMINFO *sinfo;

    STREAMINFOPROC *streaminfoproc;

    // pointers to images
    IMAGE          *images;

} streamCTRLarg_struct;





#ifdef __cplusplus
extern "C"
{
#endif

#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

/** Main TUI entry point — runs until user presses 'x' or SIGINT */
errno_t streamCTRL_CTRLscreen(void);

#ifdef __cplusplus
}
#endif

#endif // _STREAMCTRL_H
