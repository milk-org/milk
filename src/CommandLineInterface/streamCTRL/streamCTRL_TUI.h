
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


errno_t streamCTRL_CTRLscreen();



#ifdef __cplusplus
}
#endif

#endif // _STREAMCTRL_H
