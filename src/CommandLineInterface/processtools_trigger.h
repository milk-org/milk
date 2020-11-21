/**
 * @file    processtools_trigger.h
 * 
 * @brief   Process triggering
 *
 *
 */


#ifndef _PROCESSTOOLS_TRIGGER_H
#define _PROCESSTOOLS_TRIGGER_H



// input stream triggering modes

// trigger immediately
#define PROCESSINFO_TRIGGERMODE_IMMEDIATE      0

// trigger when cnt0 increments
#define PROCESSINFO_TRIGGERMODE_CNT0           1

// trigger when cnt1 increments
#define PROCESSINFO_TRIGGERMODE_CNT1           2

// trigger when semaphore is posted
#define PROCESSINFO_TRIGGERMODE_SEMAPHORE      3

// trigger after a time delay
#define PROCESSINFO_TRIGGERMODE_DELAY          4


// trigger is currently waiting for input
#define PROCESSINFO_TRIGGERSTATUS_WAITING      1

#define PROCESSINFO_TRIGGERSTATUS_RECEIVED     2
#define PROCESSINFO_TRIGGERSTATUS_TIMEDOUT     3







errno_t processinfo_waitoninputstream_init(
	PROCESSINFO *processinfo,
	imageID      trigID,
	int          triggermode,
	int          semindexrequested
);

errno_t processinfo_waitoninputstream(
    PROCESSINFO *processinfo
);


#endif

