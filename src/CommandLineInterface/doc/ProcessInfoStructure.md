# Process Info Structure (FPS) {#page_ProcessInfoStructure}

@note This file: ./src/CommandLineInterface/doc/ProcessInfoStructure.md

[TOC]




---


**The PROCESSINFO structures allow fine-grained management of real-time loop processes**

**The loop can be paused, stepped or stopped, and a counter value inspected**

---


# 1. Overview {#page_ProcessInfoStructure_Overview}

The PiS is stored in shared memory. 



---

# 2. Sample Code {#page_ProcessInfoStructure_SampleCode}


~~~~{.c}

// ===========================
// SETUP PROCESSINFO
// ===========================
PROCESSINFO *processinfo;
if((data.processinfo==1)&&(data.processinfoActive==0))
 {
     // CREATE PROCESSINFO ENTRY
     // see processtools.c in module CommandLineInterface for details
     //
     char pinfoname[200];  // short name for the processinfo instance
     // avoid spaces, name should be human-readable
     sprintf(pinfoname, "process-%s-to-%s", IDinname, IDoutname);
     processinfo = processinfo_shm_create(pinfoname, 0);
     processinfo->loopstat = 0; // loop initialization
     strcpy(processinfo->source_FUNCTION, __FUNCTION__);
     strcpy(processinfo->source_FILE,     __FILE__);
     processinfo->source_LINE = __LINE__;
     sprintf(processinfo->description, "computes something");
     char msgstring[200];
     sprintf(msgstring, "%s->%s", IDinname, IDoutname);
     processinfo_WriteMessage(processinfo, msgstring);
     data.processinfoActive = 1;
     processinfo->MeasureTiming = 0; // OPTIONAL: do not measure timing 
 }
 
 
// Process signals are caught for suitable processing and reporting.
processinfo_CatchSignals();


// ==================================
// TESTING CONDITION FOR LOOP ENTRY
// ==================================
// Pre-loop testing, anything that would prevent loop from starting should issue message
int loopOK = 1;
if(.... error condition ....)
  {
      sprintf(msgstring, "ERROR: no WFS reference");
      if(data.processinfo == 1)
      {
          processinfo->loopstat = 4; // ERROR
          processinfo_WriteMessage(processinfo, msgstring);
      }
      loopOK = 0;
  }



// ==================================
// STARTING LOOP
// ==================================
if(data.processinfo==1)
    processinfo->loopstat = 1;  // Notify processinfo that we are entering loop
long loopcnt = 0;
while(loopOK==1)
    {
      if(data.processinfo==1)
        {
            while(processinfo->CTRLval == 1)  // pause
                usleep(50);
            if(processinfo->CTRLval == 2) // single iteration
                processinfo->CTRLval = 1;
            if(processinfo->CTRLval == 3) // exit loop
            {
                loopOK = 0;
            }
        }
    //
    // Semaphore wait goes here  
    // computation start here
    if((data.processinfo==1)&&(processinfo->MeasureTiming==1))
        processinfo_exec_start(processinfo);
    // CTRLval = 5 will disable computations in loop (usually for testing)
    int doComputation = 1;
    if(data.processinfo == 1)
        if(processinfo->CTRLval == 5)
            doComputation = 0;
    if(doComputation==1)
    {
    //
    // computation ....
    //
    }
    // Post semaphore(s) and counter(s) 
    // computation done
    if((data.processinfo==1)&&(processinfo->MeasureTiming==1))
        processinfo_exec_end(processinfo);
    // OPTIONAL MESSAGE WHILE LOOP RUNNING
    if(data.processinfo==1)
        {
            char msgstring[200];
            sprintf(msgstring, "%d save threads", NBthreads);
            processinfo_WriteMessage(processinfo, msgstring);
        }
     // process signals, end loop
     processinfo_ProcessSignals(processinfo);
     loopcnt++;
     if(data.processinfo==1)
            processinfo->loopcnt = loopcnt;
    }   



// ==================================
// ENDING LOOP
// ==================================
if((data.processinfo==1)&&(processinfo->loopstat != 4))
   processinfo_cleanExit(processinfo);

~~~~



----




