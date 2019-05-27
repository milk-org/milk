# Process Info Structure (processinfo) {#page_ProcessInfoStructure}

@note This file: ./src/CommandLineInterface/doc/ProcessInfoStructure.md

[TOC]




---


**The PROCESSINFO structures allow fine-grained management of real-time loop processes**

**The loop can be paused, stepped or stopped, and a counter value inspected**

---


# 1. Overview {#page_ProcessInfoStructure_Overview}

The PiS is stored in shared memory as

	proc.<shortname>.<PID>.shm
	


---

# 2. Code Template {#page_ProcessInfoStructure_SampleCode}


~~~~{.c}

int functiontemplate_usingprocessinfo() {

    PROCESSINFO *processinfo;
    char pinfoname[200];   // short name for the processinfo instance, no spaces, no dot, name should be human-readable
    sprintf(pinfoname, "aol%ld-acqRM", loop);

    char pinfodescr[200];
    sprintf(pinfodescr, "NBcycle=%ld", NBcycle);

    char pinfomsg[200];
    sprintf(pinfomsg, "starting setup");




    processinfo = processinfo_setup(
        pinfoname,	         // short name for the processinfo instance, no spaces, no dot, name should be human-readable
        pinfodescr,    // description
        pinfomsg,  // message on startup
        __FUNCTION__, __FILE__, __LINE__
        );

	// OPTIONAL SETTINGS
    processinfo->MeasureTiming = 1; // Measure timing 
    processinfo->RT_priority = 20;  // RT_priority, 0-99. Larger number = higher priority. If <0, ignore
    processinfo->loopcntMax = 100;  // -1 if infinite loop




    // =============================================
    // OPTIONAL: TESTING CONDITION FOR LOOP ENTRY
    // =============================================
    // Pre-loop testing, anything that would prevent loop from starting should issue message

    int loopOK = 1;
    if(.... error condition ....)
    {
		// exit function with ERROR status
        processinfo_error(processinfo, "ERROR: no WFS reference");
        return RETURN_FAILURE;
    }




    // ==================================
    // STARTING LOOP
    // ==================================
    
    processinfo_loopstart(processinfo); // Notify processinfo that we are entering loop

    
    while(loopOK==1)
    {
	    loopOK = processinfo_loopstep(processinfo);
     
     
        //
        // Semaphore wait goes here  
        // computation starts here
       
        
        
        processinfo_exec_start(processinfo);    
        if(processinfo_compute_status(processinfo)==1)
        {
            //
		    // computation ....
		    //
        }
  
  
        // Post semaphore(s) and counter(s) 
        // computation done
        
        // process signals, increment loop counter
        processinfo_exec_end(processinfo);
    
    
        // OPTIONAL: MESSAGE WHILE LOOP RUNNING
        processinfo_WriteMessage(processinfo, "loop running fine");
    }   



    // ==================================
    // ENDING LOOP
    // ==================================
    processinfo_cleanExit(processinfo);

    return RETURN_SUCCESS;
}
~~~~



----




