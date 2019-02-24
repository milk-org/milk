# Function Parameter Structure (FPS) {#page_FunctionParameterStructure}

@note This file: ./src/CommandLineInterface/doc/FunctionParameterStructure.md

[TOC]




---


**The function  parameter structure (fps) exposes a function's internal variables for read and/or write.**

**The fps is stored in shared memory, in /tmp/<fpsname>.fps.shm.**




---


# 1. Overview and background {#page_FunctionParameterStructure_Overview}

## 1.1. Main elements

A FPS-enabled function will have the following elements:
- The shared memory FPS: /tmp/<fpsname>.fps.shm
- A configuration process that manages the FPS entries
- A run process (the function itself)



## 1.2. FPS name

<fpsname> consists of a root name (string), and a series of optional integers, each printed on two digits:

	<fpsname> = <fpsnameroot>.<opt0>.<opt1>...

Examples:

	myfps            # simple name, no optional integers
	myfps-00         # optional integer 00
	myfps-43-20-02   # 3 optional integers
	
@warning The FPS name does not need to match the process or function name. FPS name is specified in the CLI function as described in @ref page_FunctionParameterStructure_WritingCLIfunc.



## 1.3. FPS-related entities

name                                  | Type           | Description        | Origin
--------------------------------------|----------------|--------------------|---------------------------------
/tmp/<fpsname>.fps.shm                | shared memory  | FP structure       | Created by FPS init function 
./fpscmd/<fpsnameroot>-confinit       | script         | Initialize FPS     | Built by fpsmkcmd, can be user-edited
./fpscmd/<fpsnameroot>-confstart      | script         | Start CONF process | Built by fpsmkcmd, can be user-edited
./fpscmd/<fpsnameroot>-runstart       | script         | Start RUN process  | Built by fpsmkcmd, can be user-edited
./fpscmd/<fpsnameroot>-runstop        | script         | Stop RUN process   | Built by fpsmkcmd, can be user-edited
<fpsname>-conf                        | tmux session   | where CONF runs    | Set up by fpsCTRL
<fpsname>-run                         | tmux session   | where RUN runs     | Set up by fpsCTRL
./fpsconf/<fpsname>/...               | ASCII file     | parameter value    | <TBD>


---



	



# 2. FPS user interface {#page_FunctionParameterStructure_UserInterface}


Steps:

	$ vim fpslist.txt  # Edit file, listing functions and corresponding FPS names that will be used
	$ fpsmkcmd         # create FPS scripts in `./fpscmd/`
	$ ./fpsinitscript` # script to create FPS shared memory structures
	$ fpsCTRL -m _ALL  # FPS control tool, scan ALL FPSs (-m: force match with fpscmd/fpslist.txt) 


## 2.1. Building command scripts from a `fpslist.txt` file {#page_FunctionParameterStructure_WritingFPSCMDscripts}


The user-provided `fpslist.txt` file lists the functions and corresponding FPS names that will be in use:

~~~
# List of FPS-enabled function

fpsrootname0	CLIcommand0
fpsrootname1	CLIcommand1		optarg00	optarg01
~~~

FPS command scripts are built by

	$ fpsmkcmd
	
The command will create the FPS command scripts in directory `./fpscmd/`, which are then called by the @ref page_FunctionParameterStructure_fpsCTRL to control the CONF and RUN processes.

	

## 2.2. fpsCTRL tool {#page_FunctionParameterStructure_fpsCTRL}

The FPS control tool is started from the command line :

	$ fpsCTRL



---


# 3. Writing the CLI function (in <module>.c file)  {#page_FunctionParameterStructure_WritingCLIfunc}


A single CLI function, named <functionname>_cli, will take the following arguments:
- arg1: A command code
- arg2: Optional argument 

The command code is a string, and will determine the action to be executed:
- `_CONFINIT_`  : Initialize FPS for the function
- `_CONFSTART_` : Start the FPS configuration process
- `_CONFSTOP_`  : Stop the FPS configuration process
- `_RUNSTART_`  : Start the run process
- `_RUNSTOP_`   : Stop the run process
 
 
 
@note Why Optional argument to CLI function ?
@note Multiple instances of a C function may need to be running, each with its own FPS. An optional argument provides a mechanism to differentiate the FPSs. It is appended to the FPS name following a dash. The optional argument can be a number (usually integer) or a string.
 

Example source code below, assuming one optional long type argument:

~~~~{.c}
int_fast8_t MyFunction_cli()
{
    char fpsname[200];

    if( CLI_checkarg(1,5) + CLI_checkarg(2,2) == 0 )  // check that first arg is string, second arg is int
    {
		int OptionalArg00 = data.cmdargtoken[2].val.numl;
		
		// Set FPS interface name
		// By convention, if there are optional arguments, they should be appended to the fps name
		//
		if(data.processnameflag == 0) // name fps to something different than the process name
			sprintf(fpsname, "DMcomb-%02d", OptionalArg00);  
		else // Automatically set fps name to be process name up to first instance of character '.'
			strcpy(fpsname, data.processname0);
		

        if( strcmp(data.cmdargtoken[1].val.string,"_CONFINIT_") == 0 )   // Initialize FPS and conf process
        {
            printf("Function parameters configure\n");
            MyFunction_FPCONF( fpsname, CMDCODE_CONFINIT, OptionalArg00);
            return EXIT_SUCCESS;
        }

        if( strcmp(data.cmdargtoken[1].val.string,"_CONFSTART_") == 0 )   // Start conf process
        {
            printf("Function parameters configure\n");
            MyFunction_FPCONF( fpsname, CMDCODE_CONFSTART, OptionalArg00);
            return EXIT_SUCCESS;
        }

        if( strcmp(data.cmdargtoken[1].val.string,"_CONFSTOP_") == 0 )  // Stop conf process
        {
            printf("Function parameters configure\n");
            MyFunction_FPCONF( fpsname, CMDCODE_CONFSTOP, OptionalArg00);
            return EXIT_SUCCESS;
        }

        if( strcmp(data.cmdargtoken[1].val.string,"_RUNSTART_") == 0 )  // Run process
        {
            printf("Run function\n");
            MyFunction_RUN( fpsname );
            return EXIT_SUCCESS;
        }

        if( strcmp(data.cmdargtoken[1].val.string,"_RUNSTOP_") == 0 )  // Run process
        {
            printf("Run function\n");
            MyFunction_Stop( OptionalArg00 );
            return EXIT_SUCCESS;
        }
	}
	else
		return EXIT_FAILURE;
	
}
~~~~



---


# 4. Writing function prototypes (in <module>.h) {#page_FunctionParameterStructure_WritingPrototypes}



~~~~{.c}
int MyFunction_FPCONF(char *fpsname, uint32_t CMDmode, long optarg00);
int MyFunction_RUN(char *fpsname);
~~~~ 



---

# 5. Writing CONF function (in source .c file) {#page_FunctionParameterStructure_WritingCONFfunc}



~~~~{.c} 
//
// manages configuration parameters
// initializes configuration parameters structure
//
int MyFunction_FPCONF(
    char *fpsname,
    uint32_t CMDmode,
    long optarg00
)
{
	uint16_t loopstatus;
	

	// ===========================
	// SETUP FPS
	// ===========================
	
    FUNCTION_PARAMETER_STRUCT fps = function_parameter_FPCONFsetup(fpsname, CMDmode, &loopstatus);
    if( loopstatus == 0 ) // stop fps
        return 0;




	// ===========================
	// ALLOCATE FPS ENTRIES	
	// ===========================
	
	void *pNull = NULL;
    uint64_t FPFLAG;
    
    // Entries are added one by one with function_parameter_add_entry()
    // For each entry, we record the function parameter index (fpi_) returned by the function so that parameters can conveniently be accesses in the "LOGIC" section
    // Arguments:
    //   - pointer to fps
    //   - parameter path for GUI
    //   - description
    //   - type
    //   - flags
    //   - initialization pointer. If pNull, then the variable is not initialized
     
    long fpi_param01 = function_parameter_add_entry(&fps, ".param01", "First parameter", FPTYPE_INT64, FPFLAG_DFT_INPUT, pNull);
    
    // This parameter will be intitialized to a value of 5, min-max range from 0 to 10, and current value 5
    long param02default[4] = { 5, 0, 10, 5 };
    FPFLAG = FPFLAG_DFT_INPUT | FPFLAG_MINLIMIT | FPFLAG_MAXLIMIT;  // required to enforce the min and max limits
    FPFLAG &= ~FPFLAG_WRITECONF;  // Don't allow parameter to be written during configuration
    FPFLAG &= ~FPFLAG_WRITERUN;   // Don't allow parameter to be written during run
    long fpi_param02 = function_parameter_add_entry(&fps, ".param02", "Second parameter", FPTYPE_INT64, FPFLAG, &param02default);
    
    // if parameter type = FPTYPE_FLOAT32, make sure default is declared as float[4]
    // if parameter type = FPTYPE_FLOAT64, make sure default is declared as double[4]
    float gaindefault[4] = { 0.01, 0.0, 1.0, 0.01 };
    FPFLAG = FPFLAG_DFT_INPUT | FPFLAG_MINLIMIT | FPFLAG_MAXLIMIT;  // required to enforce the min and max limits
    long fpi_gain = function_parameter_add_entry(&fps, ".gain", "gain value", FPTYPE_FLOAT32, FPFLAG, &gaindefault);


	// This parameter is a ON / OFF toggle
	long fpi_gainset = function_parameter_add_entry(&fps, ".option.gainwrite", "gain can be changed", FPTYPE_ONOFF, FPFLAG_DFT_INPUT, pNull);

	



	// =====================================
	// PARAMETER LOGIC AND UPDATE LOOP
	// =====================================

	while ( loopstatus == 1 )
	{
		if( function_parameter_FPCONFloopstep(&fps, CMDmode, &loopstatus) == 1) // Apply logic if update is needed
		{
			// here goes the logic
			if ( fps.parray[fpi_gainset].status & FPFLAG_ONOFF )  // ON state
                {
                    fps.parray[fpi_gain].status |= FPFLAG_WRITERUN;
                    fps.parray[fpi_gain].status |= FPFLAG_USED;
                    fps.parray[fpi_gain].status |= FPFLAG_VISIBLE;
                }
                else // OFF state
                {
                    fps.parray[fpi_gain].status &= ~FPFLAG_WRITERUN;
                    fps.parray[fpi_gain].status &= ~FPFLAG_USED;
                    fps.parray[fpi_gain].status &= ~FPFLAG_VISIBLE;
                }
                
            functionparameter_CheckParametersAll(&fps);  // check all parameter values
		}		

	}

	function_parameter_FPCONFexit( &fps );

    return EXIT_SUCCESS;
}
~~~~	





---



# 6. Writing RUN function (in source .c file) {#page_FunctionParameterStructure_WritingRUNfunc}


The RUN function will connect to the FPS and execute the run loop. 

## 6.1. A simple example {#page_FunctionParameterStructure_WritingRUNfunc_simple}


~~~~{.c}
//
// run loop process
//
int MyFunction_RUN(
    char *fpsname
)
{

	// ===========================
	// CONNECT TO FPS
	// ===========================
	
	FUNCTION_PARAMETER_STRUCT fps;
	
	if(function_parameter_struct_connect(fpsname, &fps) == -1)
	{
		printf("ERROR: fps \"%s\" does not exist -> running without FPS interface\n", fpsname);
		return EXIT_FAILURE;
	}

	fps.md->runpid = getpid();  // write process PID into FPS

	
	
	// ===============================
	// GET FUNCTION PARAMETER VALUES
	// ===============================

	// parameters are addressed by their tag name
		
	// These parameters are read once, before running the loop
	//
	int param01 = functionparameter_GetParamValue_INT64(&fps, ".param01");
	int param02 = functionparameter_GetParamValue_INT64(&fps, ".param02");


	// This parameter value will be tracked during loop run, so we create a pointer for it
	// The corresponding function is functionparameter_GetParamPtr_<TYPE>
	//
	float *gain = functionparameter_GetParamPtr_FLOAT32(&fps, ".status.loopcnt");




	// ===============================
	// RUN LOOP
	// ===============================

	int loopOK = 1;
	while( loopOK == 1 )
	{
		// here we compute what we need...
		//
		
		
		// Note that some mechanism is required to set loopOK to 0 when MyFunction_Stop() is called
		// This can use a separate share memory path
	}

	return EXIT_SUCCESS;
}
~~~~




## 6.2. RUN function with FPS and processinfo {#page_FunctionParameterStructure_WritingRUNfunc_processinfo}


In this example, the loop process supports both FPS and processinfo.
This is the preferred way to code a loop process.

The example also shows using FPS to set the process realtime priority.


~~~~{.c}
//
// run loop process
//
int MyFunction_RUN(
    char *fpsname
)
{
	int RT_priority = 95; // Any number from 0-99. Higher number = stronger priority.
	struct sched_param schedpar;



	// ===========================
	// Connect to FPS 
	// ===========================

	FUNCTION_PARAMETER_STRUCT fps;
	if(function_parameter_struct_connect(fpsname, &fps) == -1)
	{
		printf("ERROR: fps \"%s\" does not exist -> running without FPS interface\n", fpsname);
		return EXIT_FAILURE;
	}
	
	fps.md->runpid = getpid();  // write process PID into FPS

	
	// GET FUNCTION PARAMETER VALUES
	// parameters are addressed by their tag name
	
	// These parameters are read once, before running the loop
	//
	int param01 = functionparameter_GetParamValue_INT64(&fps, ".param01");
	int param02 = functionparameter_GetParamValue_INT64(&fps, ".param02");


	// This parameter value will be tracked during loop run, so we create a pointer for it
	// The corresponding function is functionparameter_GetParamPtr_<TYPE>
	//
	float *gain = functionparameter_GetParamPtr_FLOAT32(&fps, ".status.loopcnt");




	// ===========================
	// processinfo support 
	// ===========================

    PROCESSINFO *processinfo;
    if(data.processinfo==1)
    {
        // CREATE PROCESSINFO ENTRY
        // see processtools.c in module CommandLineInterface for details
        //
        
        char pinfoname[200];
        char pinfostring[200];
        
        sprintf(pinfoname, "%s", fpsname);  // we re-use fpsname as processinfo name
        processinfo = processinfo_shm_create(pinfoname, 0);
        
        
        strcpy(processinfo->source_FUNCTION, __FUNCTION__);
        strcpy(processinfo->source_FILE,     __FILE__);
        processinfo->source_LINE = __LINE__;
        
        processinfo->loopstat = 0; // loop initialization

        char msgstring[200];
        sprintf(msgstring, "loopfunction example");
		processinfo_WriteMessage(processinfo, msgstring);
    }


	// Process signals are caught for suitable processing and reporting.
	processinfo_CatchSignals();



	// ===========================
	// Set realtime priority
	// ===========================

    schedpar.sched_priority = RT_priority;
#ifndef __MACH__
    r = seteuid(data.euid); // This goes up to maximum privileges
    sched_setscheduler(0, SCHED_FIFO, &schedpar); //other option is SCHED_RR, might be faster
    r = seteuid(data.ruid); //Go back to normal privileges
#endif





	// ===========================
	// Start loop
	// ===========================

    int loopCTRLexit = 0; // toggles to 1 when loop is set to exit cleanly
    if(data.processinfo==1)
        processinfo->loopstat = 1;

	int loopOK = 1;
	while( loopOK == 1 )
	{
	
		if(data.processinfo==1)
        {
            while(processinfo->CTRLval == 1)  // pause
                usleep(50);

            if(processinfo->CTRLval == 2) // single iteration
                processinfo->CTRLval = 1;

            if(processinfo->CTRLval == 3) // exit loop
                loopCTRLexit = 1;
        }

	
	
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
		// Here we compute what we need...
		//
		}
		
		// Post semaphore(s) and counter(s) 
		
		// computation done
		if((data.processinfo==1)&&(processinfo->MeasureTiming==1))
			processinfo_exec_end(processinfo);
		
		
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




	return EXIT_SUCCESS;
}
~~~~



----




