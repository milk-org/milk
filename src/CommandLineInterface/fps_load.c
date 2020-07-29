/**
 * @file    fps_load.c
 * @brief   Load FPS
 */



#include "CommandLineInterface/CLIcore.h"

#include "fps_connect.h"



long function_parameter_structure_load(
	char *fpsname
)
{
	long fpsID;
	
	printf("Loading fps %s\n", fpsname);
	fflush(stdout);
	
	DEBUG_TRACEPOINT("loading FPS %s", fpsname);
	
	// next fpsID available	
	fpsID = 0;
	
	int foundflag = 0;
	
	while ( (foundflag == 0) && (fpsID < data.NB_MAX_FPS))
	{
		if ( data.fps[fpsID].SMfd < 0 )
		{			
			foundflag = 1;
		}
		else
		{
			fpsID++;
		}
	}
	
	if(foundflag == 1)
	{
		data.fps[fpsID].NBparam = function_parameter_struct_connect(fpsname, &data.fps[fpsID], FPSCONNECT_SIMPLE);
		if (data.fps[fpsID].NBparam < 1 )
		{
			printf("--- cannot load FPS %s\n", fpsname);
			fpsID = -1;
		}
		else
		{			
			printf("--- loaded FPS %s to ID %ld\n", fpsname, fpsID);
		}
	}
	else
	{
		fpsID = -1;
	}

		
	return fpsID;
}

