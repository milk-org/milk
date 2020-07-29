/**
 * @file function_parameters.c
 * @brief Tools to help expose and control function parameters
 *
 * @see @ref page_FunctionParameterStructure
 *
 * @defgroup FPSconf Configuration function for Function Parameter Structure (FPS)
 * @defgroup FPSrun  Run function using Function Parameter Structure (FPS)
 *
 */


#define _GNU_SOURCE

/* =============================================================================================== */
/* =============================================================================================== */
/*                                        HEADER FILES                                             */
/* =============================================================================================== */
/* =============================================================================================== */


#ifndef __STDC_LIB_EXT1__
typedef int errno_t;
#endif

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <limits.h>
#include <sys/syscall.h> // needed for tid = syscall(SYS_gettid);
#include <errno.h>
#include <stdarg.h>
//#include <termios.h>
#include <time.h>

#include <sys/types.h>

#include <ncurses.h>
#include <dirent.h>

#include <pthread.h>
#include <fcntl.h> // for open
#include <unistd.h> // for close
#include <sys/mman.h> // mmap
#include <sys/stat.h> // fstat
#include <signal.h>
#include <unistd.h> // usleep

//#include <sys/ioctl.h> // for terminal size


#ifndef STANDALONE
#include <CommandLineInterface/CLIcore.h>
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_tools/COREMOD_tools.h"
#include "COREMOD_memory/COREMOD_memory.h"
#define SHAREDSHMDIR data.shmdir
#else
#include "standalone_dependencies.h"
#endif

#include "function_parameters.h"
#include "TUItools.h"

#include "fps_process_user_key.h"
#include "fps_CTRLscreen.h"


/* =============================================================================================== */
/* =============================================================================================== */
/*                                      DEFINES, MACROS                                            */
/* =============================================================================================== */
/* =============================================================================================== */



#define NB_FPS_MAX 100

#define MAXNBLEVELS 20









