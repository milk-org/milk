/**
 * @file    image_format.c
 * @brief   Convert between image formats
 *
 * read and write images other than FITS
 *
 */

#define MODULE_SHORTNAME_DEFAULT "imgformat"
#define MODULE_DESCRIPTION "Conversion between image format, I/O"

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"

#include "CR2toFITS.h"
#include "FITS_to_floatbin_lock.h"
#include "FITS_to_ushortintbin_lock.h"
#include "combineHDR.h"
#include "stream_temporal_stats.h"
#include "extract_RGGBchan.h"
#include "extract_utr.h"
#include "imtoASCII.h"
#include "loadCR2toFITSRGB.h"
#include "read_binary32f.h"
#include "writeBMP.h"


/*typedef struct
{
    int rows;
    int cols;
    unsigned char *data;
} sImage;
*/
/* This pragma is necessary so that the data in the structures is aligned to 2-byte
   boundaries.  Some different compilers have a different syntax for this line.  For
   example, if you're using cc on Solaris, the line should be #pragma pack(2).
*/
//#pragma pack(2)

static errno_t init_module_CLI()
{
    CLIADDCMD_image_format__extractRGGBchan();

    CLIADDCMD_image_format__combineHDR();
    CLIADDCMD_image_format__cred_cds_utr();
    CLIADDCMD_image_format__temporal_stats();

    CLIADDCMD_image_format__imtoASCII();

    CLIADDCMD_image_format__mkBMPimage();
    //	writeBMP_addCLIcmd();

    CLIADDCMD_image_format__CR2toFITS();
    CLIADDCMD_image_format__loadCR2toFITSRGB();
    CLIADDCMD_image_format__floatbin_lock();
    CLIADDCMD_image_format__ushortintbin_lock();
    CLIADDCMD_image_format__read_binary32f();

    // add atexit functions here

    return RETURN_SUCCESS;
}

MILK_MODULE(image_format, init_module_CLI, NULL);
