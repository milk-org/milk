/**
 * @file wisdom.c
 * @brief Wisdom module
 */

/**
 * @file wisdom.c
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#include "COREMOD_memory/COREMOD_memory.h"
#else
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#include "milkDebugTools.h"
#include "fps.h"
#include "ImageStreamIO/ImageStreamIO.h"
#endif
#endif

errno_t import_wisdom()
{
    FILE *fp;
    char  wisdom_file_single[STRINGMAXLEN_FULLFILENAME];
    char  wisdom_file_double[STRINGMAXLEN_FULLFILENAME];

#ifdef FFTWMT
    WRITE_FULLFILENAME(wisdom_file_single,
                       "%s/fftwf_mt_wisdom.dat",
                       FFTCONFIGDIR);
    WRITE_FULLFILENAME(wisdom_file_double,
                       "%s/fftw_mt_wisdom.dat",
                       FFTCONFIGDIR);
#endif

#ifndef FFTWMT
    WRITE_FULLFILENAME(wisdom_file_single, "%s/fftwf_wisdom.dat", FFTCONFIGDIR);
    WRITE_FULLFILENAME(wisdom_file_double, "%s/fftw_wisdom.dat", FFTCONFIGDIR);
#endif

    int nowisdomWarning = 0;

    if((fp = fopen(wisdom_file_single, "r")) == NULL)
    {
        nowisdomWarning = 1;
        /*
        n = snprintf(
                warnmessg,
                SBUFFERSIZE,
                "No single precision wisdom file in %s\n FFTs will not be optimized,"
                " and may run slower than if a wisdom file is used\n type \"initfft\""
                " to create the wisdom file (this will take time)",
                wisdom_file_single);

        if(n >= SBUFFERSIZE)
            PRINT_ERROR("Attempted to write string buffer with too many characters");
        PRINT_WARNING(warnmessg);
        */
    }
    else
    {
#ifdef MILK_MODULE
        if(fftwf_import_wisdom_from_file(fp) == 0)
        {
            PRINT_WARNING("Error reading wisdom");
        }
#endif
        fclose(fp);
    }

    if((fp = fopen(wisdom_file_double, "r")) == NULL)
    {
        nowisdomWarning = 1;
        /*  n = snprintf(
                  warnmessg,
                  SBUFFERSIZE,
                  "No double precision wisdom file in %s\n FFTs will not be optimized,"
                  " and may run slower than if a wisdom file is used\n type \"initfft\""
                  " to create the wisdom file (this will take time)",
                  wisdom_file_double);
          if(n >= SBUFFERSIZE)
              PRINT_ERROR("Attempted to write string buffer with too many characters");
          PRINT_WARNING(warnmessg);*/
    }
    else
    {
#ifdef MILK_MODULE
        if(fftw_import_wisdom_from_file(fp) == 0)
        {
            PRINT_WARNING("Error reading wisdom");
        }
#endif
        fclose(fp);
    }

    if(nowisdomWarning == 1)
    {
        printf("    [no fftw wisdom file, run initfft to create in %s]\n",
               FFTCONFIGDIR);
    }

    return RETURN_SUCCESS;
}

errno_t export_wisdom()
{
    FILE *fp;
    char  wisdom_file_single[STRINGMAXLEN_FULLFILENAME];
    char  wisdom_file_double[STRINGMAXLEN_FULLFILENAME];

    EXECUTE_SYSTEM_COMMAND("mkdir -p %s", FFTCONFIGDIR);

#ifdef FFTWMT
    WRITE_FULLFILENAME(wisdom_file_single,
                       "%s/fftwf_mt_wisdom.dat",
                       FFTCONFIGDIR);
    WRITE_FULLFILENAME(wisdom_file_double,
                       "%s/fftw_mt_wisdom.dat",
                       FFTCONFIGDIR);
#endif

#ifndef FFTWMT
    WRITE_FULLFILENAME(wisdom_file_single, "%s/fftwf_wisdom.dat", FFTCONFIGDIR);
    WRITE_FULLFILENAME(wisdom_file_double, "%s/fftw_wisdom.dat", FFTCONFIGDIR);
#endif

    if((fp = fopen(wisdom_file_single, "w")) == NULL)
    {
        PRINT_ERROR("Error creating wisdom file \"%s\"", wisdom_file_single);
        return RETURN_FAILURE;
    }
#ifdef MILK_MODULE
    fftwf_export_wisdom_to_file(fp);
#endif
    fclose(fp);

    if((fp = fopen(wisdom_file_double, "w")) == NULL)
    {
        PRINT_ERROR("Error creating wisdom file \"%s\"", wisdom_file_double);
        return RETURN_FAILURE;
    }
#ifdef MILK_MODULE
    fftw_export_wisdom_to_file(fp);
#endif
    fclose(fp);

    return RETURN_SUCCESS;
}
