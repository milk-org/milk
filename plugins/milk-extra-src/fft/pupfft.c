/** @file pupfft.c
 */

#include "CommandLineInterface/CLIcore.h"

#include "COREMOD_memory/COREMOD_memory.h"

#include "dofft.h"
#include "permut.h"

/* inv = 0 for direct fft and 1 for inverse fft */
/* direct = focal plane -> pupil plane  equ. fft2d(..,..,..,1) */
/* inverse = pupil plane -> focal plane equ. fft2d(..,..,..,0) */
/* options :  -reim  takes real/imaginary input and creates real/imaginary output
               -inv  for inverse fft (inv=1) */
errno_t pupfft(const char *ID_name_ampl,
               const char *ID_name_pha,
               const char *ID_name_ampl_out,
               const char *ID_name_pha_out,
               const char *options)
{
    int reim;
    int inv;

    char Ctmpname[STRINGMAXLEN_IMGNAME];
    char C1tmpname[STRINGMAXLEN_IMGNAME];

    reim = 0;
    inv  = 0;

    if(strstr(options, "-reim") != NULL)
    {
        /*	printf("taking real / imaginary input/output\n");*/
        reim = 1;
    }

    if(strstr(options, "-inv") != NULL)
    {
        /*printf("doing the inverse Fourier transform\n");*/
        inv = 1;
    }

    WRITE_IMAGENAME(Ctmpname, "_Ctmp_%d", (int) getpid());

    if(reim == 0)
    {
        mk_complex_from_amph(ID_name_ampl, ID_name_pha, Ctmpname, 0);
    }
    else
    {
        mk_complex_from_reim(ID_name_ampl, ID_name_pha, Ctmpname, 0);
    }

    permut(Ctmpname);

    WRITE_IMAGENAME(C1tmpname, "_C1tmp_%d", (int) getpid());

    if(inv == 0)
    {
        do2dfft(Ctmpname, C1tmpname); /* equ. fft2d(..,1) */
    }
    else
    {
        do2dffti(Ctmpname, C1tmpname); /* equ. fft2d(..,0) */
    }

    delete_image_ID(Ctmpname, DELETE_IMAGE_ERRMODE_WARNING);

    if(reim == 0)
    {
        /* if this line is removed, the program crashes... why ??? */
        /*	list_image_ID(data); */
        mk_amph_from_complex(C1tmpname, ID_name_ampl_out, ID_name_pha_out, 0);
    }
    else
    {
        mk_reim_from_complex(C1tmpname, ID_name_ampl_out, ID_name_pha_out, 0);
    }

    delete_image_ID(C1tmpname, DELETE_IMAGE_ERRMODE_WARNING);

    permut(ID_name_ampl_out);
    permut(ID_name_pha_out);

    return RETURN_SUCCESS;
}
