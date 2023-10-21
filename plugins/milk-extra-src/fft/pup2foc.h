/** @file pupfft.h
 */


#ifndef _MILK_FFT__PUP2FOC_H
#define _MILK_FFT__PUP2FOC_H


errno_t pup2foc_fft(
    const char * __restrict ID_name_ampl,
    const char * __restrict ID_name_pha,
    const char * __restrict ID_name_ampl_out,
    const char * __restrict ID_name_pha_out,
    const char * __restrict options
);


errno_t CLIADDCMD_milk_fft__pup2foc();

#endif
