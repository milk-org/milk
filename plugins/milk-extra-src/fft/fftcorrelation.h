/**
 * @file fftcorrelation.h
 * @brief Correlate two images using FFT
 */

errno_t CLIADDCMD_milkfft__fftcorrelation();

imageID fft_correlation(
    const char *ID_name1,
    const char *ID_name2,
    const char *ID_nameout);
