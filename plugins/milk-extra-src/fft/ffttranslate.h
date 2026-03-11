/**
 * @file ffttranslate.h
 * @brief Translate image using FFT
 */

errno_t CLIADDCMD_milkfft__ffttranslate();

int fft_image_translate(
    const char *ID_name,
    const char *ID_out,
    double xtransl,
    double ytransl);
