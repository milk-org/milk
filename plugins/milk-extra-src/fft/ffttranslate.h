/** @file ffttranslate.h
 */

errno_t ffttranslate_addCLIcmd();

int fft_image_translate(const char *ID_name,
                        const char *ID_out,
                        double      xtransl,
                        double      ytransl);
