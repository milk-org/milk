/**
 * @file imgetcircsym.h
 * @brief Imgetcircsym module
 */

/** @file imgetcircsym.h
 */

errno_t __attribute__((cold)) CLIADDCMD_image_basic__imgetcircsym();

imageID IMAGE_BASIC_get_circsym_component(const char *__restrict ID_name,
                                          const char *__restrict ID_out_name,
                                          float xcenter,
                                          float ycenter);
