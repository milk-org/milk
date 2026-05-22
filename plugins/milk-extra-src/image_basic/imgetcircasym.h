/**
 * @file imgetcircasym.h
 * @brief Imgetcircasym module
 */

/** @file imgetcircasym.h
 */

errno_t __attribute__((cold)) CLIADDCMD_image_basic__imgetcircasym();

imageID IMAGE_BASIC_get_circasym_component_byID(imageID ID,
                                                const char *__restrict ID_out_name,
                                                float       xcenter,
                                                float       ycenter,
                                                const char *options);

imageID IMAGE_BASIC_get_circasym_component(const char *__restrict ID_name,
                                           const char *__restrict ID_out_name,
                                           float       xcenter,
                                           float       ycenter,
                                           const char *options);
