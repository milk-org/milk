/**
 * @file imexpand.h
 * @brief Imexpand module
 */

/** @file imexpand.h
 */

errno_t CLIADDCMD_image_basic__imexpand();

imageID basic_expand(const char *ID_name, const char *ID_name_out, int n1, int n2);

imageID basic_expand3D(const char *ID_name, const char *ID_name_out, int n1, int n2, int n3);
