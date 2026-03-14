/**
 * @file image_add.h
 * @brief Image add module
 */

/** @file image_add.h
 */

errno_t CLIADDCMD_image_basic__image_add();

imageID basic_add(const char *__restrict ID_name1,
                  const char *__restrict ID_name2,
                  const char *__restrict ID_name_out,
                  long off1,
                  long off2);

imageID basic_add3D(const char *__restrict ID_name1,
                    const char *__restrict ID_name2,
                    const char *__restrict ID_name_out,
                    long off1,
                    long off2,
                    long off3);
