/**
 * @file images2cube.h
 * @brief Combine individual images into cube
 */

errno_t CLIADDCMD_COREMOD_iofits__images2cube();

errno_t images_to_cube(
    const char *restrict img_name,
    uint32_t nbframes,
    const char *restrict cube_name);
