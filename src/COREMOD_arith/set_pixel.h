/**
 * @file    set_pixel.h
 *
 */



errno_t set_pixel_addCLIcmd();


imageID arith_set_pixel(
    const char *ID_name,
    double      value,
    long        x,
    long        y
);



imageID arith_set_pixel_1Drange(
    const char *ID_name,
    double      value,
    long        x,
    long        y
);



imageID arith_set_row(
    const char *ID_name,
    double      value,
    long        y
);



imageID arith_set_col(
    const char *ID_name,
    double      value,
    long        x
);



imageID arith_image_zero(
    const char *ID_name
);
