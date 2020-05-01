/**
 * @file    list_image.h
 */



errno_t    memory_monitor(
    const char *termttyname
);



errno_t init_list_image_ID_ncurses(
    const char *termttyname
);

void close_list_image_ID_ncurses();

errno_t list_image_ID_ncurses();

errno_t list_image_ID_ofp(
    FILE *fo
);

errno_t list_image_ID_ofp_simple(
    FILE *fo
);

errno_t list_image_ID();

errno_t list_image_ID_file(
    const char *fname
);

errno_t list_variable_ID();

errno_t list_variable_ID_file(
    const char *fname
);
