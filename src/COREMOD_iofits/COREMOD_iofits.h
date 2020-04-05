#ifndef _IOFITS_H
#define _IOFITS_H



void __attribute__((constructor)) libinit_COREMOD_iofits();



int file_exists(const char *restrict file_name);

int is_fits_file(const char *restrict file_name);

int read_keyword(
    const char *restrict file_name,
    const char *restrict KEYWORD,
    char *restrict content
);

errno_t read_keyword_alone(
    const char *restrict file_name,
    const char *restrict KEYWORD
);


int data_type_code(int bitpix);


imageID load_fits(
    const char *restrict file_name,
    const char *restrict ID_name,
    int         errcode
);


errno_t save_db_fits(
    const char *restrict ID_name,
    const char *restrict file_name
);


errno_t save_fl_fits(
    const char *restrict ID_name,
    const char *restrict file_name
);


errno_t save_sh16_fits(
    const char *restrict ID_name,
    const char *restrict file_name
);


errno_t save_ush16_fits(
    const char *restrict ID_name,
    const char *restrict file_name
);


errno_t save_int32_fits(
    const char *restrict ID_name,
    const char *restrict file_name
);


errno_t save_uint32_fits(
    const char *restrict ID_name,
    const char *restrict file_name
);


errno_t save_int64_fits(
    const char *restrict ID_name,
    const char *restrict file_name
);




errno_t save_fits(
    const char *restrict ID_name,
    const char *restrict file_name
);


errno_t save_fits_atomic(
    const char *restrict ID_name,
    const char *restrict file_name
);


errno_t saveall_fits(
    const char *restrict savedirname
);


imageID break_cube(
    const char *restrict ID_name
);

errno_t images_to_cube(
    const char *restrict img_name,
    long                 nbframes,
    const char *restrict cube_name
);


imageID COREMOD_IOFITS_LoadMemStream(
    const char *sname,
    uint64_t   *streamflag,
    uint32_t   *imLOC
);



#endif
