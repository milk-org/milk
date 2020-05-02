/**
 * @file    savefits.h
 */



errno_t savefits_addCLIcmd();



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

