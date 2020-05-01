/**
 * @file    image_complex.h
 */


errno_t mk_complex_from_reim(
    const char *re_name,
    const char *im_name,
    const char *out_name,
    int         sharedmem
);

errno_t mk_complex_from_amph(
    const char *am_name,
    const char *ph_name,
    const char *out_name,
    int         sharedmem
);

errno_t mk_reim_from_complex(
    const char *in_name,
    const char *re_name,
    const char *im_name,
    int         sharedmem
);

errno_t mk_amph_from_complex(
    const char *in_name,
    const char *am_name,
    const char *ph_name,
    int         sharedmem
);

errno_t mk_reim_from_amph(
    const char *am_name,
    const char *ph_name,
    const char *re_out_name,
    const char *im_out_name,
    int         sharedmem
);

errno_t mk_amph_from_reim(
    const char *re_name,
    const char *im_name,
    const char *am_out_name,
    const char *ph_out_name,
    int         sharedmem
);
