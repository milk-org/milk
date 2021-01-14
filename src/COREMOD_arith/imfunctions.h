/**
 * @file    imfunctions.c
 *
 *
 */



/* Functions for bison / flex    */ 

errno_t arith_image_function_im_im__d_d(
    const char *ID_name,
    const char *ID_out,
    double (*pt2function)(double)
);


errno_t arith_image_function_imd_im__dd_d(
    const char *ID_name,
    double      v0,
    const char *ID_out,
    double (*pt2function)(double, double)
);


errno_t arith_image_function_imdd_im__ddd_d(
    const char *ID_name,
    double      v0,
    double      v1,
    const char *ID_out,
    double (*pt2function)(double, double, double)
);






errno_t arith_image_function_1_1_byID(
    imageID ID,
    imageID IDout,
    double (*pt2function)(double)
);


errno_t arith_image_function_1_1(
    const char *ID_name,
    const char *ID_out,
    double (*pt2function)(double)
);



// imagein -> imagein (in place)
errno_t arith_image_function_1_1_inplace_byID(
    imageID ID,
    double (*pt2function)(double)
);


// imagein -> imagein (in place)
errno_t arith_image_function_1_1_inplace(
    const char *ID_name,
    double (*pt2function)(double)
);



errno_t arith_image_function_2_1(
    const char *ID_name1,
    const char *ID_name2,
    const char *ID_out,
    double (*pt2function)(double, double)
);

errno_t arith_image_function_2_1_inplace_byID(
    imageID ID1,
    imageID ID2,
    double (*pt2function)(double, double)
);

errno_t arith_image_function_2_1_inplace(
    const char *ID_name1,
    const char *ID_name2,
    double (*pt2function)(double, double)
);


errno_t arith_image_function_CF_CF__CF(
    const char *ID_name1,
    const char *ID_name2,
    const char *ID_out,
    complex_float(*pt2function)(complex_float, complex_float)
);


errno_t arith_image_function_CD_CD__CD(
    const char *ID_name1,
    const char *ID_name2,
    const char *ID_out,
    complex_double(*pt2function)(complex_double, complex_double)
);




int arith_image_function_1f_1(const char *ID_name, double f1,
                              const char *ID_out, double (*pt2function)(double, double));

int arith_image_function_1f_1_inplace_byID(long ID, double f1,
        double (*pt2function)(double, double));

int arith_image_function_1f_1_inplace(const char *ID_name, double f1,
                                      double (*pt2function)(double, double));



int arith_image_function_1ff_1(const char *ID_name, double f1, double f2,
                               const char *ID_out, double (*pt2function)(double, double, double));

int arith_image_function_1ff_1_inplace(const char *ID_name, double f1,
                                       double f2, double (*pt2function)(double, double, double));

int arith_image_function_1ff_1_inplace_byID(long ID, double f1, double f2,
        double (*pt2function)(double, double, double));
