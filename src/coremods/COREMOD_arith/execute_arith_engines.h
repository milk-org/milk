/**
 * @file    execute_arith_engines.h
 * @brief   Execution engines for arithmetic parser
 */

#ifndef EXECUTE_ARITH_ENGINES_H
#define EXECUTE_ARITH_ENGINES_H

#define ARITHTOKENTYPE_UNKNOWN 0
#define ARITHTOKENTYPE_NOTEXIST 1 // non-existing variable or image
#define ARITHTOKENTYPE_VARIABLE 2
#define ARITHTOKENTYPE_NUMBER 3
#define ARITHTOKENTYPE_OPERAND 4
#define ARITHTOKENTYPE_OPENPAR 5
#define ARITHTOKENTYPE_CLOSEPAR 6
#define ARITHTOKENTYPE_COMA 7
#define ARITHTOKENTYPE_FUNCTION 8
#define ARITHTOKENTYPE_EQUAL 9
#define ARITHTOKENTYPE_IMAGE 10
#define ARITHTOKENTYPE_MULTFUNC \
    11 // function of several variables/images, returning one variable/image

int exec_arith_binary(const char *op,
                      int         lt,
                      const char *lw,
                      int         rt,
                      const char *rw,
                      char       *name,
                      int        *type,
                      int        *tmp_name_index);

int exec_arith_unary(const char *fname,
                     int         arg_wtype,
                     const char *arg_word,
                     char       *name,
                     int        *type,
                     int        *tmp_name_index);

int exec_arith_multfunc(const char *fn,
                        int         nbvarinput,
                        int         a1t,
                        const char *a1w,
                        int         a2t,
                        const char *a2w,
                        int         a3t,
                        const char *a3w,
                        char       *name,
                        int        *type,
                        int        *tmp_name_index);

#endif // EXECUTE_ARITH_ENGINES_H
