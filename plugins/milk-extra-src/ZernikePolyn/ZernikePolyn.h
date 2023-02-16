#ifndef _ZERNIKEPOLYN_H
#define _ZERNIKEPOLYN_H


void __attribute__((constructor)) libinit_ZernikePolyn();


imageID mk_zer(const char *ID_name, long SIZE, long zer_nb, float rpix);

imageID
mk_zer_unbounded(const char *ID_name, long SIZE, long zer_nb, float rpix);

errno_t mk_zer_series(const char *ID_name, long SIZE, long zer_nb, float rpix);

imageID
mk_zer_seriescube(const char *ID_namec, long SIZE, long zer_nb, float rpix);

double get_zer(const char *ID_name, long zer_nb, double radius);

double
get_zer_crop(const char *ID_name, long zer_nb, double radius, double radius1);

int get_zerns(const char *ID_name, long max_zer, double radius);

int get_zern_array(const char *ID_name,
                   long        max_zer,
                   double      radius,
                   double     *array);

int remove_zerns(const char *ID_name,
                 const char *ID_name_out,
                 int         max_zer,
                 double      radius);

long ZERNIKEPOLYN_rmPiston(const char *ID_name, const char *IDmask_name);

int remove_TTF(const char *ID_name, const char *ID_name_out, double radius);

double fit_zer(const char *ID_name,
               long        maxzer_nb,
               double      radius,
               double     *zvalue,
               double     *residual);

#endif
