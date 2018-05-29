#ifndef _IOFITS_H
#define _IOFITS_H



void __attribute__ ((constructor)) libinit_COREMOD_iofits();
int init_COREMOD_iofits();



int file_exists(const char * restrict file_name);

int is_fits_file(const char * restrict file_name);

int read_keyword(const char * restrict file_name, const char * restrict KEYWORD, char * restrict content);

int read_keyword_alone(const char * restrict file_name, const char * restrict KEYWORD);

int data_type_code(int bitpix);

long load_fits(const char * restrict file_name, const char * restrict ID_name, int errcode); 

int save_db_fits(const char * restrict ID_name, const char * restrict file_name);

int save_fl_fits(const char * restrict ID_name, const char * restrict file_name);

int save_sh_fits(const char * restrict ID_name, const char * restrict file_name);

int save_fits(const char * restrict ID_name, const char * restrict file_name);
int save_fits_atomic(const char * restrict ID_name, const char * restrict file_name);

int saveall_fits(const char * restrict savedirname);

int break_cube(const char * restrict ID_name);

int images_to_cube(const char * restrict img_name, long nbframes, const char * restrict cube_name);

#endif
