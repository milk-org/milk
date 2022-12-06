%{
#include <math.h>  /* For math functions, cos(), sin(), etc. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
  //#include "calc.h"  /* Contains definition of `symrec'        */

#include "CLIcore.h"
#include "COREMOD_memory/COREMOD_memory.h"
#include "COREMOD_iofits/COREMOD_iofits.h"
#include "COREMOD_arith/COREMOD_arith.h"


int yylex();
int yyerror(char *);


//extern DATA data;


// NOTATIONS
// d: double
// im: image

 char calctmpimname[200];

  %}

%union {
  long     val_l;  /* long */
  float    val_f;  /* float */
  double   val_d;  /* For returning numbers.     */
  char  string[200];   /* For returning strings (variables, images)  */
  double (*fnctptr)();    /* pointer to function -> double */
}

%token <val_l> TKNUMl /* long */
%token <val_f> TKNUMf /* single precision number */
%token <val_d> TKNUMd /* double precision number */
%token <string> TKVAR /* existing variable name */
%token <string> TKNVAR /* new valiable name */
%token <string> TKIMAGE /* image */
%token <string> TKCOMMAND /* command */
%token <fnctptr> TKFUNC_d_d /* function double -> double */
%token <fnctptr> TKFUNC_dd_d /* function double double -> double */
%token <fnctptr> TKFUNC_ddd_d /* function double double double -> double */
%token <fnctptr> TKFUNC_im_d /* function image -> double */
%token <fnctptr> TKFUNC_imd_d /* function image double -> double */

%type <val_l> expl
%type <val_d> expd
%type <string> exps

%right '='
%left '-' '+'
%left '*' '/'
%left NEG     /* Negation--unary minus */
%right '^'    /* Exponentiation        */

/* Grammar follows */

%%

input:   /* empty */
        | input line
;

line:
'\n'
| expd '\n'  {
printf("\t double: %.10g\n", $1);
data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_FLOAT;
data.cmdargtoken[data.cmdNBarg].val.numf = $1;
}
| expl '\n'  {
printf("\t long:   %ld\n", $1);
data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_LONG;
data.cmdargtoken[data.cmdNBarg].val.numl = $1;
}
| exps '\n' { if(data.Debug>0) {printf("\t string: %s\n", $1);}
    //data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_STRING;
snprintf(data.cmdargtoken[data.cmdNBarg].val.string, STRINGMAXLEN_CMDARGTOKEN_VAL, "%s", $1);
}
| error '\n' { yyerrok;                  }
;

expl:     TKNUMl       { $$ = $1;        if(data.Debug>0){printf("this is a long\n");}}
| expl '+' expl        { $$ = $1 + $3;   if(data.Debug>0){printf("long + long\n");}}
| expl '-' expl        { $$ = $1 - $3;   if(data.Debug>0){printf("long - long\n");}}
| expl '*' expl        { $$ = $1 * $3;   if(data.Debug>0){printf("long * long\n");}}
| '-' expl  %prec NEG  { $$ = -$2;       if(data.Debug>0){printf("-long\n");}}
| expl '^' expl        { $$ = (long) pow ($1,$3);  if(data.Debug>0){printf("long ^ long\n");}}
| '(' expl ')'         { $$ = $2; }
;

expd:      TKNUMd      { $$ = $1;        if(data.Debug>0){printf("this is a double\n");}}
| TKVAR                { $$ = data.variable[variable_ID($1)].value.f;   }
| TKVAR '=' expl       { $$ = $3; create_variable_ID($1, $3);  }
| TKVAR '=' expd       { $$ = $3; create_variable_ID($1, $3);  }
| TKNVAR '=' expl      { $$ = $3; create_variable_ID($1, $3);  if(data.Debug>0){printf("creating long\n");}}
| TKNVAR '=' expd      { $$ = $3; create_variable_ID($1, $3);  if(data.Debug>0){printf("creating double\n");}}
| expl '+' expd        { $$ = $1 + $3;   if(data.Debug>0){printf("long + double\n");}}
| expd '+' expl        { $$ = $1 + $3;   if(data.Debug>0){printf("double + long\n");}}
| expd '+' expd        { $$ = $1 + $3;   if(data.Debug>0){printf("double + double\n");}}
| expl '-' expd        { $$ = $1 - $3;   if(data.Debug>0){printf("long - double\n");}}
| expd '-' expl        { $$ = $1 - $3;   if(data.Debug>0){printf("double - long\n");}}
| expd '-' expd        { $$ = $1 - $3;   if(data.Debug>0){printf("double - double\n");}}
| expl '*' expd        { $$ = (double) $1 * $3;   if(data.Debug>0){printf("long * double\n");}}
| expd '*' expl        { $$ = $1 * $3;   if(data.Debug>0){printf("double * long\n");}}
| expd '*' expd        { $$ = $1 * $3;   if(data.Debug>0){printf("double * double\n");}}
| expl '/' expl        { $$ = (double) $1 / $3;   if(data.Debug>0){printf("long / long\n");}}
| expl '/' expd        { $$ = (double) $1 / $3;   if(data.Debug>0){printf("long / double\n");}}
| expd '/' expl        { $$ = $1 / $3;   if(data.Debug>0){printf("double / long\n");}}
| expd '/' expd        { $$ = $1 / $3;   if(data.Debug>0){printf("double / double\n");}}
| '-' expd  %prec NEG  { $$ = -$2;       if(data.Debug>0){printf("-double\n");}}
| expl '^' expd        { $$ = pow ((double) $1,$3);  if(data.Debug>0){printf("long ^ double\n");}}
| expd '^' expl        { $$ = pow ($1,(double) $3);  if(data.Debug>0){printf("double ^ long\n");}}
| expd '^' expd        { $$ = pow ($1,$3);  if(data.Debug>0){printf("double ^ double\n");}}
| TKFUNC_d_d expd ')'  { $$ = $1($2);  if(data.Debug>0){printf("double=func(double)\n");}}
| TKFUNC_d_d expl ')'  { $$ = $1((double) $2);  if(data.Debug>0){printf("double=func(double)\n");}}
| TKFUNC_dd_d expd ',' expd ')'  { $$ = $1($2,$4);  if(data.Debug>0){printf("double=func(double,double)\n");}}
| TKFUNC_dd_d expl ',' expd ')'  { $$ = $1((double) $2,$4);  if(data.Debug>0){printf("double=func(long->double,double)\n");}}
| TKFUNC_dd_d expd ',' expl ')'  { $$ = $1($2,(double) $4);  if(data.Debug>0){printf("double=func(double,long->double)\n");}}
| TKFUNC_dd_d expl ',' expl ')'  { $$ = $1((double) $2,(double) $4);  if(data.Debug>0){printf("double=func(long->double,long->double)\n");}}
| TKFUNC_ddd_d expd ',' expd ',' expd ')'  { $$ = $1($2,$4,$6);  if(data.Debug>0){printf("double=func(double,double,double)\n");}}
| TKFUNC_ddd_d expl ',' expd ',' expd ')'  { $$ = $1((double) $2,$4,$6);  if(data.Debug>0){printf("double=func(long->double,double,double)\n");}}
| TKFUNC_ddd_d expd ',' expl ',' expd ')'  { $$ = $1($2,(double) $4,$6);  if(data.Debug>0){printf("double=func(double,long->double,double)\n");}}
| TKFUNC_ddd_d expl ',' expl ',' expd ')'  { $$ = $1((double) $2,(double) $4,$6);  if(data.Debug>0){printf("double=func(long->double,long->double,double)\n");}}
| TKFUNC_ddd_d expd ',' expd ',' expl ')'  { $$ = $1($2,$4,(double) $6);  if(data.Debug>0){printf("double=func(double,double,long->double)\n");}}
| TKFUNC_ddd_d expl ',' expd ',' expl ')'  { $$ = $1((double) $2,$4,(double) $6);  if(data.Debug>0){printf("double=func(long->double,double,long->double)\n");}}
| TKFUNC_ddd_d expd ',' expl ',' expl ')'  { $$ = $1($2,(double) $4,(double) $6);  if(data.Debug>0){printf("double=func(double,long->double,long->double)\n");}}
| TKFUNC_ddd_d expl ',' expl ',' expl ')'  { $$ = $1((double) $2,(double) $4,(double) $6);  if(data.Debug>0){printf("double=func(long->double,long->double,long->double)\n");}}
| TKFUNC_im_d exps ')'  { $$ = $1($2);  if(data.Debug>0){printf("double=func(image)\n");}}
| TKFUNC_imd_d exps ',' expd ')'  { $$ = $1($2,$4);  if(data.Debug>0){printf("double=func(image,double)\n");}}
| '(' expd ')'         { $$ = $2;                         }
;

//$$ = strdup($1)

exps:    TKNVAR         {strcpy($$, $1);        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_STRING; if(data.Debug>0){printf("this is a string (new variable/image)\n");}}
| TKIMAGE               {strcpy($$, $1);        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_EXISTINGIMAGE; if(data.Debug>0){printf("this is a string (existing image)\n");}}
| TKCOMMAND             {strcpy($$, $1);        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_COMMAND; if(data.Debug>0){printf("this is a string (command)\n");}}
| TKIMAGE '=' exps    {strcpy($$, $1);        delete_image_ID($1,1); chname_image_ID($3,$1); if(data.Debug>0){printf("changing name\n");}}
| TKNVAR '=' exps    {strcpy($$, $1);        chname_image_ID($3,$1); if(data.Debug>0){printf("changing name\n");}}
| exps '+' exps      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_add($1, $3, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("image + image\n");}}
| exps '+' expd      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstadd($1,(double) $3, calctmpimname); strcpy($$, calctmpimname);  if(data.Debug>0){printf("image + double\n");}}
| exps '+' expl      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstadd($1,(double) $3, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("image + long\n");}}
| expd '+' exps      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstadd($3,(double) $1, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("double + image\n");}}
| expl '+' exps      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstadd($3,(double) $1, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("long + image\n");}}
| exps '-' exps      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_sub($1, $3, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("image + image\n");}}
| exps '-' expd      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstadd($1,(double) -$3, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("image - double\n");}}
| exps '-' expl      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstadd($1,(double) -$3, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("image - long\n");}}
| expd '-' exps      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstsubm($3,(double) $1, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("double - image\n");}}
| expl '-' exps      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstsubm($3,(double) $1, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("long - image\n");}}
| exps '*' exps      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_mult($1, $3, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("image * image\n");}}
| exps '*' expd      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstmult($1,(double) $3, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("image * double\n");}}
| exps '*' expl      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstmult($1,(double) $3, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("image * long\n");}}
| expd '*' exps      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstmult($3,(double) $1, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("double * image\n");}}
| expl '*' exps      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstmult($3,(double) $1, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("long * image\n");}}
| exps '/' exps      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_div($1, $3, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("image / image\n");}}
| exps '/' expd      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstdiv($1,(double) $3, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("image - double\n");}}
| exps '/' expl      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstdiv($1,(double) $3, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("image - long\n");}}
| expd '/' exps      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstdiv($3,(double) $1, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("double - image\n");}}
| expl '/' exps      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstdiv($3,(double) $1, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("long - image\n");}}
| exps '^' expl      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstpow($1,(double) $3, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("image^long\n");}}
| exps '^' expd      {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_cstpow($1,(double) $3, calctmpimname); strcpy($$, calctmpimname); if(data.Debug>0){printf("image^double\n");}}
| TKFUNC_d_d exps ')'           {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_function_im_im__d_d($2, calctmpimname, $1); strcpy($$, calctmpimname); if(data.Debug>0){printf("double_func(double)\n");}}
| TKFUNC_dd_d exps ',' expd ')' {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_function_imd_im__dd_d($2, (double) $4, calctmpimname, $1); strcpy($$, calctmpimname); if(data.Debug>0){printf("double_func(double, double)\n");}}
| TKFUNC_ddd_d exps ',' expd ',' expd ')' {snprintf(calctmpimname,200,"_tmpcalc%ld",data.calctmp_imindex); data.calctmp_imindex++; arith_image_function_imdd_im__ddd_d($2, (double) $4, (double) $6, calctmpimname, $1); strcpy($$, calctmpimname); if(data.Debug>0){printf("double_func(double, double, double)\n");}}
| '(' exps ')'         { strcpy($$, $2);                         }
;


/* End of grammar */
%%


#include <stdio.h>

int yylex();

int yyerror (s)  /* Called by yyparse on error */
     char *s;
{
  printf ("PARSING ERROR ON COMMAND LINE ARG %ld: %s\n", data.cmdNBarg, s);
  data.parseerror = 1;
	return 0;
}
