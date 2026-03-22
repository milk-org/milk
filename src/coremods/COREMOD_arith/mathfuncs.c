/**
 * @file    mathfuncs.c
 * @brief   simple math functions
 *
 *
 */

#ifdef MILK_NO_CLI
#include "CLIcore_standalone.h"
#else
#include "libmilkdata/milkdata.h"
#endif // complex types
#include <math.h>

double Pacos(double a)
{
    return ((double) acos(a));
}
double Pasin(double a)
{
    return ((double) asin(a));
}
double Patan(double a)
{
    return ((double) atan(a));
}
double Pceil(double a)
{
    return ((double) ceil(a));
}
double Pcos(double a)
{
    return ((double) cos(a));
}
double Pcosh(double a)
{
    return ((double) cosh(a));
}
double Pexp(double a)
{
    return ((double) exp(a));
}
double Pfabs(double a)
{
    return ((double) fabs(a));
}
double Pfloor(double a)
{
    return ((double) floor(a));
}
double Pln(double a)
{
    return ((double) log(a));
}
double Plog(double a)
{
    return ((double) log10(a));
}
double Psqrt(double a)
{
    return ((double) sqrt(a));
}
double Psin(double a)
{
    return ((double) sin(a));
}
double Psinh(double a)
{
    return ((double) sinh(a));
}
double Ptan(double a)
{
    return ((double) tan(a));
}
double Ptanh(double a)
{
    return ((double) tanh(a));
}

double Ppositive(double a)
{
    double value = 0.0;
    if(a > 0.0)
    {
        value = (double) 1.0;
    }
    return (value);
}

double Pfmod(double a, double b)
{
    return ((double) fmod(a, b));
}

double Ppow(double a, double b)
{
    return ((double) pow(a, b));
}

double Padd(double a, double b)
{
    return ((double) a + b);
}

double Psubm(double a, double b)
{
    return ((double) b - a);
}

double Psub(double a, double b)
{
    return ((double) a - b);
}

double Pmult(double a, double b)
{
    return ((double) a * b);
}

double Pdiv(double a, double b)
{
    return ((double) a / b);
}

double Pdiv1(double a, double b)
{
    return ((double) b / a);
}

double Pminv(double a, double b)
{
    if(a < b)
    {
        return (a);
    }
    else
    {
        return (b);
    }
}

double Pmaxv(double a, double b)
{
    if(a > b)
    {
        return (a);
    }
    else
    {
        return (b);
    }
}

double Ptestlt(double a, double b)
{
    if(a < b)
    {
        return ((double) 1.0);
    }
    else
    {
        return ((double) 0.0);
    }
}

double Ptestmt(double a, double b)
{
    if(a < b)
    {
        return ((double) 0.0);
    }
    else
    {
        return ((double) 1.0);
    }
}

double Pteste(double a, double b)
{
    if(a == b)
    {
        return ((double) 1.0);
    }
    else
    {
        return ((double) 0.0);
    }
}

double Ptestne(double a, double b)
{
    if(a != b)
    {
        return ((double) 1.0);
    }
    else
    {
        return ((double) 0.0);
    }
}

double Ptestle(double a, double b)
{
    if(a <= b)
    {
        return ((double) 1.0);
    }
    else
    {
        return ((double) 0.0);
    }
}

double Ptestge(double a, double b)
{
    if(a >= b)
    {
        return ((double) 1.0);
    }
    else
    {
        return ((double) 0.0);
    }
}

double Pand(double a, double b)
{
    if((a != 0.0) && (b != 0.0))
    {
        return ((double) 1.0);
    }
    else
    {
        return ((double) 0.0);
    }
}

double Por(double a, double b)
{
    if((a != 0.0) || (b != 0.0))
    {
        return ((double) 1.0);
    }
    else
    {
        return ((double) 0.0);
    }
}

double Ptrunc(double a, double b, double c)
{
    double value;
    value = a;
    if(a < b)
    {
        value = b;
    };
    if(a > c)
    {
        value = c;
    };
    return (value);
}

complex_double CPadd_CD_CD(complex_double a, complex_double b)
{
    complex_double v;
    v.re = a.re + b.re;
    v.im = a.im + b.im;
    return (v);
}

complex_double CPsub_CD_CD(complex_double a, complex_double b)
{
    complex_double v;
    v.re = a.re - b.re;
    v.im = a.im - b.im;
    return (v);
}

complex_double CPmult_CD_CD(complex_double a, complex_double b)
{
    complex_double v;
    v.re = a.re * b.re - a.im * b.im;
    v.im = a.re * b.im + a.im * b.re;
    return (v);
}

complex_double CPdiv_CD_CD(complex_double a, complex_double b)
{
    complex_double v;
    double         den;

    den = b.re * b.re + b.im * b.im;
    
    v.re = (a.re * b.re + a.im * b.im) / den;
    v.im = (a.im * b.re - a.re * b.im) / den;

    return (v);
}

complex_float CPadd_CF_CF(complex_float a, complex_float b)
{
    complex_float v;
    v.re = a.re + b.re;
    v.im = a.im + b.im;
    return (v);
}

complex_float CPsub_CF_CF(complex_float a, complex_float b)
{
    complex_float v;
    v.re = a.re - b.re;
    v.im = a.im - b.im;
    return (v);
}

complex_float CPmult_CF_CF(complex_float a, complex_float b)
{
    complex_float v;
    v.re = a.re * b.re - a.im * b.im;
    v.im = a.re * b.im + a.im * b.re;
    return (v);
}

complex_float CPdiv_CF_CF(complex_float a, complex_float b)
{
    complex_float v;
    float         den;

    den = b.re * b.re + b.im * b.im;

    v.re = (a.re * b.re + a.im * b.im) / den;
    v.im = (a.im * b.re - a.re * b.im) / den;

    return (v);
}
