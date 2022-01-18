/**
 * @file    mathfuncs.c
 * @brief   simple math functions
 *
 *
 */

#include "CommandLineInterface/CLIcore.h" // complex types
#include <math.h>

double Pacos(double a) { return ((double)acos(a)); }
double Pasin(double a) { return ((double)asin(a)); }
double Patan(double a) { return ((double)atan(a)); }
double Pceil(double a) { return ((double)ceil(a)); }
double Pcos(double a) { return ((double)cos(a)); }
double Pcosh(double a) { return ((double)cosh(a)); }
double Pexp(double a) { return ((double)exp(a)); }
double Pfabs(double a) { return ((double)fabs(a)); }
double Pfloor(double a) { return ((double)floor(a)); }
double Pln(double a) { return ((double)log(a)); }
double Plog(double a) { return ((double)log10(a)); }
double Psqrt(double a) { return ((double)sqrt(a)); }
double Psin(double a) { return ((double)sin(a)); }
double Psinh(double a) { return ((double)sinh(a)); }
double Ptan(double a) { return ((double)tan(a)); }
double Ptanh(double a) { return ((double)tanh(a)); }

double Ppositive(double a)
{
    double value = 0.0;
    if (a > 0.0)
    {
        value = (double)1.0;
    }
    return (value);
}

double Pfmod(double a, double b) { return ((double)fmod(a, b)); }

double Ppow(double a, double b)
{
    if (b > 0)
    {
        return ((double)pow(a, b));
    }
    else
    {
        return ((double)pow(a, -b));
    }
}

double Padd(double a, double b) { return ((double)a + b); }

double Psubm(double a, double b) { return ((double)b - a); }

double Psub(double a, double b) { return ((double)a - b); }

double Pmult(double a, double b) { return ((double)a * b); }

double Pdiv(double a, double b) { return ((double)a / b); }

double Pdiv1(double a, double b) { return ((double)b / a); }

double Pminv(double a, double b)
{
    if (a < b)
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
    if (a > b)
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
    if (a < b)
    {
        return ((double)1.0);
    }
    else
    {
        return ((double)0.0);
    }
}

double Ptestmt(double a, double b)
{
    if (a < b)
    {
        return ((double)0.0);
    }
    else
    {
        return ((double)1.0);
    }
}

double Ptrunc(double a, double b, double c)
{
    double value;
    value = a;
    if (a < b)
    {
        value = b;
    };
    if (a > c)
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
    double amp, pha;
    double are, aim, bre, bim;

    are = a.re;
    aim = a.im;
    bre = b.re;
    bim = b.im;

    amp = sqrt(are * are + aim * aim);
    amp /= sqrt(bre * bre + bim * bim);
    pha = atan2(aim, are);
    pha -= atan2(bim, bre);

    v.re = (double)(amp * cos(pha));
    v.im = (double)(amp * sin(pha));

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
    float amp, pha;
    float are, aim, bre, bim;

    are = a.re;
    aim = a.im;
    bre = b.re;
    bim = b.im;

    amp = sqrt(are * are + aim * aim);
    amp /= sqrt(bre * bre + bim * bim);
    pha = atan2(aim, are);
    pha -= atan2(bim, bre);

    v.re = (float)(amp * cos(pha));
    v.im = (float)(amp * sin(pha));

    return (v);
}
