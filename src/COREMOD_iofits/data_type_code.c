/**
 * @file    data_type_code.c
 */

int data_type_code(int bitpix)
{
    int code;
    /*
      bitpix      Datatype             typecode    Mnemonic
      1           bit, X                   1        TBIT
      8           byte, B                 11        TBYTE
                  logical, L              14        TLOGICAL
                  ASCII character, A      16        TSTRING
      16          short integer, I        21        TSHORT
      32          integer, J                      41        TLONG
      64          64-bit long signed integer 'K'  81        TLONGLONG
     -32          real, E                 42        TFLOAT
     -64          double precision, D     82        TDOUBLE
                  complex, C              83        TCOMPLEX
                  double complex, M      163        TDBLCOMPLEX
                  */
    code = 0;
    if(bitpix == 1)
    {
        code = 1;
    }
    if(bitpix == 8)
    {
        code = 11;
    }
    if(bitpix == 16)
    {
        code = 21;
    }
    if(bitpix == 32)
    {
        code = 41;
    }
    if(bitpix == 64)
    {
        code = 81;
    }
    if(bitpix == -32)
    {
        code = 42;
    }
    if(bitpix == -64)
    {
        code = 82;
    }
    return (code);
}
