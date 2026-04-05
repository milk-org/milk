/**
 * @file    datatype_dispatch.h
 * @brief   X-macro for numeric datatype dispatch
 *
 * Eliminates duplicated 10-branch else-if chains
 * across the generic function-pointer dispatch
 * functions in imfunctions.c.
 *
 * Private header — only included from imfunctions.c.
 */

#ifndef DATATYPE_DISPATCH_H
#define DATATYPE_DISPATCH_H

/**
 * @brief Dispatch a per-pixel body over all types
 *
 * Expands BODY(member) for each numeric datatype.
 * The caller defines BODY as a macro that uses
 * 'member' to index the IMAGE union array:
 *   im->array.member[ii]
 *
 * The DOUBLE branch is handled separately by
 * BODY_D to allow different output types
 * (float output for non-double, double for double).
 *
 * @param dt      uint8_t datatype value
 * @param BODY    Macro(member) for non-double types
 * @param BODY_D  Macro for DOUBLE type
 */
#define FOR_EACH_DATATYPE(dt, BODY, BODY_D)  \
    if      (dt == _DATATYPE_UINT8)          \
    {                                        \
        BODY(UI8)                            \
    }                                        \
    else if (dt == _DATATYPE_INT8)           \
    {                                        \
        BODY(SI8)                            \
    }                                        \
    else if (dt == _DATATYPE_UINT16)         \
    {                                        \
        BODY(UI16)                           \
    }                                        \
    else if (dt == _DATATYPE_INT16)          \
    {                                        \
        BODY(SI16)                           \
    }                                        \
    else if (dt == _DATATYPE_UINT32)         \
    {                                        \
        BODY(UI32)                           \
    }                                        \
    else if (dt == _DATATYPE_INT32)          \
    {                                        \
        BODY(SI32)                           \
    }                                        \
    else if (dt == _DATATYPE_UINT64)         \
    {                                        \
        BODY(UI64)                           \
    }                                        \
    else if (dt == _DATATYPE_INT64)          \
    {                                        \
        BODY(SI64)                           \
    }                                        \
    else if (dt == _DATATYPE_FLOAT)          \
    {                                        \
        BODY(F)                              \
    }                                        \
    else if (dt == _DATATYPE_DOUBLE)         \
    {                                        \
        BODY_D                               \
    }

#endif /* DATATYPE_DISPATCH_H */
