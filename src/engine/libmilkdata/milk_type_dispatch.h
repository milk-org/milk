/**
 * @file    milk_type_dispatch.h
 * @brief   X-macros for numeric datatype dispatch
 *
 * Eliminates duplicated 10-branch else-if chains
 * across generic function-pointer dispatch functions.
 */

#ifndef MILK_TYPE_DISPATCH_H
#define MILK_TYPE_DISPATCH_H

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
#define MILK_FOR_EACH_DATATYPE(dt, BODY, BODY_D)  \
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

/**
 * @brief Dispatch a per-pixel body over complex types
 *
 * Expands BODY(member) for each complex datatype.
 *
 * @param dt      uint8_t datatype value
 * @param BODY    Macro(member) for complex types
 */
#define MILK_FOR_EACH_COMPLEX_TYPE(dt, BODY)  \
    if      (dt == _DATATYPE_COMPLEX_FLOAT)   \
    {                                         \
        BODY(CF)                              \
    }                                         \
    else if (dt == _DATATYPE_COMPLEX_DOUBLE)  \
    {                                         \
        BODY(CD)                              \
    }

/**
 * @brief Dispatch an FPS stream test over all datatypes
 *
 * Expands BODY(type_name) for each numeric datatype, where
 * type_name is the uppercase string name of the datatype (e.g. UINT8).
 *
 * @param dt_flag  The parameter's fpflag field
 * @param BODY     Macro(dt_flag, type_name) that gets expanded
 */
#define MILK_FOR_EACH_FPS_STREAM_TYPE(dt_flag, BODY) \
    BODY(dt_flag, UINT8)   \
    BODY(dt_flag, INT8)    \
    BODY(dt_flag, UINT16)  \
    BODY(dt_flag, INT16)   \
    BODY(dt_flag, UINT32)  \
    BODY(dt_flag, INT32)   \
    BODY(dt_flag, UINT64)  \
    BODY(dt_flag, INT64)   \
    BODY(dt_flag, HALF)    \
    BODY(dt_flag, FLOAT)   \
    BODY(dt_flag, DOUBLE)

#endif /* MILK_TYPE_DISPATCH_H */
