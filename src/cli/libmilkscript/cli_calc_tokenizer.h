/**
 * @file cli_calc_tokenizer.h
 * @brief Hand-written lexer for CLI expression parsing
 *
 * Replaces the flex-generated lexer (calc_flex.l).
 * Tokenizes CLI input into numbers, operators, function
 * names, and identifiers (variables/images/commands).
 */

#ifndef CLI_CALC_TOKENIZER_H
#define CLI_CALC_TOKENIZER_H

/**
 * @brief Maximum number of tokens per input line
 */
#define CLI_CALC_MAX_TOKENS 64

/**
 * @brief Maximum length of a token string value
 */
#define CLI_CALC_TOKEN_MAXLEN 200

/**
 * @brief Token type enumeration
 */
typedef enum
{
    TOK_EOF = 0,
    TOK_LONG,        /**< integer literal          */
    TOK_DOUBLE,      /**< floating-point literal    */
    TOK_FUNC_D_D,    /**< func(double)->double      */
    TOK_FUNC_DD_D,   /**< func(d,d)->double         */
    TOK_FUNC_DDD_D,  /**< func(d,d,d)->double       */
    TOK_FUNC_IM_D,   /**< func(image)->double        */
    TOK_FUNC_IMD_D,  /**< func(image,double)->double */
    TOK_FUNC_IMIM_D, /**< func(im,im)->double        */
    TOK_FUNC_WHERE,  /**< where(cond, a, b)          */
    TOK_VAR,         /**< existing variable          */
    TOK_IMAGE,       /**< existing image             */
    TOK_COMMAND,     /**< registered CLI command     */
    TOK_NVAR,        /**< new variable name          */
    TOK_OP_PLUS,
    TOK_OP_MINUS,
    TOK_OP_STAR,
    TOK_OP_SLASH,
    TOK_OP_CARET,
    TOK_OP_MOD,
    TOK_OP_LT,
    TOK_OP_LE,
    TOK_OP_GT,
    TOK_OP_GE,
    TOK_OP_EQ,
    TOK_OP_NEQ,
    TOK_OP_AND,
    TOK_OP_OR,
    TOK_OP_NOT,
    TOK_OP_PIPE,
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_COMMA,
    TOK_EQUAL,
    TOK_OP_PLUS_EQ,  /**< +=  */
    TOK_OP_MINUS_EQ, /**< -=  */
    TOK_OP_STAR_EQ,  /**< *=  */
    TOK_OP_SLASH_EQ, /**< /=  */
    TOK_OP_QUESTION, /**< ? (ternary) */
    TOK_OP_COLON,    /**< : (ternary) */
    TOK_FUNC_S_D,    /**< func(string)->double */
    /* Bitwise operators */
    TOK_OP_BAND,     /**< & (bitwise AND) */
    TOK_OP_BOR,      /**< | (bitwise OR)  */
    TOK_OP_BXOR,     /**< ^ (bitwise XOR) */
    TOK_OP_BNOT,     /**< ~ (bitwise NOT) */
    TOK_OP_LSHIFT,   /**< <<              */
    TOK_OP_RSHIFT,   /**< >>              */
    /* String functions */
    TOK_FUNC_S_S,    /**< func(string)->string */
    TOK_FUNC_SDD_S,  /**< func(s,d,d)->string  */
    TOK_FUNC_SSS_S,  /**< func(s,s,s)->string  */
    /* Format conversions */
    TOK_FUNC_D_S,    /**< func(double)->string */
    TOK_NEWLINE
} cli_token_type;

/**
 * @brief A single token produced by the lexer
 */
typedef struct
{
    cli_token_type type;
    long           val_l;
    double         val_d;
    double       (*fnctptr)();
    char           sval[CLI_CALC_TOKEN_MAXLEN];
} cli_token;

/**
 * @brief Tokenize an input string into an array of tokens
 *
 * @param input     null-terminated input string
 * @param tokens    output array (caller provides)
 * @param max_tok   size of tokens array
 * @return          number of tokens produced, or -1 on error
 */
int cli_tokenize(
    const char *input,
    cli_token  *tokens,
    int        max_tok);

#endif /* CLI_CALC_TOKENIZER_H */
