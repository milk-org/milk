import re

with open('execute_arith.c', 'r') as f:
    content = f.read()

# Chunk 1
c1_target = """#include "imfunctions.h"


#define ARITHTOKENTYPE_UNKNOWN  0
#define ARITHTOKENTYPE_NOTEXIST 1 // non-existing variable or image
#define ARITHTOKENTYPE_VARIABLE 2
#define ARITHTOKENTYPE_NUMBER   3
#define ARITHTOKENTYPE_OPERAND  4
#define ARITHTOKENTYPE_OPENPAR  5
#define ARITHTOKENTYPE_CLOSEPAR 6
#define ARITHTOKENTYPE_COMA     7
#define ARITHTOKENTYPE_FUNCTION 8
#define ARITHTOKENTYPE_EQUAL    9
#define ARITHTOKENTYPE_IMAGE    10
#define ARITHTOKENTYPE_MULTFUNC 11 // function of several variables/images, returning one variable/image"""

c1_rep = """#include "imfunctions.h"
#include "execute_arith_engines.h\""""

content = content.replace(c1_target, c1_rep)

# Chunk 2
c2_target = """            if(word_type[highest_priority_index] == ARITHTOKENTYPE_OPERAND)
            {
                /* Aliases to reduce line length */
                int hpi = highest_priority_index;
                CREATE_IMAGENAME(name,"""
c2_pattern = re.escape(c2_target) + r".*?nbword = nbword - 2;\n            }"
c2_match = re.search(c2_pattern, content, re.DOTALL)
if c2_match:
    c2_rep = """            if(word_type[highest_priority_index] == ARITHTOKENTYPE_OPERAND)
            {
                int hpi = highest_priority_index;
                CREATE_IMAGENAME(name,
                                 "_tmp%d_%d",
                                 tmp_name_index,
                                 (int) getpid());

                if (exec_arith_binary(word[hpi], 
                                      word_type[hpi - 1], word[hpi - 1], 
                                      word_type[hpi + 1], word[hpi + 1], 
                                      name, &type, &tmp_name_index) != 0)
                {
                    return RETURN_FAILURE;
                }

                snprintf(word[hpi - 1], sizeof(word[hpi - 1]), "%s", name);
                word_type[hpi - 1] = type;
                for(j = hpi; j < nbword - 2; j++)
                {
                    snprintf(word[j], sizeof(word[j]), "%s", word[j + 2]);
                    word_type[j] = word_type[j + 2];
                }
                nbword = nbword - 2;
            }"""
    content = content[:c2_match.start()] + c2_rep + content[c2_match.end():]

# Chunk 3
c3_target = """            if(word_type[highest_priority_index] == ARITHTOKENTYPE_FUNCTION)
            {
                CREATE_IMAGENAME(name,"""
c3_pattern = re.escape(c3_target) + r".*?nbword = nbword - 1;\n            }"
c3_match = re.search(c3_pattern, content, re.DOTALL)
if c3_match:
    c3_rep = """            if(word_type[highest_priority_index] == ARITHTOKENTYPE_FUNCTION)
            {
                CREATE_IMAGENAME(name,
                                 "_tmp%d_%d",
                                 tmp_name_index,
                                 (int) getpid());

                int hpi = highest_priority_index;
                if (exec_arith_unary(word[hpi], word_type[hpi + 1], word[hpi + 1], 
                                     name, &type, &tmp_name_index) != 0)
                {
                    return RETURN_FAILURE;
                }

                snprintf(word[highest_priority_index], sizeof(word[highest_priority_index]), "%s", name);
                word_type[highest_priority_index] = type;
                for(j = highest_priority_index + 1;
                    j < nbword - 1;
                    j++)
                {
                    snprintf(word[j], sizeof(word[j]), "%s", word[j + 1]);
                    word_type[j] = word_type[j + 1];
                }
                nbword = nbword - 1;
            }"""
    content = content[:c3_match.start()] + c3_rep + content[c3_match.end():]

# Chunk 4
c4_target = """            if(word_type[highest_priority_index] == ARITHTOKENTYPE_MULTFUNC)
            {
                nbvarinput = isfunction_sev_var(
                    word[highest_priority_index]);
                CREATE_IMAGENAME(name,"""
c4_pattern = re.escape(c4_target) + r".*?nbword = nbword - nbvarinput \* 2 - 1;\n            }"
c4_match = re.search(c4_pattern, content, re.DOTALL)
if c4_match:
    c4_rep = """            if(word_type[highest_priority_index] == ARITHTOKENTYPE_MULTFUNC)
            {
                nbvarinput = isfunction_sev_var(
                    word[highest_priority_index]);
                CREATE_IMAGENAME(name,
                                 "_tmp%d_%d",
                                 tmp_name_index,
                                 (int) getpid());

                int hpi = highest_priority_index;
                int a1t = 0, a2t = 0, a3t = 0;
                const char *a1w = NULL, *a2w = NULL, *a3w = NULL;

                if (nbvarinput >= 1) { a1t = word_type[hpi + 2]; a1w = word[hpi + 2]; }
                if (nbvarinput >= 2) { a2t = word_type[hpi + 4]; a2w = word[hpi + 4]; }
                if (nbvarinput >= 3) { a3t = word_type[hpi + 6]; a3w = word[hpi + 6]; }

                if (exec_arith_multfunc(word[hpi], nbvarinput, 
                                        a1t, a1w, 
                                        a2t, a2w, 
                                        a3t, a3w, 
                                        name, &type, &tmp_name_index) != 0)
                {
                    return RETURN_FAILURE;
                }

                snprintf(word[highest_priority_index], sizeof(word[highest_priority_index]), "%s", name);
                word_type[highest_priority_index] = type;
                for(j = highest_priority_index + 1;
                    j < nbword - (nbvarinput * 2 + 1);
                    j++)
                {
                    snprintf(word[j],
                           sizeof(word[j]),
                           "%s",
                           word[j + (nbvarinput * 2 + 1)]);
                    word_type[j] =
                        word_type[j + (nbvarinput * 2 + 1)];
                }
                nbword = nbword - nbvarinput * 2 - 1;
            }"""
    content = content[:c4_match.start()] + c4_rep + content[c4_match.end():]

with open('execute_arith.c', 'w') as f:
    f.write(content)

