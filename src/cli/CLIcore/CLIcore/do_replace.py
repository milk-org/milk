import sys

with open('CLIcore_UI_prompt.c', 'r') as f:
    lines = f.readlines()

new_cli_expand_env = """void cli_expand_env(
    char *line,
    int   maxlen
)
{
    char out[STRINGMAXLEN_CLICMDLINE];
    int  opos = 0;
    int  i = 0;

    while(line[i] != '\\0' && opos < maxlen - 1)
    {
        if(line[i] == '$')
        {
            /* Skip $(( — arithmetic let wordexp handle it */
            if(line[i + 1] == '(' && line[i + 2] == '(')
            {
                out[opos++] = line[i++];
                out[opos++] = line[i++];
                out[opos++] = line[i++];
                continue;
            }
            /* Skip $( — command subst let wordexp handle it */
            if(line[i + 1] == '(')
            {
                out[opos++] = line[i++];
                out[opos++] = line[i++];
                continue;
            }

            i++;
            int has_brace = 0;
            if(line[i] == '{')
            {
                has_brace = 1;
                i++;
            }

            char varname[256];
            int  vlen = 0;

            while(line[i] != '\\0' && vlen < 255)
            {
                char c = line[i];
                if(!((c >= 'A' && c <= 'Z')
                     || (c >= 'a' && c <= 'z')
                     || (c >= '0' && c <= '9')
                     || c == '_'
                     || c == '?'
                     || c == '.'))
                {
                    break;
                }
                varname[vlen++] = line[i++];
                if(c == '?')
                {
                    break;
                }
            }
            varname[vlen] = '\\0';

            /* Check if this is a supported $VAR string, else just copy literal.
             * E.g. ${#foo} is not supported here, handled elsewhere. */
            if(has_brace)
            {
                if(line[i] == '}')
                {
                    i++;
                }
                else
                {
                    /* Something complex like ${var:-def}, let wordexp handle it.
                     * We output ${varname back into out and continue. */
                    out[opos++] = '$';
                    out[opos++] = '{';
                    for(int k=0; k<vlen; k++) {
                        if(opos < maxlen - 1) out[opos++] = varname[k];
                    }
                    continue;
                }
            }

            const char *val = cli_var_lookup(varname);
            if(val != NULL)
            {
                emit_str(out, &opos, maxlen, val);
            }
            else
            {
                /* Variable not found or milk doesn't know it. We let wordexp
                 * try to expand it later from environment! Just emit original string */
                out[opos++] = '$';
                if (has_brace) out[opos++] = '{';
                for(int k=0; k<vlen; k++) {
                    if(opos < maxlen - 1) out[opos++] = varname[k];
                }
                if (has_brace && opos < maxlen - 1) out[opos++] = '}';
            }
        }
        else if (line[i] == '\\\\' && line[i+1] == '$')
        {
            /* Let wordexp handle escaped dollars, just copy both characters verbatim */
            out[opos++] = line[i++];
            out[opos++] = line[i++];
        }
        else
        {
            out[opos++] = line[i++];
        }
    }
    out[opos] = '\\0';
    strncpy(line, out, (size_t) maxlen);
    line[maxlen - 1] = '\\0';
}
"""

idx_tilde = -1
idx_cmdsub_end = -1
idx_emit_str = -1
idx_expand_braced = -1
idx_cli_expand_env = -1

for i, line in enumerate(lines):
    if line.startswith('void cli_expand_tilde('):
        idx_tilde = i
    if line.startswith('void emit_str('):
        idx_emit_str = i
    if line.startswith('void expand_braced('):
        idx_expand_braced = i
    if line.startswith('void cli_expand_env('):
        idx_cli_expand_env = i

for i in range(idx_tilde - 1, -1, -1):
    if lines[i].startswith('/**'):
        idx_tilde = i
        break

for i in range(idx_emit_str - 1, -1, -1):
    if lines[i].startswith('/*'):
        idx_cmdsub_end = i - 1
        break

for i in range(idx_expand_braced - 1, -1, -1):
    if lines[i].startswith('/**'):
        idx_expand_braced = i
        break

new_lines = []
new_lines.extend(lines[:idx_tilde])
new_lines.extend(lines[idx_cmdsub_end+1:idx_expand_braced])
new_lines.append(new_cli_expand_env)

with open('CLIcore_UI_prompt.c', 'w') as f:
    f.writelines(new_lines)
