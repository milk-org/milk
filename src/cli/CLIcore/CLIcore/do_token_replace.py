import sys

with open('CLIcore_UI.c', 'r') as f:
    lines = f.readlines()

replacement = """        // extract first word
        // Replaced internal tokenization with POSIX wordexp to handle nested quotes safely
        
        cli_export_vars_for_wordexp(); // export variables prior to wordexp evaluation
        
        wordexp_t p;
        int we_ret = wordexp(data.CLIcmdline, &p, WRDE_SHOWERR | WRDE_UNDEF);
        if(we_ret == 0)
        {
            for(size_t i = 0; i < p.we_wordc; i++)
            {
                if (data.cmdNBarg >= MAX_NUMBER_ARG - 1) break;
                
                char *cmdargstring = p.we_wordv[i];
                
                if(data.cmdNBarg > 0
                   && data.cmdargtoken[0].type
                      == CMDARGTOKEN_TYPE_COMMAND
                   && (cmdargstring[0] == '-'
                       || cmdargstring[0] == '/'))
                {
                    strncpy(
                        data.cmdargtoken[data.cmdNBarg]
                            .val.string,
                        cmdargstring,
                        STRINGMAXLEN_CMDARGTOKEN_VAL - 1);
                    data.cmdargtoken[data.cmdNBarg]
                        .val.string[
                            STRINGMAXLEN_CMDARGTOKEN_VAL
                            - 1] = '\\0';
                    data.cmdargtoken[data.cmdNBarg]
                        .type = CMDARGTOKEN_TYPE_RAWSTRING;
                }
                else
                {
                    snprintf(str, strmaxlen,
                             "%s\\n", cmdargstring);
                    cli_parse(str);
                }
                data.cmdNBarg++;
            }
            wordfree(&p);
        }
        else
        {
            // Fallback if wordexp fails (e.g. WRDE_SYNTAX due to unmatched quotes)
            // It will trigger CMDARGTOKEN_TYPE_UNSOLVED which then correctly routes to bash transparently!
            strncpy(data.cmdargtoken[0].val.string, data.CLIcmdline, STRINGMAXLEN_CMDARGTOKEN_VAL - 1);
            data.cmdargtoken[0].val.string[STRINGMAXLEN_CMDARGTOKEN_VAL - 1] = '\\0';
            data.cmdargtoken[0].type = CMDARGTOKEN_TYPE_RAWSTRING;
            data.cmdNBarg = 1;
        }

        data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_UNSOLVED;
"""

idx_start = -1
idx_end = -1

for i, line in enumerate(lines):
    if line.strip() == '// extract first word':
        idx_start = i
    if line.strip() == 'data.cmdargtoken[data.cmdNBarg].type = CMDARGTOKEN_TYPE_UNSOLVED;':
        idx_end = i
        break

if idx_start != -1 and idx_end != -1:
    new_lines = lines[:idx_start] + [replacement] + lines[idx_end+1:]
    with open('CLIcore_UI.c', 'w') as f:
        f.writelines(new_lines)
    print("Replaced tokenization block successfully")
else:
    print(f"Error: Could not find block boundaries {idx_start} {idx_end}")
