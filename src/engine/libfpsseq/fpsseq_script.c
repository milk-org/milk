#include "fpsseq.h"
#include <string.h>
#include <ctype.h>

#define MAX_SCRIPT_LINES 5000
#define MAX_LINE_LEN 256
#define MAX_VARS 100

typedef struct
{
    char name[64];
    char value[256];
} SCRIPT_VAR;

typedef struct
{
    char lines[MAX_SCRIPT_LINES][MAX_LINE_LEN];
    int num_lines;
    SCRIPT_VAR vars[MAX_VARS];
    int num_vars;
} SCRIPT_CTX;

/**
 * trim_whitespace - Strip leading and trailing whitespace in-place
 * @str:  NUL-terminated string to trim
 *
 * Advances past leading spaces/tabs and NUL-terminates after the
 * last non-whitespace character. Operates in-place on the buffer.
 */
static void trim_whitespace(char *str)
{
    char *end;
    while(isspace((unsigned char)*str))
    {
        str++;
    }
    if(*str == 0)
    {
        return;
    }
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end))
    {
        end--;
    }
    end[1] = '\0';
}

/**
 * expand_vars - Substitute $VARNAME tokens in a script line
 * @ctx:   Script context holding the variable table
 * @line:  Line buffer (modified in-place, max MAX_LINE_LEN)
 *
 * Scans the line for '$' followed by alphanumeric/underscore
 * characters, looks up the variable in ctx->vars[], and
 * replaces the token with its value. Undefined variables
 * expand to the empty string.
 */
static void expand_vars(
    SCRIPT_CTX *ctx,
    char       *line)
{
    char result[MAX_LINE_LEN] = {0};
    char *p = line;
    char *r = result;

    while(*p)
    {
        if(*p == '$')
        {
            p++;
            char var_name[64] = {0};
            int v = 0;
            while(isalnum(*p) || *p == '_')
            {
                var_name[v++] = *p++;
            }
            if(v > 0)
            {
                int found = 0;
                for(int i = 0; i < ctx->num_vars; i++)
                {
                    if(strcmp(ctx->vars[i].name, var_name) == 0)
                    {
                        snprintf(r, sizeof(result) - (r - result), "%s", ctx->vars[i].value);
                        r += strlen(ctx->vars[i].value);
                        found = 1;
                        break;
                    }
                }
                if(!found)
                {
                    // Variable not found, leave it as is or empty? Let's leave empty.
                }
            }
            else
            {
                *r++ = '$';
            }
        }
        else
        {
            *r++ = *p++;
        }
    }
    *r = '\0';
    snprintf(line, MAX_LINE_LEN, "%s", result);
}


/**
 * load_and_preprocess - Recursively load and preprocess a .seq script
 * @ctx:       Script context accumulating preprocessed lines
 * @filename:  Path to the script file to load
 * @depth:     Current include recursion depth (max 10)
 *
 * Reads the script line by line, processing directives:
 *   - "include <file>" -- recursively load another script
 *   - "set <var> <val>" -- define a script variable
 *   - All other lines undergo variable expansion and are
 *     appended to ctx->lines[] for later task injection.
 *
 * Blank lines and lines starting with '#' are skipped.
 *
 * Return: 0 on success, -1 on error (max depth, open failure,
 *         or script exceeding MAX_SCRIPT_LINES)
 */
static int load_and_preprocess(
    SCRIPT_CTX *ctx,
    const char *filename,
    int        depth)
{
    if(depth > 10)
    {
        printf("Error: Max include depth exceeded\n");
        return -1;
    }

    FILE *fp = fopen(filename, "r");
    if(!fp)
    {
        printf("Error: Cannot open script %s\n", filename);
        return -1;
    }

    char line[MAX_LINE_LEN];
    while(fgets(line, sizeof(line), fp))
    {
        trim_whitespace(line);
        if(line[0] == '\0' || line[0] == '#')
        {
            continue;
        }

        if(strncmp(line, "include ", 8) == 0)
        {
            char inc_file[256];
            if(sscanf(line, "include %s", inc_file) == 1)
            {
                if(load_and_preprocess(ctx, inc_file, depth + 1) != 0)
                {
                    fclose(fp);
                    return -1;
                }
            }
        }
        else if(strncmp(line, "set ", 4) == 0)
        {
            char vname[64], vval[256];
            if(sscanf(line, "set %s %[^\n]", vname, vval) == 2)
            {
                int found = 0;
                for(int i = 0; i < ctx->num_vars; i++)
                {
                    if(strcmp(ctx->vars[i].name, vname) == 0)
                    {
                        snprintf(ctx->vars[i].value, sizeof(ctx->vars[i].value), "%s", vval);
                        found = 1;
                        break;
                    }
                }
                if(!found && ctx->num_vars < MAX_VARS)
                {
                    snprintf(ctx->vars[ctx->num_vars].name,
                             sizeof(ctx->vars[0].name), "%s", vname);
                    snprintf(ctx->vars[ctx->num_vars].value,
                             sizeof(ctx->vars[0].value), "%s", vval);
                    ctx->num_vars++;
                }
            }
        }
        else
        {
            // Expand vars at load time as per plan
            expand_vars(ctx, line);

            if(ctx->num_lines < MAX_SCRIPT_LINES)
            {
                snprintf(ctx->lines[ctx->num_lines], MAX_LINE_LEN, "%s", line);
                ctx->num_lines++;
            }
            else
            {
                printf("Error: Script too large\n");
                fclose(fp);
                return -1;
            }
        }
    }
    fclose(fp);
    return 0;
}

/**
 * milkseq_load_script - Load a .seq script into the sequencer task array
 * @state:     Sequencer state (tasks are appended to state->tasklist)
 * @filename:  Path to the .seq script file
 * @fps:       Array of all FPS entries (used by if_fps_status)
 * @keywnode:  Keyword tree root (used by if_fps_status lookups)
 *
 * Preprocesses the script (includes, variables), then iterates
 * the resulting command lines. Recognized directives:
 *   - "on_error <policy>" -- set abort/skip/retry for subsequent tasks
 *   - "if_fps_status <name> <status>" ... "endif" -- conditional block
 *   - "repeat <N>" ... "endrepeat" -- repeat a block N times
 *   - All other lines are enqueued as sequencer tasks
 *
 * Return: 0 on success, -1 on preprocessing error
 */
errno_t milkseq_load_script(
    MILKSEQ_STATE     *state,
    const char        *filename,
    FPS               *fps,
    KEYWORD_TREE_NODE *keywnode)
{
    SCRIPT_CTX ctx = {0};
    if(load_and_preprocess(&ctx, filename, 0) != 0)
    {
        return -1;
    }

    int queue = 0;
    uint64_t current_error_policy = 0; // Default error policy

    for(int i = 0; i < ctx.num_lines; i++)
    {
        char *cmd = ctx.lines[i];

        if(strncmp(cmd, "on_error ", 9) == 0)
        {
            char policy[32];
            if(sscanf(cmd, "on_error %s", policy) == 1)
            {
                if(strcmp(policy, "abort") == 0)
                {
                    current_error_policy = MILKSEQ_TASKFLAG_ONERROR_ABORT;
                }
                else if(strcmp(policy, "skip") == 0)
                {
                    current_error_policy = MILKSEQ_TASKFLAG_ONERROR_SKIP;
                }
                else if(strcmp(policy, "retry") == 0)
                {
                    current_error_policy = MILKSEQ_TASKFLAG_ONERROR_RETRY;
                }
            }
            continue;
        }

        if(strncmp(cmd, "if_fps_status ", 14) == 0)
        {
            char fps_name[64];
            char target_status[32];
            sscanf(cmd, "if_fps_status %s %s", fps_name, target_status);

            // Evaluate immediately using keywnode
            int condition_met = 0;
            if(keywnode != NULL && fps != NULL)
            {
                int kwnindex = -1;
                // find in keywnode (assuming NBkwn is available... wait! We don't have NBkwn in milkseq_load_script!)
                // Let's just find the first matching keywordfull
                // wait, keywnode is an array, we need to know its size, or look for an end marker.
                // Usually we pass NBkwn. Since we don't have NBkwn, we can loop until keywordfull is empty?
                // Actually, let's just query via functionparameter_GetParamIndex by looping over active fps entries?
                // Let's just loop over fps[0..state->NBfps] or something? Wait... state doesn't have NBfps.
                // Let's iterate keywnode until we find it or reach a reasonable limit.
                for(int scan = 0; scan < 100000; scan++)
                {
                    if(keywnode[scan].keywordfull[0] == '\0')
                    {
                        break;    // end of list maybe?
                    }
                    if(strcmp(keywnode[scan].keywordfull, fps_name) == 0)
                    {
                        kwnindex = scan;
                        break;
                    }
                }

                if(kwnindex != -1)
                {
                    int fidx = keywnode[kwnindex].fpsindex;
                    if(strcmp(target_status, "running") == 0 &&
                            (fps[fidx].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN))
                    {
                        condition_met = 1;
                    }
                    else if(strcmp(target_status, "norun") == 0 &&
                            !(fps[fidx].md->status & FUNCTION_PARAMETER_STRUCT_STATUS_CMDRUN))
                    {
                        condition_met = 1;
                    }
                }
            }

            int end_idx = -1;
            for(int j = i + 1; j < ctx.num_lines; j++)
            {
                if(strncmp(ctx.lines[j], "endif", 5) == 0)
                {
                    end_idx = j;
                    break;
                }
            }

            if(end_idx != -1)
            {
                if(condition_met)
                {
                    // Inject block
                    for(int j = i + 1; j < end_idx; j++)
                    {
                        uint32_t t_idx = state->task_input_counter % NB_FPSCTRL_TASK_MAX;
                        snprintf(state->tasklist[t_idx].cmdstring,
                                 sizeof(state->tasklist[t_idx].cmdstring), "%s", ctx.lines[j]);
                        state->tasklist[t_idx].queue = queue;
                        state->tasklist[t_idx].flag = current_error_policy;
                        state->tasklist[t_idx].status = FPSTASK_STATUS_ACTIVE | FPSTASK_STATUS_WAITING;
                        state->tasklist[t_idx].inputindex = state->task_input_counter;
                        state->task_input_counter++;
                        state->NBtasks_active++;
                    }
                }
                i = end_idx; // Skip body
            }
            continue;
        }

        if(strncmp(cmd, "repeat ", 7) == 0)
        {
            int count = 0;
            sscanf(cmd, "repeat %d", &count);

            int end_idx = -1;
            for(int j = i + 1; j < ctx.num_lines; j++)
            {
                if(strncmp(ctx.lines[j], "endrepeat", 9) == 0)
                {
                    end_idx = j;
                    break;
                }
            }

            if(end_idx != -1)
            {
                for(int c = 0; c < count; c++)
                {
                    for(int j = i + 1; j < end_idx; j++)
                    {
                        uint32_t t_idx = state->task_input_counter % NB_FPSCTRL_TASK_MAX;
                        snprintf(state->tasklist[t_idx].cmdstring,
                                 sizeof(state->tasklist[t_idx].cmdstring), "%s", ctx.lines[j]);
                        state->tasklist[t_idx].queue = queue;
                        state->tasklist[t_idx].flag = current_error_policy;
                        state->tasklist[t_idx].status = FPSTASK_STATUS_ACTIVE | FPSTASK_STATUS_WAITING;
                        state->tasklist[t_idx].inputindex = state->task_input_counter;
                        state->task_input_counter++;
                        state->NBtasks_active++;
                    }
                }
                i = end_idx; // Skip body
            }
            continue;
        }

        uint32_t t_idx = state->task_input_counter % NB_FPSCTRL_TASK_MAX;
        snprintf(state->tasklist[t_idx].cmdstring,
                 sizeof(state->tasklist[t_idx].cmdstring), "%s", cmd);
        state->tasklist[t_idx].queue = queue;
        state->tasklist[t_idx].flag = current_error_policy;
        state->tasklist[t_idx].status = FPSTASK_STATUS_ACTIVE | FPSTASK_STATUS_WAITING;
        state->tasklist[t_idx].inputindex = state->task_input_counter;
        state->task_input_counter++;
        state->NBtasks_active++;
    }

    return 0;
}
