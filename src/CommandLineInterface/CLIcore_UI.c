/**
 * @file CLIcore_UI.c
 * 
 * @brief User input (UI) functions
 *
 */



#include <readline/readline.h>
#include <readline/history.h>


#include "CommandLineInterface/CLIcore.h"






/**
 * @brief Readline callback
 *
 **/
void rl_cb_linehandler(char *linein)
{
    if(NULL == linein)
    {
        return;
    }

    data.CLIexecuteCMDready = 1;
 
    // copy input into data.CLIcmdline
    strcpy(data.CLIcmdline, linein);
 
    CLI_execute_line();

    free(linein);
}





errno_t runCLI_prompt(
    char *promptstring,
    char *prompt
)
{
	int color_cyan = 36;
	
	
    if(data.quiet == 0) {

        if(strlen(promptstring) > 0)
        {
            if(data.processnameflag == 0)
            {
                sprintf(prompt, "%c[%d;%dm%s >%c[%dm ", 0x1B, 1, color_cyan, promptstring, 0x1B, 0);
            }
            else
            {
                sprintf(prompt, "%c[%d;%dm%s-%s >%c[%dm ", 0x1B, 1, color_cyan, promptstring,
                        data.processname, 0x1B, 0);
            }
        }
        else
        {
            sprintf(prompt, "%c[%d;%dm%s >%c[%dm ", 0x1B, 1, color_cyan, data.processname, 0x1B, 0);
        }
    }
    else
    {
		sprintf(prompt," ");
    }



    return RETURN_SUCCESS;
}









static void *xmalloc(int size)
{
    void *buf;

    buf = malloc(size);
    if(!buf)
    {
        fprintf(stderr, "Error: Out of memory. Exiting.'n");
        exit(1);
    }

    return buf;
}



static char *dupstr(char *s)
{
    char *r;

    r = (char *) xmalloc((strlen(s) + 1));
    strcpy(r, s);
    return (r);
}




static char *CLI_generator(
    const char *text,
    int         state
)
{
    static unsigned int list_index;
    static unsigned int list_index1;
    static unsigned int len;
    char      *name;

	//printf("[generator %d %d %d]\n", state, data.CLImatchMode, list_index);

    if(!state)
    {
        list_index = 0;
        list_index1 = 0;
        len = strlen(text);
    }

    if(data.CLImatchMode == 0)
        while(list_index < data.NBcmd)
        {
            name = data.cmd[list_index].key;
            list_index++;
            if(strncmp(name, text, len) == 0)
            {
                return (dupstr(name));
            }
        }

    while(list_index1 < data.NB_MAX_IMAGE)
    {
        int iok;
        iok = data.image[list_index1].used;
        if(iok == 1)
        {
            name = data.image[list_index1].name;
            //	  printf("  name %d = %s %s\n", list_index1, data.image[list_index1].name, name);
        }
        list_index1++;
        if(iok == 1)
        {
            if(strncmp(name, text, len) == 0)
            {
                return (dupstr(name));
            }
        }
    }
    return ((char *)NULL);

}




/** @brief readline custom completion
 * 
 * Invoked when pressing TAB
 */

char **CLI_completion(
    const char *text,
    int start,
    int __attribute__((unused)) end
)
{
    char **matches;

	//printf("[%d %s]", start, rl_line_buffer);
	//rl_message("[%d %s]", start, rl_line_buffer);
	//rl_redisplay();
	//rl_forced_update_display();

    matches = (char **)NULL;

    if(start == 0)
    {
        data.CLImatchMode = 0;    // try to match string with command name
    }
    else
    {		
        data.CLImatchMode = 1;    // do not try to match with command
    }

    matches = rl_completion_matches((char *)text, &CLI_generator);

    //    else
    //  rl_bind_key('\t',rl_abort);

    return (matches);

}




